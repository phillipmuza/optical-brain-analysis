r"""
Tracer signal as a function of depth below each animal's OWN brain surface, in its own image space.

No atlas, no deformation fields, no region parcellation, no cross-animal normalisation - see
README.md for why that matters. The measurement is the
closest light-sheet analogue of what the IVIS camera sees.

Two details that the design turns on:

  signed distance   Up to 17% of the thresholded tracer sits OUTSIDE the tissue mask, on the pial
                    surface and in the basal cisterns. Depth is therefore signed - negative outside
                    the brain, positive inside - and the histogram starts below zero, so that
                    compartment is measured rather than discarded.
  outer envelope    Depth is measured from an ENVELOPE, not from the tissue mask itself. Ventricles
                    that connect to the outside in 3D, the midline fissure and the cortex/cerebellum
                    gap would otherwise count as "surface" and make deep voxels look shallow. The
                    envelope seals them with a closing; the tight mask is still what defines tissue
                    for the background estimate.

Per animal x tracer x side (dorsal/ventral) it writes histograms over signed depth:
  n_voxels          voxels in that shell
  sum_above_bg      sum of (raw - background), the amount of signal, not clipped
  sum_in_mask       the same restricted to voxels above that animal's fixed threshold
  n_in_mask         how many those were
Wrapped (negative) voxels are excluded throughout; their count is recorded.

Run:
  python native_depth.py --qc an4     # one animal, prints checks, writes a figure
  python native_depth.py              # all animals -> depth/<animal>.npz
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

import config

HERE = os.path.dirname(os.path.abspath(__file__))

# Every value this script needs comes from config.<name>, resolved by config.resolve() when config is
# imported. Nothing may be snapshotted here at import time, and no function may take one as a
# default argument: both would freeze the dataset that was configured first.


def load_mask(animal):
    """The native-space brain mask built by 02_build_masks.py."""
    data = np.load(os.path.join(config.MASK_DIR, f'{animal}.npz'))
    shape = tuple(int(s) for s in data['shape'])
    return np.unpackbits(data['packed'])[:int(np.prod(shape))].reshape(shape).astype(bool)


def outer_envelope(mask, closing_radius=None):
    """
    The brain's outer surface: the mask with internal cavities and narrow clefts sealed.

    Closing is done with distance transforms rather than a ball structuring element, which is far
    cheaper at this radius, then holes are filled so that ventricles reaching the outside are closed.
    """
    if closing_radius is None:
        closing_radius = config.ENVELOPE_CLOSING_VOX
    pad = closing_radius + 2
    padded = np.pad(mask, pad)
    padded = ndimage.distance_transform_edt(~padded) <= closing_radius
    padded = ndimage.distance_transform_edt(padded) > closing_radius
    padded = ndimage.binary_fill_holes(padded)
    crop = tuple(slice(pad, pad + s) for s in mask.shape)
    return padded[crop]


def signed_depth(envelope):
    """Distance in mm from the envelope surface: positive inside the brain, negative outside."""
    inside = ndimage.distance_transform_edt(envelope).astype(np.float32)
    outside = ndimage.distance_transform_edt(~envelope).astype(np.float32)
    return (inside - outside) * np.float32(config.VOXEL_MM)


def dorsal_mask(envelope):
    """
    True where a voxel is in the dorsal half of its own column.

    Orientation is 'asr' (brainreg was run with --orientation asr), so axis 1 runs dorsal -> ventral
    and each (anterior-posterior, left-right) column is split at its own mid-height. The split
    follows the shape of the brain rather than one global plane, which matters because the brain is
    not flat.

    A column with no envelope voxels in it has no mid-height, so `mid` becomes inf and every voxel in
    that column counts as dorsal. That only ever applies outside the brain, where the thresholded
    measures have nothing anyway - but the all-voxel (sum_above_bg) histograms do include those
    voxels, so the quirk is recorded in tests rather than assumed away.
    """
    dv = np.arange(envelope.shape[1], dtype=np.float32)[None, :, None]
    filled = np.where(envelope, dv, np.nan)
    with np.errstate(invalid='ignore'):
        top = np.nanmin(filled, axis=1)
        bottom = np.nanmax(filled, axis=1)
    mid = np.where(np.isnan(top), np.inf, (top + bottom) / 2)
    return dv < mid[:, None, :]


def robust_background(values):
    median = float(np.median(values))
    return median, 1.4826 * float(np.median(np.abs(values - median)))


def animal_profile(animal):
    """Depth histograms for one animal, both tracers, both sides."""
    mask = load_mask(animal)
    envelope = outer_envelope(mask)
    depth = signed_depth(envelope)
    dorsal = dorsal_mask(envelope)

    edges = np.arange(config.DEPTH_LO, config.DEPTH_HI + config.DEPTH_STEP / 2, config.DEPTH_STEP,
                     dtype=np.float32)
    n_bins = len(edges) - 1
    result = {'edges': edges, 'mask_volume_mm3': float(mask.sum()) * config.VOXEL_MM3,
              'envelope_volume_mm3': float(envelope.sum()) * config.VOXEL_MM3,
              'dataset': config.DATASET_NAME, 'provenance': config.provenance()}

    for tracer, folder in config.CHANNELS.items():
        raw = tifffile.imread(config.image_path(animal, f'{folder}.tif'))
        valid_tissue = mask & (raw >= 0)
        bg, sd = robust_background(raw[valid_tissue].astype(np.float64))
        threshold = bg + config.K_MAD * sd

        counts = np.zeros(2 * n_bins)
        sums = np.zeros(2 * n_bins)
        counts_mask = np.zeros(2 * n_bins)
        sums_mask = np.zeros(2 * n_bins)
        for z0 in range(0, mask.shape[0], config.SLAB):
            z1 = min(z0 + config.SLAB, mask.shape[0])
            d = depth[z0:z1]
            r = raw[z0:z1]
            keep = (r >= 0) & (d >= config.DEPTH_LO) & (d < config.DEPTH_HI)
            if not keep.any():
                continue
            index = ((d - config.DEPTH_LO) / config.DEPTH_STEP).astype(np.int32)
            # keep already restricts d to the binned range, so this only guards against float
            # rounding at the top edge - but an index of n_bins makes bincount return a longer array,
            # which would shift every dorsal bin by one instead of failing
            np.clip(index, 0, n_bins - 1, out=index)
            index += np.where(dorsal[z0:z1], 0, n_bins)          # dorsal first, then ventral
            index = index[keep]
            above = (r[keep].astype(np.float32) - np.float32(bg))
            counts += np.bincount(index, minlength=2 * n_bins)
            sums += np.bincount(index, weights=above, minlength=2 * n_bins)
            hot = r[keep] > threshold
            counts_mask += np.bincount(index[hot], minlength=2 * n_bins)
            sums_mask += np.bincount(index[hot], weights=above[hot], minlength=2 * n_bins)

        for k, side in enumerate(('dorsal', 'ventral')):
            block = slice(k * n_bins, (k + 1) * n_bins)
            result[f'{tracer}_{side}_n_voxels'] = counts[block]
            result[f'{tracer}_{side}_sum_above_bg'] = sums[block]
            result[f'{tracer}_{side}_n_in_mask'] = counts_mask[block]
            result[f'{tracer}_{side}_sum_in_mask'] = sums_mask[block]
        result[f'{tracer}_background'] = np.float32(bg)
        result[f'{tracer}_threshold'] = np.float32(threshold)
        result[f'{tracer}_wrapped_voxels'] = np.int64((raw < 0).sum())
        del raw, valid_tissue
    return result


def qc(animal):
    """One animal: does the envelope look right, and where does the signal sit relative to the surface?"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t0 = time.time()
    mask = load_mask(animal)
    envelope = outer_envelope(mask)
    depth = signed_depth(envelope)
    dorsal = dorsal_mask(envelope)
    print(f'{animal}: mask {mask.sum() * config.VOXEL_MM3:.0f} mm3, '
          f'envelope {envelope.sum() * config.VOXEL_MM3:.0f} mm3 '
          f'(+{100 * (envelope.sum() / mask.sum() - 1):.1f}%), {time.time() - t0:.0f} s')
    print(f'  depth inside the brain: max {depth.max():.2f} mm; outside: min {depth.min():.2f} mm')
    print(f'  dorsal half holds {100 * dorsal[envelope].mean():.0f}% of envelope voxels')

    profile = animal_profile(animal)
    edges = profile['edges']
    centres = (edges[:-1] + edges[1:]) / 2
    for tracer in config.CHANNELS:
        total = sum(profile[f'{tracer}_{s}_sum_in_mask'].sum() for s in ('dorsal', 'ventral'))
        outside = sum(profile[f'{tracer}_{s}_sum_in_mask'][centres < 0].sum() for s in ('dorsal', 'ventral'))
        print(f'  {tracer}: background {profile[f"{tracer}_background"]:.0f}, '
              f'threshold {profile[f"{tracer}_threshold"]:.0f}; '
              f'{100 * outside / total:.1f}% of thresholded signal lies outside the surface')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    plane = mask.shape[0] // 2
    ax = axes[0]
    im = ax.imshow(np.clip(depth[plane], -0.4, 2.0), cmap='RdBu_r', vmin=-0.4, vmax=2.0, interpolation='nearest')
    ax.contour(envelope[plane], levels=[0.5], colors='black', linewidths=0.8)
    ax.contour(mask[plane], levels=[0.5], colors='#66CCFF', linewidths=0.6)
    ax.contour(dorsal[plane], levels=[0.5], colors='#7f7f7f', linewidths=0.6, linestyles=':')
    fig.colorbar(im, ax=ax, fraction=0.04, label='signed depth (mm)')
    ax.set_title(f'{animal}: signed depth, plane {plane}\nblack = envelope, cyan = mask, dotted = dorsal/ventral split',
                 fontsize=10, loc='left')
    ax.set_axis_off()

    for ax, tracer in zip(axes[1:], config.CHANNELS):
        for side, colour in (('dorsal', '#4C72B0'), ('ventral', '#C1666B')):
            ax.plot(centres, profile[f'{tracer}_{side}_sum_in_mask'], color=colour, label=f'{side}, thresholded')
            ax.plot(centres, profile[f'{tracer}_{side}_sum_above_bg'], color=colour, linestyle=':', alpha=0.7,
                    label=f'{side}, all voxels')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_yscale('symlog', linthresh=1e5)
        ax.set_xlim(config.DEPTH_LO, 2.0)
        ax.set_xlabel('signed depth below the brain surface (mm)', fontsize=10)
        ax.set_ylabel('integrated signal above background', fontsize=10)
        ax.set_title(f'{tracer}: where the signal sits', fontsize=10, loc='left')
        ax.legend(frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    os.makedirs(config.QC_DIR, exist_ok=True)
    path = os.path.join(config.QC_DIR, f'{animal}_native_depth.png')
    plt.savefig(path, dpi=120)
    print(f'  wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qc', metavar='ANIMAL', help='check one animal and write a figure')
    args = parser.parse_args()
    for line in config.validate():
        print(line)
    print(config.describe(), '\n', flush=True)

    if args.qc:
        qc(args.qc)
        return

    os.makedirs(config.DEPTH_DIR, exist_ok=True)
    data_map = config.load_data_map()
    listed = sorted([f[:-4] for f in os.listdir(config.MASK_DIR) if f.endswith('.npz')],
                    key=config.sort_key)
    # A mask can outlive the animal's place in the cohort - written before it was excluded, or by a
    # run with a different exclusion list. Analysing it anyway would put an animal into the
    # statistics that the data map and the README both say is not there.
    excluded = [a for a in listed if a in config.EXCLUDE_ANIMALS]
    animals = [a for a in listed if a not in excluded]
    if excluded:
        print(f'note: ignoring {excluded}: excluded for {config.DATASET_NAME}. '
              f'Delete their masks in {config.MASK_DIR} if that is not what you want.\n')
    missing = [a for a in config.animals() if a not in animals]
    if missing:
        print(f'no mask for {missing} - run 02_build_masks.py for this dataset first\n')
    start = time.time()
    for animal in animals:
        t0 = time.time()
        profile = animal_profile(animal)
        profile['treatment'] = str(data_map.loc[animal, 'treatment'])
        np.savez_compressed(os.path.join(config.DEPTH_DIR, f'{animal}.npz'), **profile)
        print(f'{animal}: {time.time() - t0:.0f} s', flush=True)
    print(f'\nwrote {len(animals)} depth profiles to {config.DEPTH_DIR} '
          f'in {(time.time() - start) / 60:.1f} min')
    print(f'manifest: {config.write_manifest("03_depth_profiles", {"profiles_written": len(animals)})}')


if __name__ == '__main__':
    main()
