r"""
Resample each animal's raw tracer images into atlas space, so the same coronal plane is the same
anatomy in every animal and brains can be compared side by side.

Each sample voxel carries its atlas coordinate in the brainreg deformation fields. Every tissue voxel
is therefore dropped into the atlas voxel it maps to, and each atlas voxel takes the mean raw
intensity of the sample voxels that landed in it. Nothing is normalised or thresholded here: the
saved volumes are raw intensities on the atlas grid, so a figure can either window all animals
together (absolute, IVIS-comparable, valid because acquisition settings were identical) or window
each animal against its own background.

Voxels are kept wherever there is tissue, NOT only inside the registered atlas, so tracer sitting on
the pial surface and in the basal cisterns - the compartment IVIS mostly sees - stays visible as a
rim outside the atlas outline. Sample voxels whose atlas coordinate falls outside the atlas volume
are dropped rather than clipped, so nothing piles up at the edges.

Saved per animal (atlas_space/<animal>.npz), on the atlas grid binned by BIN (40 um by default):
  FITC, TxR   mean raw intensity per atlas voxel, uint16, 0 where no sample voxel landed
  n           how many sample voxels landed in each atlas voxel, uint8 (clipped), for QC
  bg_FITC, bg_TxR   that animal's raw background, for the per-animal scaling

Run:
  python atlas_space_images.py --qc an4     # one animal + a check figure
  python atlas_space_images.py              # all animals -> atlas_space/<animal>.npz
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tifffile

from fixed_threshold_intensity import (PARENT, DATA_MAP, ATLAS_DIR, EXCLUDE_ANIMALS, TRACER_DIRS,
                                       VOXEL_UM, robust_background, tissue_mask, read_channel)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'atlas_space')
QC_DIR = os.path.join(HERE, 'qc')
BIN = 2                      # atlas voxels per output voxel: 2 -> 40 um, small enough for figures
VOXEL_MM = VOXEL_UM / 1000


def binned_shape(shape, bin_size=BIN):
    return tuple(int(np.ceil(s / bin_size)) for s in shape)


def resample_animal(animal, atlas_shape, bin_size=BIN):
    """
    Mean raw intensity per (binned) atlas voxel for both channels of one animal.

    Returns:
    dict: 'FITC', 'TxR' (float32 mean intensity), 'n' (int32 sample voxels per atlas voxel),
          'bg_FITC', 'bg_TxR' (that animal's raw background).
    """
    reg_dir = os.path.join(PARENT, animal, 'registration_dir')
    labels = tifffile.imread(os.path.join(reg_dir, 'registered_atlas.tiff'))
    inside = labels > 0
    tissue, _ = tissue_mask(animal, inside)
    del labels

    # every tissue voxel, whether or not the registered atlas covers it
    flat_sample = np.flatnonzero(tissue.ravel())
    shape = binned_shape(atlas_shape, bin_size)

    # atlas coordinate of each of those voxels, one deformation field at a time
    index = np.zeros(len(flat_sample), dtype=np.int64)
    keep = np.ones(len(flat_sample), dtype=bool)
    stride = [int(np.prod(shape[i + 1:])) for i in range(3)]
    for axis in range(3):
        field = tifffile.imread(os.path.join(reg_dir, f'deformation_field_{axis}.tiff')).ravel()
        coord = np.rint(field[flat_sample] / VOXEL_MM / bin_size).astype(np.int64)
        del field
        keep &= (coord >= 0) & (coord < shape[axis])
        index += np.clip(coord, 0, shape[axis] - 1) * stride[axis]

    index = index[keep]
    n_bins = int(np.prod(shape))
    counts = np.bincount(index, minlength=n_bins)
    result = {'n': counts.reshape(shape).astype(np.int32)}
    for tracer in TRACER_DIRS:
        raw, _ = read_channel(animal, tracer)
        values = raw.ravel()[flat_sample][keep].astype(np.float64)
        bg, _ = robust_background(raw[inside & tissue].astype(np.float64))
        total = np.bincount(index, weights=values, minlength=n_bins)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(counts > 0, total / np.maximum(counts, 1), 0.0)
        result[tracer] = mean.reshape(shape).astype(np.float32)
        result[f'bg_{tracer}'] = np.float32(bg)
        del raw, values, total, mean
    return result


def save(animal, data):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, f'{animal}.npz'),
        FITC=np.clip(data['FITC'], 0, 65535).astype(np.uint16),
        TxR=np.clip(data['TxR'], 0, 65535).astype(np.uint16),
        n=np.clip(data['n'], 0, 255).astype(np.uint8),
        bg_FITC=data['bg_FITC'], bg_TxR=data['bg_TxR'])


def qc(animal, bin_size=BIN):
    """Resample one animal and check it looks like a brain in the right place."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    annotation = tifffile.imread(os.path.join(ATLAS_DIR, 'annotation.tiff'))
    t0 = time.time()
    data = resample_animal(animal, annotation.shape, bin_size)
    print(f'{animal}: resampled in {time.time() - t0:.0f} s; grid {data["n"].shape}')

    atlas_small = annotation[::bin_size, ::bin_size, ::bin_size] > 0
    covered = data['n'] > 0
    print(f'  atlas voxels with at least one sample voxel: {100 * covered[atlas_small].mean():.1f}%')
    print(f'  sample voxels per atlas voxel inside the brain: median {np.median(data["n"][atlas_small & covered]):.1f}')
    print(f'  signal-bearing voxels outside the atlas outline: {covered[~atlas_small].sum():,} '
          '(pial surface and cisterns, which the atlas does not cover)')
    for tracer in TRACER_DIRS:
        v = data[tracer]
        print(f'  {tracer}: background {data[f"bg_{tracer}"]:.0f}; mean inside brain {v[atlas_small & covered].mean():.0f}; '
              f'99.9th percentile {np.percentile(v[covered], 99.9):.0f}')

    planes = np.linspace(0.15, 0.85, 6) * atlas_small.shape[0]
    fig, axes = plt.subplots(2, 6, figsize=(17, 6.4))
    for row, tracer in enumerate(TRACER_DIRS):
        hi = np.percentile(data[tracer][covered], 99.5)
        for ax, p in zip(axes[row], planes.astype(int)):
            ax.imshow(data[tracer][p], cmap='magma', vmin=0, vmax=hi, interpolation='nearest')
            ax.contour(atlas_small[p], levels=[0.5], colors='#66CCFF', linewidths=0.6)
            ax.set_title(f'{tracer} plane {p * bin_size * VOXEL_MM:.1f} mm', fontsize=9)
            ax.set_axis_off()
    fig.suptitle(f'{animal}: raw intensity resampled into atlas space (blue = atlas brain outline)',
                 x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    os.makedirs(QC_DIR, exist_ok=True)
    out = os.path.join(QC_DIR, f'{animal}_atlas_space.png')
    plt.savefig(out, dpi=120)
    print(f'wrote {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qc', metavar='ANIMAL', help='resample one animal and write a check figure')
    parser.add_argument('--bin', type=int, default=BIN, help=f'atlas voxels per output voxel (default {BIN})')
    args = parser.parse_args()

    if args.qc:
        qc(args.qc, args.bin)
        return

    annotation_shape = tifffile.imread(os.path.join(ATLAS_DIR, 'annotation.tiff')).shape
    data_map = (pd.read_csv(DATA_MAP, encoding='utf-8-sig')
                  .rename(columns={'blinded_number': 'animal_number'}).set_index('animal_number'))
    animals = sorted([a for a in os.listdir(PARENT)
                      if a.startswith('an') and os.path.isdir(os.path.join(PARENT, a, 'registration_dir'))
                      and a not in EXCLUDE_ANIMALS], key=lambda a: int(a[2:]))
    assert all(a in data_map.index for a in animals)
    start = time.time()
    for animal in animals:
        t0 = time.time()
        save(animal, resample_animal(animal, annotation_shape, args.bin))
        print(f'{animal}: {time.time() - t0:.0f} s', flush=True)
    print(f'\nwrote {len(animals)} volumes to {OUT_DIR} in {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    main()
