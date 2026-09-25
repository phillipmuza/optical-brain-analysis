r"""
A brain mask built in the animal's own image space, with no atlas involved.

Why a new mask (see README.md, "Why not use the pipeline mask"):
  * the pipeline's mask_generation.py applies a per-slice CONVEX HULL, which cannot follow the
    concavities of a brain (cortex/cerebellum gap, midline fissure, ventral curvature) and comes out
    far larger than the brain - 1023 mm3 against 586 mm3 for the registered atlas in the ATX cohort;
  * the quick `tissue_mask()` written for the intensity work takes its background statistics from
    voxels *inside the registered atlas*, so it silently depends on the registration we are trying
    to avoid depending on.

This builds the mask from the image alone:

  threshold   Otsu on the intensity histogram, which is strongly bimodal (dark mounting medium vs
              tissue). Triangle and a background-peak rule are computed alongside for comparison.
  clean up    fill holes in 3D (ventricles and dark interior structures belong to the brain),
              keep the largest connected component (drops debris, dust and detached fragments),
              then a small morphological closing to smooth the boundary without convexifying it.

The mask is deliberately NOT convexified and NOT smoothed heavily: the whole point is to follow the
true surface, because every measurement in the native-space analysis is referenced to it.

Run:
  python 02_build_masks.py --qc an4 an36 an16   # compare masks on a few animals, write figures
  python 02_build_masks.py                      # all animals -> masks/<animal>.npz (bit-packed)

Add --cohort NAME to either to run a dataset other than the default (see config.py).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_triangle

import config

HERE = os.path.dirname(os.path.abspath(__file__))

# Every value this script needs comes from config.<name>, and config.select() resolves those in
# main() before any work starts. Nothing may be snapshotted here at import time, and no function
# may take one as a default argument: both would freeze whichever cohort was selected first.


def read_reference(animal):
    """The image the mask is built from (the registration channel, in the animal's own space)."""
    return tifffile.imread(config.image_path(animal, config.REFERENCE_IMAGE))


def candidate_thresholds(image):
    """
    Threshold candidates from the intensity histogram alone.

    Otsu splits the dark mounting medium from tissue and is the default. Triangle sits lower (it is
    built for a dominant background peak with a long tail) and is kept as a more inclusive
    alternative. The background-peak rule mirrors what mask_generation.py does, for comparison.
    """
    sample = image[::config.SUBSAMPLE, ::config.SUBSAMPLE, ::config.SUBSAMPLE].ravel()
    # 16-bit wraparound puts the brightest voxels at large negative values (see check_wraparound.py).
    # They are few, but they stretch the histogram range and wreck any bin-position rule, so every
    # threshold here is estimated on the valid (non-negative) voxels only.
    valid = sample[sample >= 0]
    otsu = float(threshold_otsu(valid))
    triangle = float(threshold_triangle(valid))
    hist, edges = np.histogram(valid, bins=256)
    peak = float(edges[np.argmax(hist[:50]) + 1]) * 1.001      # as in mask_generation._calculate_threshold
    hist_all, edges_all = np.histogram(sample, bins=256)        # what that rule returns if wrapping is ignored
    peak_unguarded = float(edges_all[np.argmax(hist_all[:50]) + 1]) * 1.001
    return {'otsu': otsu, 'triangle': triangle, 'background_peak': peak,
            'background_peak_unguarded': peak_unguarded}


def build_mask(image, threshold, closing_radius=None):
    """
    Threshold -> fill holes -> largest connected component -> small closing -> fill holes again.

    Returns:
    numpy.ndarray: boolean mask of the brain in the animal's own image space.
    """
    if closing_radius is None:
        closing_radius = config.MASK_CLOSING_VOX
    mask = image > threshold
    mask = ndimage.binary_fill_holes(mask)

    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(np.argmax(sizes))
    del labels

    if closing_radius > 0:
        # closing via distance transforms: much cheaper than a ball structuring element at this size
        pad = closing_radius + 1
        padded = np.pad(mask, pad)
        dilated = ndimage.distance_transform_edt(~padded) <= closing_radius
        padded = ndimage.distance_transform_edt(dilated) > closing_radius
        crop = tuple(slice(pad, pad + s) for s in mask.shape)
        mask = padded[crop]
    return ndimage.binary_fill_holes(mask)


def pipeline_mask(animal):
    """The mask the pipeline made (convex-hulled), if it is on disk."""
    path = os.path.join(config.DATA_DIR, animal, 'registration_dir', 'brain_mask.tif')
    if not os.path.exists(path):
        stem = config.CHANNELS.get('FITC', next(iter(config.CHANNELS.values()), 'fitc'))
        path = os.path.join(config.DATA_DIR, animal, stem, 'brain_mask.tif')
    return tifffile.imread(path) > 0 if os.path.exists(path) else None


def atlas_tissue_mask(animal, image):
    """
    Optional comparison masks, when an existing registration happens to be on disk: a threshold set
    from voxels inside the registered atlas, and the atlas itself. Returns (None, None) when there is
    no registration - this pipeline does not need one.
    """
    path = os.path.join(config.DATA_DIR, animal, 'registration_dir', 'registered_atlas.tiff')
    if not os.path.exists(path):
        return None, None
    labels = tifffile.imread(path)
    inside = labels > 0
    values = image[inside].astype(np.float64)
    median = float(np.median(values))
    sd = 1.4826 * float(np.median(np.abs(values - median)))
    return image > (median - 2 * sd), inside


def compare(animal):
    """Metrics for each candidate mask, plus how much tracer signal each one contains."""
    image = read_reference(animal)
    thresholds = candidate_thresholds(image)
    new = build_mask(image, thresholds['otsu'])
    quick, atlas = atlas_tissue_mask(animal, image)
    pipeline = pipeline_mask(animal)

    masks = {'pipeline (convex hull)': pipeline, 'quick threshold (atlas-based)': quick,
             'new (Otsu + largest component)': new, 'registered atlas': atlas}
    # an earlier tracer segmentation, if one exists, only to report how much of its signal each
    # mask contains; absent for a fresh dataset
    signal = {}
    for tracer, folder in config.CHANNELS.items():
        path = os.path.join(config.DATA_DIR, animal, folder, 'thresholded_image.tif')
        if os.path.exists(path):
            signal[tracer] = tifffile.imread(path) > 0

    n_wrapped = int((image < 0).sum())
    rows = []
    for name, mask in masks.items():
        if mask is None:
            continue
        row = {'animal': animal, 'mask': name, 'wrapped_voxels': n_wrapped,
               'volume_mm3': float(mask.sum()) * config.VOXEL_MM3}
        if atlas is not None:
            row['dice_with_atlas'] = float(2 * (mask & atlas).sum() / (mask.sum() + atlas.sum()))
            row['atlas_outside_mask_%'] = 100 * float((atlas & ~mask).sum()) / float(atlas.sum())
        for tracer, sig in signal.items():
            row[f'{tracer}_signal_inside_%'] = 100 * float((sig & mask).sum()) / float(sig.sum())
        rows.append(row)
    return image, thresholds, masks, pd.DataFrame(rows)


def qc_figure(animal, image, thresholds, masks, metrics):
    """Sections with every mask outlined, the histogram with each threshold, and the metrics table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colours = {'pipeline (convex hull)': '#C1666B', 'quick threshold (atlas-based)': '#E8A33D',
               'new (Otsu + largest component)': '#66CCFF', 'registered atlas': '#8ED081'}
    present = [name for name in colours if masks.get(name) is not None]

    shape = image.shape
    views = [('coronal', 0, int(shape[0] * 0.35)), ('coronal', 0, int(shape[0] * 0.6)),
             ('sagittal', 2, int(shape[2] * 0.5)), ('axial', 1, int(shape[1] * 0.5))]

    fig = plt.figure(figsize=(18, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])
    for k, (name, axis, index) in enumerate(views):
        ax = fig.add_subplot(grid[0, k])
        plane = np.take(image, index, axis=axis)
        hi = np.percentile(plane, 99.5)
        ax.imshow(plane, cmap='gray', vmin=0, vmax=max(hi, 1), interpolation='nearest')
        for mask_name in present:
            ax.contour(np.take(masks[mask_name], index, axis=axis), levels=[0.5],
                       colors=colours[mask_name], linewidths=0.9)
        ax.set_title(f'{name} {index}', fontsize=10)
        ax.set_axis_off()

    ax = fig.add_subplot(grid[1, :2])
    sample = image[::config.SUBSAMPLE, ::config.SUBSAMPLE, ::config.SUBSAMPLE].ravel()
    ax.hist(sample, bins=np.linspace(0, np.percentile(sample, 99.9), 200), color='#999999')
    for style, (label, value) in zip(['-', '--', ':', '-.'], thresholds.items()):
        ax.axvline(value, color='#333333', linestyle=style, linewidth=1.4, label=f'{label} = {value:.0f}')
    ax.set_yscale('log')
    ax.set_xlabel('intensity (registration channel)', fontsize=10)
    ax.set_ylabel('voxels', fontsize=10)
    ax.set_title('Intensity histogram and threshold candidates', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)

    ax = fig.add_subplot(grid[1, 2:])
    ax.axis('off')
    table = metrics.drop(columns='animal').round(2)
    ax.table(cellText=table.values, colLabels=[c.replace('_', ' ') for c in table.columns],
             loc='center', cellLoc='center').scale(1, 1.6)
    ax.set_title('Mask metrics', fontsize=11, loc='left')

    fig.legend(handles=[Line2D([0], [0], color=colours[n], lw=2, label=n) for n in present],
               loc='lower center', ncol=len(present), frameon=False, fontsize=10)
    fig.suptitle(f'{animal}: brain masks compared, in the animal\'s own image space', x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    os.makedirs(config.QC_DIR, exist_ok=True)
    path = os.path.join(config.QC_DIR, f'{animal}_mask_comparison.png')
    plt.savefig(path, dpi=115)
    plt.close(fig)
    print(f'  wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qc', nargs='+', metavar='ANIMAL', help='compare masks on these animals and write figures')
    parser.add_argument('--cohort', help='cohort from config.COHORTS (default: $NATIVE_DEPTH_COHORT)')
    args = parser.parse_args()
    config.select(args.cohort)
    for line in config.validate():
        print(line)
    print(config.describe(), '\n', flush=True)

    if args.qc:
        all_metrics = []
        for animal in args.qc:
            t0 = time.time()
            image, thresholds, masks, metrics = compare(animal)
            print(f'{animal}: thresholds ' + ', '.join(f'{k} {v:.0f}' for k, v in thresholds.items())
                  + f' ({time.time() - t0:.0f} s)')
            with pd.option_context('display.width', 200):
                print(metrics.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
            qc_figure(animal, image, thresholds, masks, metrics)
            all_metrics.append(metrics)
            del image, masks
        pd.concat(all_metrics, ignore_index=True).to_csv(os.path.join(config.QC_DIR, 'mask_comparison.csv'), index=False)
        return

    os.makedirs(config.MASK_DIR, exist_ok=True)
    data_map = config.load_data_map()
    animals = config.animals()
    rows, start = [], time.time()
    for animal in animals:
        t0 = time.time()
        image = read_reference(animal)
        thresholds = candidate_thresholds(image)
        mask = build_mask(image, thresholds['otsu'])
        np.savez_compressed(os.path.join(config.MASK_DIR, f'{animal}.npz'),
                            packed=np.packbits(mask), shape=np.array(mask.shape),
                            threshold=np.float32(thresholds['otsu']),
                            cohort=config.COHORT, provenance=config.provenance())
        rows.append({'animal_number': animal, 'treatment': data_map.loc[animal, 'treatment'],
                     'threshold_otsu': thresholds['otsu'], 'volume_mm3': float(mask.sum()) * config.VOXEL_MM3})
        print(f'{animal}: volume {rows[-1]["volume_mm3"]:.0f} mm3, {time.time() - t0:.0f} s', flush=True)
        del image, mask
    pd.DataFrame(rows).to_csv(os.path.join(config.MASK_DIR, 'mask_summary.csv'), index=False)
    print(f'\nwrote {len(animals)} masks to {config.MASK_DIR} in {(time.time() - start) / 60:.1f} min')
    print(f'manifest: {config.write_manifest("02_build_masks", {"masks_written": len(animals)})}')


if __name__ == '__main__':
    main()
