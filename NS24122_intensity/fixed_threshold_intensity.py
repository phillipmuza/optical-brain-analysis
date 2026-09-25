r"""
Regional tracer intensity with one fixed threshold per animal.

Why this exists
---------------
The pipeline's tracer mask (tracer_segmentation.py) thresholds every image slice separately with a
hybrid Yen/Otsu rule. That makes the mask adapt to each slice's own contrast, so:
  * a slice with no tracer still gets a threshold, and so still gets "signal";
  * a uniform change in how much tracer is present largely cancels out - the threshold moves with it;
  * the mean intensity inside the mask is coupled to the mask (a lower threshold admits dimmer
    voxels, which drags the mean down), so mask-mean intensity partly measures the threshold.
Coverage from that pipeline therefore measures where locally-bright signal is, not how much tracer
there is. This script measures amount, with detection and quantification kept independent:

  measure in  atlas voxels that hold tissue (the registered atlas overhangs the sample at its
              edges; those dark voxels would otherwise subtract a background's worth of intensity
              from their region, by an amount that depends on registration fit - see tissue_mask)
  detect      one threshold for the whole brain of an animal/channel:
              T = background + k * 1.4826 * MAD, on the raw image by default (--detect unsharp
              thresholds the high-pass image instead: flatter across the brain, but it discards
              diffuse signal and needs the rectified scale estimator)
  quantify    on the raw 20 um image (downsampled/<channel>.tif), above that animal's own raw
              background, and NOT clipped at 0, so background noise cancels in the sum instead of
              accumulating (clipping inflated an4's FITC total by 86%)

Outputs, per animal x channel x atlas region x hemisphere (tidy CSV per animal in cache/):
  volume_mm3               tissue volume of the region in that hemisphere
  n_voxels_above           voxels above the fixed threshold           -> coverage
  integrated_in_mask       sum of (raw - background) over those voxels -> amount, mask-based
  integrated_all           sum of (raw - background) over every tissue voxel of the region,
                           mask-free (the measure that does not depend on the threshold at all)
  mean_in_mask             mean raw intensity of the above-threshold voxels (for reference only)
Group comparisons normalise each region to the animal's whole-brain total (share of that animal's
tracer), which cancels imaging gain and infusion success; the raw columns are kept as a reference.

Signal that registration places just outside the atlas is handled as the pipeline does it: voxels
within FILL_CAP_UM of the atlas are assigned to their nearest labelled region (--no-fill turns this
off and counts only voxels inside the registered atlas).

Run:
  python fixed_threshold_intensity.py --qc an4      # single animal + QC figures, writes nothing else
  python fixed_threshold_intensity.py               # all animals -> cache/<animal>_regions.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

PARENT = r'E:\tracer_uptake\wt_mice\new_analysis_0926'
DATA_MAP = r'E:\tracer_uptake\wt_mice\data_map.csv'
ATLAS_DIR = r'C:\Users\skgtpm1\.brainglobe\perens_lsfm_mouse_20um_v1.2'
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
QC_DIR = os.path.join(HERE, 'qc')

EXCLUDE_ANIMALS = ['an12', 'an14', 'an35',    # as in the IVIS analysis of this cohort
                   'an25', 'an26']            # experimental errors, dropped 16/09/2026
TRACER_DIRS = {'FITC': 'fitc', 'TxR': 'txr'}
HEMISPHERES = {'left': 1, 'right': 2}         # label values in registered_hemispheres.tiff
K_MAD = 5.0                                   # threshold = background + K_MAD robust SDs
K_SENSITIVITY = (3.0, 5.0, 8.0)               # reported by the QC run
DETECT = 'raw'                                # image the detection threshold is applied to: 'raw' or 'unsharp'
REGISTRATION_IMAGE = 'registration.tif'       # anatomy reference (a copy of one signal channel in this pipeline)
TISSUE_K = 2.0                                # tissue = reference > background - TISSUE_K robust SDs
FILL_CAP_UM = 200.0
VOXEL_UM = 20.0
VOXEL_MM3 = (VOXEL_UM / 1000) ** 3


def robust_background(values):
    """Median and robust SD (1.4826 * MAD) of a set of intensities."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, 1.4826 * mad


def rectified_scale(values):
    """
    Robust SD of a rectified, zero-centred distribution such as the saved unsharp image
    (max(median - gaussian, 0)), where about half the voxels are exactly 0 so the median and the
    MAD are both 0 and robust_background() degenerates. The upper quantiles are untouched by the
    clipping, so the scale is read off the quartile of the surviving half: Q75 / 0.6745.
    """
    return float(np.percentile(values, 75)) / 0.6745


def read_channel(animal, tracer):
    """Raw 20 um image (what the pipeline segmented) and the saved high-pass image, for one channel."""
    folder = TRACER_DIRS[tracer]
    raw = tifffile.imread(os.path.join(PARENT, animal, 'downsampled', f'{folder}.tif'))
    unsharp = tifffile.imread(os.path.join(PARENT, animal, folder, 'unsharp_image.tif'))
    return raw, unsharp


def tissue_mask(animal, brain):
    """
    Voxels of the registered atlas that actually hold tissue.

    The registered atlas overhangs the sample at the edges, and those voxels are far darker than
    tissue. Left in, each one subtracts roughly a background's worth of intensity from its region's
    total, and the amount of overhang varies between animals with how snugly the registration fits -
    which is itself group-related in this cohort. Tissue is defined on the registration image with a
    low threshold (TISSUE_K robust SDs *below* background), so it marks tissue rather than tracer.

    Returns:
    tuple: (tissue mask over the whole volume, fraction of atlas voxels that are tissue)
    """
    reference = tifffile.imread(os.path.join(PARENT, animal, 'downsampled', REGISTRATION_IMAGE))
    median, sd = robust_background(reference[brain].astype(np.float64))
    tissue = reference > (median - TISSUE_K * sd)
    return tissue, float(tissue[brain].mean())


def channel_threshold(image, brain, k=K_MAD, detect=DETECT):
    """
    One detection threshold for the whole brain of one animal x channel.

    Parameters:
    image (ndarray): raw image if detect == 'raw', the unsharp image if detect == 'unsharp'.
    detect (str): 'raw' thresholds the raw image at background + k robust SDs, which keeps the
        threshold in interpretable intensity units but ignores illumination gradients; 'unsharp'
        thresholds the high-pass image, which is flat across the brain but needs the rectified
        scale estimator and discards diffuse signal.

    Returns:
    tuple: (threshold, background level, robust SD).
    """
    values = image[brain].astype(np.float64)
    if detect == 'raw':
        median, sd = robust_background(values)
    else:
        median, sd = 0.0, rectified_scale(values)
    return median + k * sd, median, sd


def nearest_region_fill(labels, brain, extra):
    """
    Region label for voxels outside the registered atlas, as the pipeline assigns them: the label of
    the nearest labelled voxel, for voxels within FILL_CAP_UM of it.

    Parameters:
    labels (ndarray): registered atlas labels, 0 outside.
    brain (ndarray): labels > 0.
    extra (ndarray): boolean voxels outside the atlas to assign (already restricted to what matters).

    Returns:
    tuple: (coordinates kept, labels for them)
    """
    surface = brain & ~ndimage.binary_erosion(brain, border_value=1)
    surface_coords = np.argwhere(surface)
    tree = cKDTree(surface_coords)
    coords = np.argwhere(extra)
    dist, nearest = tree.query(coords, distance_upper_bound=FILL_CAP_UM / VOXEL_UM, workers=-1)
    keep = np.isfinite(dist)
    return coords[keep], labels[tuple(surface_coords[nearest[keep]].T)]


def region_table(animal, treatment, fill=True, k=K_MAD, detect=DETECT):
    """One row per channel x atlas region x hemisphere for an animal."""
    reg_dir = os.path.join(PARENT, animal, 'registration_dir')
    labels = tifffile.imread(os.path.join(reg_dir, 'registered_atlas.tiff'))
    hemi = tifffile.imread(os.path.join(reg_dir, 'registered_hemispheres.tiff'))
    brain = labels > 0
    tissue, tissue_fraction = tissue_mask(animal, brain)
    measured = brain & tissue          # atlas voxels that hold tissue: everything below is measured here
    unique_labels = np.unique(labels)  # sorted; label 0 included and dropped later by the volume > 0 filter

    rows, meta = [], []
    for tracer in TRACER_DIRS:
        raw, unsharp = read_channel(animal, tracer)
        assert raw.shape == labels.shape == hemi.shape == unsharp.shape, f'{animal}/{tracer}: shape mismatch'

        detect_image = raw if detect == 'raw' else unsharp
        threshold, bg_detect, sd_detect = channel_threshold(detect_image, measured, k, detect)
        mask = (detect_image > threshold) & tissue
        bg_raw, sd_raw = robust_background(raw[measured].astype(np.float64))
        # Not clipped at 0: noise then cancels in the sum, so the regional total is an unbiased
        # estimate of the tracer signal above background. Clipping would make every background
        # voxel contribute ~0.4 SD, which across a whole region swamps the real signal.
        above = raw.astype(np.float32) - bg_raw

        # voxels to count: inside the atlas, plus (optionally) tracer just outside it, assigned to
        # the nearest region exactly as the pipeline's signal lookup does
        pieces = [(labels[measured], hemi[measured], mask[measured], above[measured])]
        if fill:
            outside = mask & ~brain
            if outside.any():
                coords, filled_labels = nearest_region_fill(labels, brain, outside)
                idx = tuple(coords.T)
                pieces.append((filled_labels, hemi[idx], np.ones(len(coords), bool), above[idx]))

        # atlas ids are sparse and huge (a few hundred labels, ids up to ~6e8), so bincount over the id
        # itself would allocate an array per id value; index into the sorted unique labels instead
        n_labels = len(unique_labels)
        for name, hemi_id in HEMISPHERES.items():
            volume = np.zeros(n_labels)
            n_above = np.zeros(n_labels)
            integrated_mask = np.zeros(n_labels)
            integrated_all = np.zeros(n_labels)
            for lab, hem, msk, val in pieces:
                side = hem == hemi_id
                if not side.any():
                    continue
                lab_s, msk_s, val_s = np.searchsorted(unique_labels, lab[side]), msk[side], val[side]
                volume += np.bincount(lab_s, minlength=n_labels)
                lab_m, val_m = lab_s[msk_s], val_s[msk_s]
                n_above += np.bincount(lab_m, minlength=n_labels)
                integrated_mask += np.bincount(lab_m, weights=val_m, minlength=n_labels)
                integrated_all += np.bincount(lab_s, weights=val_s, minlength=n_labels)

            present = np.flatnonzero(volume > 0)
            rows.append(pd.DataFrame({
                'animal_number': animal, 'treatment': treatment, 'tracer': tracer, 'hemisphere': name,
                'id': unique_labels[present],
                'volume_mm3': volume[present] * VOXEL_MM3,
                'n_voxels_above': n_above[present],
                'coverage_pct': 100 * n_above[present] / volume[present],
                'integrated_in_mask': integrated_mask[present],
                'integrated_all': integrated_all[present],
                'mean_in_mask': np.divide(integrated_mask[present], n_above[present],
                                          out=np.full(len(present), np.nan), where=n_above[present] > 0) + bg_raw,
            }))

        meta.append({'animal_number': animal, 'treatment': treatment, 'tracer': tracer, 'k_mad': k,
                     'detect_on': detect, 'threshold': threshold, 'bg_detect': bg_detect, 'sd_detect': sd_detect,
                     'bg_raw': bg_raw, 'sd_raw': sd_raw, 'tissue_fraction_of_atlas': tissue_fraction,
                     'brain_voxels': int(brain.sum()), 'measured_voxels': int(measured.sum()),
                     'voxels_above': int((mask & measured).sum()),
                     'coverage_pct_brain': 100 * float((mask & measured).sum()) / float(measured.sum()),
                     'integrated_all_brain': float(above[measured].sum()),
                     'integrated_in_mask_brain': float(above[measured & mask].sum())})
        del raw, unsharp, mask, above
    return pd.concat(rows, ignore_index=True), pd.DataFrame(meta)


def qc(animal):
    """Single-animal check: thresholds, old vs new mask, per-slice behaviour, k sensitivity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(QC_DIR, exist_ok=True)
    data_map = (pd.read_csv(DATA_MAP, encoding='utf-8-sig')
                  .rename(columns={'blinded_number': 'animal_number'}).set_index('animal_number'))
    treatment = data_map.loc[animal, 'treatment']
    labels = tifffile.imread(os.path.join(PARENT, animal, 'registration_dir', 'registered_atlas.tiff'))
    brain = labels > 0
    tissue, tissue_fraction = tissue_mask(animal, brain)
    measured = brain & tissue
    print(f'{animal} ({treatment}): {brain.sum():,} atlas voxels of {brain.size:,}, volume {brain.sum() * VOXEL_MM3:.0f} mm3')
    print(f'  tissue inside the atlas: {100 * tissue_fraction:.1f}% ({measured.sum():,} voxels); '
          f'the rest is atlas overhanging the sample and is excluded\n')

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    for row, tracer in enumerate(TRACER_DIRS):
        raw, unsharp = read_channel(animal, tracer)
        pipeline_mask = tifffile.imread(os.path.join(PARENT, animal, TRACER_DIRS[tracer], 'thresholded_image.tif')) > 0
        bg_raw, sd_raw = robust_background(raw[measured].astype(np.float64))
        pipeline_coverage = 100 * (pipeline_mask & measured).sum() / measured.sum()

        print(f'--- {tracer}')
        print(f'  raw background (tissue): median {bg_raw:.0f}, robust SD {sd_raw:.0f}')
        print(f'  pipeline (per-slice adaptive) coverage: {pipeline_coverage:.3f}%')
        for mode, image in (('raw', raw), ('unsharp', unsharp)):
            for k in K_SENSITIVITY:
                t, bg, sd = channel_threshold(image, measured, k, mode)
                cov = 100 * ((image > t) & measured).sum() / measured.sum()
                marker = '  <- default' if (k == K_MAD and mode == DETECT) else ''
                print(f'  detect on {mode:7} k = {k:>3}: T = {t:8.1f} (bg {bg:.0f} + {k:g} x {sd:.1f})  coverage {cov:6.3f}%{marker}')

        # how much of a "total signal" is rectified noise if the sum is clipped at 0
        above = raw.astype(np.float32) - bg_raw
        total_unclipped = float(above[measured].sum())
        total_clipped = float(np.maximum(above[measured], 0).sum())
        print(f'  tissue total of (raw - background): unclipped {total_unclipped:.3e}, clipped at 0 {total_clipped:.3e} '
              f'-> clipping would add {100 * (total_clipped - total_unclipped) / total_clipped:.0f}% rectified noise')
        print(f'  (over the whole atlas instead of tissue only, the unclipped total would be {float(above[brain].sum()):.3e})')

        threshold, bg_d, sd_d = channel_threshold(raw if DETECT == 'raw' else unsharp, measured, K_MAD, DETECT)
        mask = ((raw if DETECT == 'raw' else unsharp) > threshold) & measured
        union = (mask | (pipeline_mask & measured)).sum()
        print(f'  default mask vs pipeline mask: {100 * (mask & pipeline_mask & measured).sum() / max(union, 1):.1f}% of their union shared; '
              f'new-only {100 * (mask & ~pipeline_mask).sum() / max(mask.sum(), 1):.1f}%, pipeline-only '
              f'{100 * (pipeline_mask & measured & ~mask).sum() / max((pipeline_mask & measured).sum(), 1):.1f}%')
        print(f'  signal captured: {100 * above[mask].sum() / total_unclipped:.0f}% of the tissue total sits in the new mask, '
              f'{100 * above[pipeline_mask & measured].sum() / total_unclipped:.0f}% in the pipeline mask')

        # (a) raw intensity in the brain, with both candidate thresholds
        ax = axes[row, 0]
        vals = raw[measured].astype(np.float64)
        ax.hist(vals, bins=np.linspace(np.percentile(vals, 0.1), np.percentile(vals, 99.9), 200), color='#999999')
        for k, colour in zip(K_SENSITIVITY, ['#4C72B0', '#C1666B', '#555555']):
            t, _, _ = channel_threshold(raw, measured, k, 'raw')
            ax.axvline(t, color=colour, lw=1.5, label=f'raw, k = {k:g}')
        ax.axvline(bg_raw, color='black', lw=1, ls=':', label='background')
        ax.set_yscale('log')
        ax.set_title(f'{tracer} (a) raw intensity in brain', fontsize=10, loc='left')
        ax.set_xlabel('raw intensity', fontsize=9)
        ax.legend(frameon=False, fontsize=8)

        # (b) per-slice coverage along the axis the pipeline thresholds on (axis 2): a single global
        # threshold should not drift across slices unless illumination does
        ax = axes[row, 1]
        denom = measured.sum(axis=(0, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            ax.plot(100 * (pipeline_mask & measured).sum(axis=(0, 1)) / denom, color='#C1666B', lw=1.2,
                    label=f'pipeline, per-slice ({pipeline_coverage:.2f}%)')
            for mode, image, colour in (('raw', raw, '#4C72B0'), ('unsharp', unsharp, '#2E8B57')):
                t, _, _ = channel_threshold(image, measured, K_MAD, mode)
                m = (image > t) & measured
                ax.plot(100 * m.sum(axis=(0, 1)) / denom, color=colour, lw=1.2,
                        label=f'fixed on {mode} ({100 * m.sum() / measured.sum():.2f}%)')
        ax.set_title(f'{tracer} (b) coverage per slice (threshold axis)', fontsize=10, loc='left')
        ax.set_xlabel('slice (axis 2)', fontsize=9)
        ax.set_ylabel('% of brain voxels in slice', fontsize=9)
        ax.legend(frameon=False, fontsize=8)

        # (c, d) two sections: raw image with both masks outlined
        planes = [int(brain.shape[2] * f) for f in (0.4, 0.6)]
        for col, plane in zip((2, 3), planes):
            ax = axes[row, col]
            img = raw[:, :, plane].astype(np.float32)
            hi = np.percentile(img[measured[:, :, plane]], 99.5) if measured[:, :, plane].any() else img.max()
            ax.imshow(img, cmap='gray', vmin=bg_raw * 0.8, vmax=max(hi, bg_raw + 1), interpolation='nearest')
            ax.contour(brain[:, :, plane], levels=[0.5], colors='#FFD166', linewidths=0.6)
            ax.contour(pipeline_mask[:, :, plane] & brain[:, :, plane], levels=[0.5], colors='#C1666B', linewidths=0.5)
            ax.contour(mask[:, :, plane], levels=[0.5], colors='#4C72B0', linewidths=0.5)
            ax.set_title(f'{tracer} slice {plane}', fontsize=10, loc='left')
            ax.set_axis_off()
        print()
        del raw, unsharp, pipeline_mask, mask

    fig.legend(handles=[Line2D([0], [0], color='#FFD166', label='registered atlas (brain)'),
                        Line2D([0], [0], color='#C1666B', label='pipeline mask (per-slice adaptive)'),
                        Line2D([0], [0], color='#4C72B0', label=f'fixed threshold on {DETECT}, k = {K_MAD:g}')],
               loc='lower center', ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f'{animal} ({treatment}): fixed per-animal threshold vs the pipeline\'s per-slice threshold',
                 x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = os.path.join(QC_DIR, f'{animal}_threshold_qc.png')
    plt.savefig(out, dpi=120)
    print(f'wrote {out}')

    # the region table itself, so the aggregation is checked too
    t0 = time.time()
    regions, meta = region_table(animal, treatment)
    print(f'\nregion table: {regions.shape[0]} rows in {time.time() - t0:.0f} s')
    with pd.option_context('display.width', 220):
        print(meta.to_string(index=False))
        acronym = {s['id']: s['acronym'] for s in json.load(open(os.path.join(ATLAS_DIR, 'structures.json')))}
        top = (regions[regions.tracer == 'TxR'].sort_values('integrated_all', ascending=False).head(8).copy())
        top['acronym'] = top['id'].map(acronym)
        print('\nlargest TxR regions by integrated intensity:')
        print(top[['acronym', 'hemisphere', 'volume_mm3', 'coverage_pct', 'integrated_in_mask', 'integrated_all']].round(2).to_string(index=False))
    # left + right volumes should reproduce the registration's own volumes.csv
    volumes = pd.read_csv(os.path.join(PARENT, animal, 'registration_dir', 'volumes.csv'))
    total = regions[regions.tracer == 'TxR'].groupby('hemisphere')['volume_mm3'].sum()
    print(f"\nvolume check: left {total['left']:.1f} mm3 vs volumes.csv {volumes['left_volume_mm3'].sum():.1f}, "
          f"right {total['right']:.1f} vs {volumes['right_volume_mm3'].sum():.1f} "
          "(small excess = tracer assigned from just outside the atlas)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qc', metavar='ANIMAL', help='check one animal and write QC figures, nothing else')
    parser.add_argument('--no-fill', action='store_true', help='count only voxels inside the registered atlas')
    parser.add_argument('--k', type=float, default=K_MAD, help=f'threshold in robust SDs above background (default {K_MAD:g})')
    parser.add_argument('--detect', choices=('raw', 'unsharp'), default=DETECT,
                        help=f'image the detection threshold is applied to (default {DETECT})')
    args = parser.parse_args()

    if args.qc:
        qc(args.qc)
        return

    # the no-fill variant is a sensitivity check, so it is kept separate from the main cache
    cache = CACHE if not args.no_fill else CACHE + '_nofill'
    summary_name = 'threshold_summary.csv' if not args.no_fill else 'threshold_summary_nofill.csv'
    os.makedirs(cache, exist_ok=True)
    data_map = (pd.read_csv(DATA_MAP, encoding='utf-8-sig')
                  .rename(columns={'blinded_number': 'animal_number'}).set_index('animal_number'))
    animals = sorted([a for a in os.listdir(PARENT)
                      if a.startswith('an') and os.path.isdir(os.path.join(PARENT, a, 'registration_dir'))
                      and a not in EXCLUDE_ANIMALS], key=lambda a: int(a[2:]))
    all_meta, start = [], time.time()
    for animal in animals:
        t0 = time.time()
        regions, meta = region_table(animal, data_map.loc[animal, 'treatment'], fill=not args.no_fill,
                                     k=args.k, detect=args.detect)
        regions.to_csv(os.path.join(cache, f'{animal}_regions.csv'), index=False)
        all_meta.append(meta)
        print(f'{animal}: {regions.shape[0]} rows, {time.time() - t0:.0f} s', flush=True)
    meta_path = os.path.join(HERE, summary_name)
    pd.concat(all_meta, ignore_index=True).to_csv(meta_path, index=False)
    print(f'\nwrote {len(animals)} region tables to {CACHE} and {meta_path} in {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    main()
