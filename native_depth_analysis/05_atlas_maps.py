r"""
Where the surface and the deep compartments sit across the brain, and how they change with dose.

Optional, and the only stage of this pipeline that needs an atlas. The native-space analysis (03, 04)
measures each animal against its own surface, which is the right instrument for *how much* tracer
there is, but it says nothing about *where* along the brain. For that the animals have to be on a
common grid, so these figures use the atlas-space resampled volumes named in config - on the Perens
LSFM 20 um atlas, binned by config.ATLAS_BIN for display - purely as a display alignment. The
compartments are defined by distance from the atlas brain surface, matching the surface/deep split
used in the native-space statistics:

  surface   <= SURFACE_MM below the brain surface
  deep      >  SURFACE_MM

Signal is the same quantity as in the statistics: raw intensity above that animal's background,
counted only where it exceeds that animal's fixed threshold.

Inputs, all per cohort in config.py:
  ATLAS_ANNOTATION    the annotation of config.ATLAS itself, in atlas space
  ATLAS_SPACE_DIR     one npz per animal: tracer volumes on the atlas grid, each with bg_<tracer>,
                      plus 'n' for coverage
  THRESHOLD_SUMMARY   per animal x tracer threshold, as 03 computes them

The resampling step that produces ATLAS_SPACE_DIR is NOT in this repository - it lives with the
intensity work (../NS24122_intensity/atlas_space_images.py) and has to be run there, per cohort,
before this stage can run. So this stage reproduces for a cohort whose atlas-space volumes already
exist, and not otherwise. config.validate() warns when those paths are absent; nothing else in the
pipeline depends on them.

Outputs (figures/):
  compartment_maps_<tracer>.png   group-mean signal across the brain, with the surface/deep boundary
                                  drawn and the dose/reference ratio underneath
  coronal_profile.png             integrated signal per coronal plane, hindbrain -> olfactory bulb

Run:  python 05_atlas_maps.py            # cohort from config / $NATIVE_DEPTH_COHORT
      python 05_atlas_maps.py --cohort anaesthetic
"""
import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

import config

HERE = os.path.dirname(os.path.abspath(__file__))

SMOOTH_SIGMA_VOX = 2.0          # 80 um: makes sparse above-threshold voxels legible as a density
CLOSING_RADIUS_VOX = 5                        # 100 um, the same envelope as everywhere else
PLANES_MM = [1.5, 3.0, 4.5, 6.0, 7.2, 8.4, 9.4]
DIVERGING = LinearSegmentedColormap.from_list('warm_cool', ['#4C72B0', '#F2F2F2', '#C1666B'])


def compartments():
    """The two compartment descriptions, with the boundary taken from config.SURFACE_MM."""
    return {'surface': f'brain surface to {config.SURFACE_MM:g} mm depth',
            'deep': f'deeper than {config.SURFACE_MM:g} mm'}


def atlas_compartments():
    """Brain mask and the surface / deep compartment masks on the binned atlas grid."""
    annotation = tifffile.imread(config.ATLAS_ANNOTATION)
    mask = annotation > 0
    r, pad = CLOSING_RADIUS_VOX, CLOSING_RADIUS_VOX + 2
    padded = np.pad(ndimage.binary_fill_holes(mask), pad)
    padded = ndimage.distance_transform_edt(~padded) <= r
    padded = ndimage.binary_fill_holes(ndimage.distance_transform_edt(padded) > r)
    crop = tuple(slice(pad, pad + s) for s in mask.shape)
    depth = (ndimage.distance_transform_edt(padded).astype(np.float32) * config.ATLAS_VOXEL_MM)[crop]

    brain = mask[::config.ATLAS_BIN, ::config.ATLAS_BIN, ::config.ATLAS_BIN]
    depth = depth[::config.ATLAS_BIN, ::config.ATLAS_BIN, ::config.ATLAS_BIN]
    return brain, {'surface': brain & (depth <= config.SURFACE_MM),
                   'deep': brain & (depth > config.SURFACE_MM)}


def load_signal():
    """Per animal x tracer: thresholded above-background signal on the atlas grid, plus coverage."""
    summary = pd.read_csv(config.THRESHOLD_SUMMARY)
    key_columns = [config.ANIMAL_COLUMN, 'tracer']
    if config.ANIMAL_COLUMN not in summary.columns:        # the intensity work uses this name
        key_columns = ['animal_number', 'tracer']
    missing = [c for c in key_columns + ['threshold'] if c not in summary.columns]
    if missing:
        raise SystemExit(f'{config.THRESHOLD_SUMMARY} has no column {missing}; it has '
                         f'{list(summary.columns)}. This table comes from the intensity work; point '
                         f'config.THRESHOLD_SUMMARY at it, or relax the column names here.')
    thresholds = summary.set_index(key_columns)
    data_map = config.load_data_map()
    animals = sorted([f[:-4] for f in os.listdir(config.ATLAS_SPACE_DIR)
                      if f.endswith('.npz') and f[:-4] not in config.EXCLUDE_ANIMALS],
                     key=config.sort_key)
    signal, covered, treatment_of = {}, {}, {}
    for animal in animals:
        data = dict(np.load(os.path.join(config.ATLAS_SPACE_DIR, f'{animal}.npz')))
        covered[animal] = data['n'] > 0
        treatment_of[animal] = data_map.loc[animal, 'treatment']
        for tracer in config.CHANNELS:
            value = data[tracer].astype(np.float32)
            bg = float(data[f'bg_{tracer}'])
            threshold = float(thresholds.loc[(animal, tracer), 'threshold'])
            signal[(animal, tracer)] = np.where(value > threshold, value - bg, 0).astype(np.float32)
    groups = {t: [a for a in animals if treatment_of[a] == t] for t in config.GROUP_ORDER}
    return signal, covered, groups


def compartment_maps(signal, covered, groups, brain, compartment_masks, tracer):
    """
    Group-mean signal across the brain, with the surface / deep boundary drawn on.

    One map per tracer rather than one per compartment: the compartments are contiguous, so a dashed
    line at SURFACE_MM below the surface shows the split without cutting the brain into two figures.
    Above-threshold voxels are sparse at this resolution, so the group mean is lightly smoothed
    within the brain - dividing by the smoothed validity mask keeps the edges from being pulled
    towards zero. It is a local density of tracer signal, not a new quantity.
    """
    planes = [int(round(mm / config.ATLAS_SPACING_MM)) for mm in PLANES_MM]
    means = {}
    for treatment in config.GROUP_ORDER:
        total = np.zeros(brain.shape, np.float32)
        count = np.zeros(brain.shape, np.float32)
        for animal in groups[treatment]:
            valid = brain & covered[animal]
            total += np.where(valid, signal[(animal, tracer)], 0)
            count += valid
        with np.errstate(invalid='ignore', divide='ignore'):
            mean3d = np.where(count > 0, total / np.maximum(count, 1), 0)
        weight = (count > 0).astype(np.float32)
        smoothed = ndimage.gaussian_filter(mean3d * weight, sigma=SMOOTH_SIGMA_VOX)
        norm = ndimage.gaussian_filter(weight, sigma=SMOOTH_SIGMA_VOX)
        with np.errstate(invalid='ignore', divide='ignore'):
            smoothed = np.where(norm > 0.05, smoothed / np.maximum(norm, 1e-6), np.nan)
        smoothed[~brain] = np.nan
        means[treatment] = np.stack([smoothed[p] for p in planes])

    vmax = float(np.nanpercentile(np.concatenate([m.ravel() for m in means.values()]), 99.5))
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#3a3a3a')
    n_rows = len(config.GROUP_ORDER) + len(config.GROUP_ORDER[1:])
    fig, axes = plt.subplots(n_rows, len(planes), figsize=(2.15 * len(planes), 1.85 * n_rows), squeeze=False)

    def boundary(ax, plane):
        """The SURFACE_MM contour: everything outside it is the surface compartment."""
        ax.contour(compartment_masks['deep'][plane], levels=[0.5], colors='#66CCFF',
                   linestyles='--', linewidths=0.8)

    for row, treatment in enumerate(config.GROUP_ORDER):
        for col, plane in enumerate(planes):
            ax = axes[row, col]
            im = ax.imshow(means[treatment][col], cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            boundary(ax, plane)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'{plane * config.ATLAS_SPACING_MM:.1f} mm', fontsize=10)
        axes[row, 0].text(-0.05, 0.5, config.short(treatment), transform=axes[row, 0].transAxes,
                          rotation=90, ha='right', va='center', fontsize=11,
                          color=config.GROUP_COLORS[treatment], fontweight='bold')
    for k, treatment in enumerate(config.GROUP_ORDER[1:]):
        for col, plane in enumerate(planes):
            ax = axes[len(config.GROUP_ORDER) + k, col]
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.log2((means[treatment][col] + 1) / (means[config.GROUP_ORDER[0]][col] + 1))
            rim = ax.imshow(ratio, cmap=DIVERGING, vmin=-2, vmax=2, interpolation='nearest')
            boundary(ax, plane)
            ax.set_axis_off()
        axes[len(config.GROUP_ORDER) + k, 0].text(
            -0.05, 0.5, f'log2 {config.short(treatment)}\n/ {config.short(config.GROUP_ORDER[0])}',
            transform=axes[len(config.GROUP_ORDER) + k, 0].transAxes, rotation=90, ha='right',
            va='center', fontsize=9)

    fig.colorbar(im, ax=axes[:len(config.GROUP_ORDER), :], fraction=0.012, pad=0.01,
                 label='mean tracer signal above background')
    fig.colorbar(rim, ax=axes[len(config.GROUP_ORDER):, :], fraction=0.012, pad=0.01,
                 label=f'log$_2$ ratio to {config.short(config.GROUP_ORDER[0])}')
    fig.suptitle(f'{tracer}: tracer signal across the brain — group means\n'
                 f'dashed line = {config.SURFACE_MM:g} mm below the surface; outside it is the surface '
                 'compartment, inside it the deep compartment', x=0.01, ha='left', fontsize=13)
    path = os.path.join(config.FIG_DIR, f'compartment_maps_{tracer}.png')
    fig.savefig(path, dpi=115, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def coronal_profile(signal, covered, groups, brain, compartment_masks):
    """Integrated signal per coronal plane, for each compartment, hindbrain -> olfactory bulb."""
    atlas_per_plane = brain.sum(axis=(1, 2)).astype(float)
    positions = np.arange(brain.shape[0]) * config.ATLAS_SPACING_MM
    rows, curves = [], {}
    for tracer in config.CHANNELS:
        for name, mask in compartment_masks.items():
            for animal, treatment in ((a, t) for t in config.GROUP_ORDER for a in groups[t]):
                sel = mask & covered[animal]
                per_plane = np.where(sel, signal[(animal, tracer)], 0).sum(axis=(1, 2))
                # a plane the animal did not image should be missing, not zero
                coverage = (brain & covered[animal]).sum(axis=(1, 2)) / np.maximum(atlas_per_plane, 1)
                per_plane = np.where(coverage > 0.5, per_plane, np.nan)
                curves[(tracer, name, animal)] = per_plane
                rows.append({'animal': animal, 'treatment': treatment, 'tracer': tracer, 'compartment': name,
                             'total': float(np.nansum(per_plane))})
    pd.DataFrame(rows).to_csv(os.path.join(config.OUT_DIR, 'coronal_profile_totals.csv'), index=False)

    fig, axes = plt.subplots(len(config.CHANNELS), len(compartments()), figsize=(14, 8.5), sharex=True)
    for r, tracer in enumerate(config.CHANNELS):
        for c, (name, description) in enumerate(compartments().items()):
            ax = axes[r, c]
            for treatment in config.GROUP_ORDER:
                stack = np.array([curves[(tracer, name, a)] for a in groups[treatment]])
                with np.errstate(invalid='ignore'):
                    mean = np.nanmean(stack, axis=0)
                    sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(stack), axis=0))
                ax.plot(positions, mean, color=config.GROUP_COLORS[treatment], linewidth=2,
                        label=config.short(treatment))
                ax.fill_between(positions, mean - sem, mean + sem, color=config.GROUP_COLORS[treatment],
                                alpha=0.2, linewidth=0)
            ax.set_title(f'{tracer}: {description}', fontsize=11, loc='left')
            if r == len(config.CHANNELS) - 1:
                ax.set_xlabel('position along the brain (mm from the anterior tip)', fontsize=10)
            if c == 0:
                ax.set_ylabel('integrated signal per coronal plane', fontsize=10)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
    # axis 0 of the atlas runs anterior -> posterior, so inverting it once (the axes share x) puts the
    # hindbrain on the left and the olfactory bulb on the right
    axes[0, 0].invert_xaxis()
    for ax in axes[-1]:
        ax.text(0.01, -0.16, '← hindbrain', transform=ax.transAxes, fontsize=9, color='#777777')
        ax.text(0.99, -0.16, 'olfactory bulb →', transform=ax.transAxes, ha='right', fontsize=9, color='#777777')
    fig.suptitle('How the tracer signal changes along the brain, by compartment (group mean ± SEM)',
                 x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    path = os.path.join(config.FIG_DIR, 'coronal_profile.png')
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cohort', help='cohort from config.COHORTS (default: $NATIVE_DEPTH_COHORT)')
    args = parser.parse_args()
    config.select(args.cohort)
    for line in config.validate():
        print(line)
    print(config.describe(), '\n', flush=True)

    os.makedirs(config.FIG_DIR, exist_ok=True)
    os.makedirs(config.OUT_DIR, exist_ok=True)
    brain, compartment_masks = atlas_compartments()
    print(f'atlas grid {brain.shape}; surface compartment '
          f'{100 * compartment_masks["surface"].sum() / brain.sum():.0f}% of brain voxels, '
          f'deep {100 * compartment_masks["deep"].sum() / brain.sum():.0f}%')
    signal, covered, groups = load_signal()
    for treatment, members in groups.items():
        print(f'{treatment} (n={len(members)})')
    for tracer in config.CHANNELS:
        compartment_maps(signal, covered, groups, brain, compartment_masks, tracer)
    coronal_profile(signal, covered, groups, brain, compartment_masks)


if __name__ == '__main__':
    main()