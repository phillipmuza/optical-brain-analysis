r"""
Where the surface and the deep compartments sit across the brain, and how they change with dose.

The native-space analysis (native_depth.py) measures each animal against its own surface, which is
the right instrument for *how much* tracer there is but says nothing about *where* along the brain.
For that the animals have to be on a common grid, so these figures use the atlas-space resampled
volumes (../NS24122_intensity/atlas_space, 40 um) purely as a display alignment. The compartments
are defined by distance from the atlas brain surface, matching the 0.5 mm split used in the
native-space statistics:

  surface   <= 0.5 mm below the brain surface
  deep      >  0.5 mm

Signal is the same quantity as in the statistics: raw intensity above that animal's background,
counted only where it exceeds that animal's fixed threshold.

Outputs (figures/):
  compartment_maps_<tracer>.png    group-mean signal across the brain, with the surface/deep
                                  boundary drawn and the dose/Vehicle ratio underneath
  coronal_profile.png              integrated signal per coronal plane, hindbrain -> olfactory bulb

Run:  python brain_maps.py
"""
import os
import sys

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
INTENSITY_DIR = os.path.join(os.path.dirname(HERE), 'NS24122_intensity')
sys.path.insert(0, INTENSITY_DIR)
from fixed_threshold_intensity import (ATLAS_DIR, DATA_MAP, TRACER_DIRS,     # noqa: E402
                                       EXCLUDE_ANIMALS)
from atlas_space_images import OUT_DIR as ATLAS_SPACE_DIR, BIN               # noqa: E402

FIG_DIR = os.path.join(HERE, 'figures')
OUT_DIR = os.path.join(HERE, 'outputs')
VOXEL_MM = 0.02
SURFACE_MM = 0.5
SMOOTH_SIGMA_VOX = 2.0          # 80 um: makes sparse above-threshold voxels legible as a density
CLOSING_RADIUS_VOX = 5                        # 100 um, the same envelope as everywhere else
PLANES_MM = [1.5, 3.0, 4.5, 6.0, 7.2, 8.4, 9.4]
TREATMENT_ORDER = ['Vehicle', 'NS24122_10mg', 'NS24122_30mg']
GROUP_COLORS = {'Vehicle': '#7f7f7f', 'NS24122_10mg': '#4C72B0', 'NS24122_30mg': '#C1666B'}
DIVERGING = LinearSegmentedColormap.from_list('warm_cool', ['#4C72B0', '#F2F2F2', '#C1666B'])
COMPARTMENTS = {'surface': f'brain surface to {SURFACE_MM:g} mm depth',
                'deep': f'deeper than {SURFACE_MM:g} mm'}


def short(treatment):
    return treatment.replace('NS24122_', '')


def atlas_compartments():
    """Brain mask and the surface / deep compartment masks on the binned atlas grid."""
    annotation = tifffile.imread(os.path.join(ATLAS_DIR, 'annotation.tiff'))
    mask = annotation > 0
    r, pad = CLOSING_RADIUS_VOX, CLOSING_RADIUS_VOX + 2
    padded = np.pad(ndimage.binary_fill_holes(mask), pad)
    padded = ndimage.distance_transform_edt(~padded) <= r
    padded = ndimage.binary_fill_holes(ndimage.distance_transform_edt(padded) > r)
    crop = tuple(slice(pad, pad + s) for s in mask.shape)
    depth = (ndimage.distance_transform_edt(padded).astype(np.float32) * VOXEL_MM)[crop]

    brain = mask[::BIN, ::BIN, ::BIN]
    depth = depth[::BIN, ::BIN, ::BIN]
    return brain, {'surface': brain & (depth <= SURFACE_MM), 'deep': brain & (depth > SURFACE_MM)}


def load_signal():
    """Per animal x tracer: thresholded above-background signal on the atlas grid, plus coverage."""
    thresholds = (pd.read_csv(os.path.join(INTENSITY_DIR, 'threshold_summary.csv'))
                    .set_index(['animal_number', 'tracer']))
    data_map = (pd.read_csv(DATA_MAP, encoding='utf-8-sig')
                  .rename(columns={'blinded_number': 'animal_number'}).set_index('animal_number'))
    animals = sorted([f[:-4] for f in os.listdir(ATLAS_SPACE_DIR)
                      if f.endswith('.npz') and f[:-4] not in EXCLUDE_ANIMALS], key=lambda a: int(a[2:]))
    signal, covered, treatment_of = {}, {}, {}
    for animal in animals:
        data = dict(np.load(os.path.join(ATLAS_SPACE_DIR, f'{animal}.npz')))
        covered[animal] = data['n'] > 0
        treatment_of[animal] = data_map.loc[animal, 'treatment']
        for tracer in TRACER_DIRS:
            value = data[tracer].astype(np.float32)
            bg = float(data[f'bg_{tracer}'])
            threshold = float(thresholds.loc[(animal, tracer), 'threshold'])
            signal[(animal, tracer)] = np.where(value > threshold, value - bg, 0).astype(np.float32)
    groups = {t: [a for a in animals if treatment_of[a] == t] for t in TREATMENT_ORDER}
    return signal, covered, groups


def compartment_maps(signal, covered, groups, brain, compartments, tracer):
    """
    Group-mean signal across the brain, with the surface / deep boundary drawn on.

    One map per tracer rather than one per compartment: the compartments are contiguous, so a dashed
    line at SURFACE_MM below the surface shows the split without cutting the brain into two figures.
    Above-threshold voxels are sparse at this resolution, so the group mean is lightly smoothed
    within the brain - dividing by the smoothed validity mask keeps the edges from being pulled
    towards zero. It is a local density of tracer signal, not a new quantity.
    """
    planes = [int(round(mm / (BIN * VOXEL_MM))) for mm in PLANES_MM]
    means = {}
    for treatment in TREATMENT_ORDER:
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
    n_rows = len(TREATMENT_ORDER) + len(TREATMENT_ORDER[1:])
    fig, axes = plt.subplots(n_rows, len(planes), figsize=(2.15 * len(planes), 1.85 * n_rows), squeeze=False)

    def boundary(ax, plane):
        """The SURFACE_MM contour: everything outside it is the surface compartment."""
        ax.contour(compartments['deep'][plane], levels=[0.5], colors='#66CCFF',
                   linestyles='--', linewidths=0.8)

    for row, treatment in enumerate(TREATMENT_ORDER):
        for col, plane in enumerate(planes):
            ax = axes[row, col]
            im = ax.imshow(means[treatment][col], cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            boundary(ax, plane)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'{plane * BIN * VOXEL_MM:.1f} mm', fontsize=10)
        axes[row, 0].text(-0.05, 0.5, short(treatment), transform=axes[row, 0].transAxes, rotation=90,
                          ha='right', va='center', fontsize=11, color=GROUP_COLORS[treatment], fontweight='bold')
    for k, treatment in enumerate(TREATMENT_ORDER[1:]):
        for col, plane in enumerate(planes):
            ax = axes[len(TREATMENT_ORDER) + k, col]
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.log2((means[treatment][col] + 1) / (means[TREATMENT_ORDER[0]][col] + 1))
            rim = ax.imshow(ratio, cmap=DIVERGING, vmin=-2, vmax=2, interpolation='nearest')
            boundary(ax, plane)
            ax.set_axis_off()
        axes[len(TREATMENT_ORDER) + k, 0].text(
            -0.05, 0.5, f'log2 {short(treatment)}\n/ {short(TREATMENT_ORDER[0])}',
            transform=axes[len(TREATMENT_ORDER) + k, 0].transAxes, rotation=90, ha='right', va='center', fontsize=9)

    fig.colorbar(im, ax=axes[:len(TREATMENT_ORDER), :], fraction=0.012, pad=0.01,
                 label='mean tracer signal above background')
    fig.colorbar(rim, ax=axes[len(TREATMENT_ORDER):, :], fraction=0.012, pad=0.01,
                 label=f'log$_2$ ratio to {short(TREATMENT_ORDER[0])}')
    fig.suptitle(f'{tracer}: tracer signal across the brain — group means\n'
                 f'dashed line = {SURFACE_MM:g} mm below the surface; outside it is the surface compartment, '
                 'inside it the deep compartment', x=0.01, ha='left', fontsize=13)
    path = os.path.join(FIG_DIR, f'compartment_maps_{tracer}.png')
    fig.savefig(path, dpi=115, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def coronal_profile(signal, covered, groups, brain, compartments):
    """Integrated signal per coronal plane, for each compartment, hindbrain -> olfactory bulb."""
    atlas_per_plane = brain.sum(axis=(1, 2)).astype(float)
    positions = np.arange(brain.shape[0]) * BIN * VOXEL_MM
    rows, curves = [], {}
    for tracer in TRACER_DIRS:
        for name, mask in compartments.items():
            for animal, treatment in ((a, t) for t in TREATMENT_ORDER for a in groups[t]):
                sel = mask & covered[animal]
                per_plane = np.where(sel, signal[(animal, tracer)], 0).sum(axis=(1, 2))
                # a plane the animal did not image should be missing, not zero
                coverage = (brain & covered[animal]).sum(axis=(1, 2)) / np.maximum(atlas_per_plane, 1)
                per_plane = np.where(coverage > 0.5, per_plane, np.nan)
                curves[(tracer, name, animal)] = per_plane
                rows.append({'animal': animal, 'treatment': treatment, 'tracer': tracer, 'compartment': name,
                             'total': float(np.nansum(per_plane))})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'coronal_profile_totals.csv'), index=False)

    fig, axes = plt.subplots(len(TRACER_DIRS), len(COMPARTMENTS), figsize=(14, 8.5), sharex=True)
    for r, tracer in enumerate(TRACER_DIRS):
        for c, (name, description) in enumerate(COMPARTMENTS.items()):
            ax = axes[r, c]
            for treatment in TREATMENT_ORDER:
                stack = np.array([curves[(tracer, name, a)] for a in groups[treatment]])
                with np.errstate(invalid='ignore'):
                    mean = np.nanmean(stack, axis=0)
                    sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(stack), axis=0))
                ax.plot(positions, mean, color=GROUP_COLORS[treatment], linewidth=2, label=short(treatment))
                ax.fill_between(positions, mean - sem, mean + sem, color=GROUP_COLORS[treatment],
                                alpha=0.2, linewidth=0)
            ax.set_title(f'{tracer}: {description}', fontsize=11, loc='left')
            if r == len(TRACER_DIRS) - 1:
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
    path = os.path.join(FIG_DIR, 'coronal_profile.png')
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    brain, compartments = atlas_compartments()
    print(f'atlas grid {brain.shape}; surface compartment {100 * compartments["surface"].sum() / brain.sum():.0f}% '
          f'of brain voxels, deep {100 * compartments["deep"].sum() / brain.sum():.0f}%')
    signal, covered, groups = load_signal()
    for treatment, members in groups.items():
        print(f'{treatment} (n={len(members)})')
    for tracer in TRACER_DIRS:
        compartment_maps(signal, covered, groups, brain, compartments, tracer)
    coronal_profile(signal, covered, groups, brain, compartments)


if __name__ == '__main__':
    main()
