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

Inputs, produced by 05_atlas_space.py from this cohort's own registration (no other folder involved):
  config.ATLAS_SPACE_DIR     one npz per animal: tracer volumes on the atlas grid, each with
                             bg_<tracer>, plus 'n' for coverage
  config.THRESHOLD_SUMMARY   per animal x tracer threshold, as 05 computes them
  config.ATLAS_ANNOTATION    the annotation of config.ATLAS itself, in atlas space

Stage 05 must have run for this cohort first; this script says so rather than failing on an empty
path. Nothing else in the pipeline depends on either - 01-04 run with no atlas at all.

Outputs (figures/):
  compartment_maps_<tracer>.png   group-mean signal across the brain, with the surface/deep boundary
                                  drawn and the dose/reference ratio underneath
  coronal_profile.png             integrated signal per coronal plane, hindbrain -> olfactory bulb
  <tracer>_per_animal_absolute.png  one row per animal, one column per display plane, every animal
                                    in ONE intensity window: a brightness difference between animals
                                    is a difference in how much tracer arrived
  <tracer>_per_animal_relative.png  the same layout with each animal divided by its own background,
                                    which leaves only the distribution within each brain

Run:  python 06_atlas_maps.py            # the dataset in config.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy import ndimage

import config

HERE = os.path.dirname(os.path.abspath(__file__))

SMOOTH_SIGMA_VOX = 2.0          # 80 um: makes sparse above-threshold voxels legible as a density
CLOSING_RADIUS_VOX = 5                        # 100 um, the same envelope as everywhere else
PLANES_MM = [1.5, 3.0, 4.5, 6.0, 7.2, 8.4, 9.4]
DIVERGING = LinearSegmentedColormap.from_list('warm_cool', ['#4C72B0', '#F2F2F2', '#C1666B'])
# The per-animal montage: 0 to this percentile of the signal-bearing voxels pooled over the cohort,
# so every animal is drawn in one window. Grey, not black, where no sample voxel was placed - a
# region the animal did not image must not read as a region with no signal.
MONTAGE_PERCENTILE = 99.5
NOT_IMAGED = '#3a3a3a'
MONTAGE_DPI = 110


def compartments():
    """The two compartment descriptions, with the boundary taken from config.SURFACE_MM."""
    return {'surface': f'brain surface to {config.SURFACE_MM:g} mm depth',
            'deep': f'deeper than {config.SURFACE_MM:g} mm'}


def display_planes(brain):
    """
    The row indices of the display planes, for an atlas grid of this shape.

    A grid can be smaller than PLANES_MM assumes (a smaller atlas, or a bigger ATLAS_BIN), so keep
    the planes that exist and fall back to an even spread rather than indexing off the end.
    """
    planes = [int(round(mm / config.ATLAS_SPACING_MM)) for mm in PLANES_MM]
    planes = [p for p in planes if 0 <= p < brain.shape[0]]
    if not planes:
        planes = list(np.linspace(0, brain.shape[0] - 1, min(6, brain.shape[0])).astype(int))
    return planes


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
    animals = sorted([f[:-4] for f in os.listdir(config.ATLAS_SPACE_DIR)
                      if f.endswith('.npz') and f[:-4] not in config.EXCLUDE_ANIMALS],
                     key=config.sort_key)
    signal, covered, treatment_of = {}, {}, {}
    for animal in animals:
        data = dict(np.load(os.path.join(config.ATLAS_SPACE_DIR, f'{animal}.npz')))
        covered[animal] = data['n'] > 0
        treatment_of[animal] = config.treatment_of(animal)
        for tracer in config.CHANNELS:
            value = data[tracer].astype(np.float32)
            bg = float(data[f'bg_{tracer}'])
            threshold = float(thresholds.loc[(animal, tracer), 'threshold'])
            signal[(animal, tracer)] = np.where(value > threshold, value - bg, 0).astype(np.float32)
    groups = {t: [a for a in animals if treatment_of[a] == t] for t in config.GROUP_ORDER}
    return signal, covered, groups


def load_raw_planes(groups, planes):
    """
    Every animal's raw atlas-space intensity, on the display planes only.

    Read from 05's volumes with nothing applied to them: no threshold and no background subtraction.
    That is what the per-animal montage is for - 05 saves raw intensities precisely so a figure can
    window every animal together, and the per-animal view is the same data divided by the animal's
    own background. Only the planes that are drawn are kept, so eighteen animals cost megabytes here
    rather than the gigabytes the whole volumes would.
    """
    raw, covered, background = {}, {}, {}
    for animal in sum(groups.values(), []):
        with np.load(os.path.join(config.ATLAS_SPACE_DIR, f'{animal}.npz')) as data:
            covered[animal] = data['n'][planes] > 0
            for tracer in config.CHANNELS:
                raw[(animal, tracer)] = data[tracer][planes].astype(np.float32)
                background[(animal, tracer)] = float(data[f'bg_{tracer}'])
    return raw, covered, background


def montage_window(raw, background, groups, tracer, mode, n_voxels=20000):
    """
    The one display window the whole cohort is drawn in: 0 to the 99.5th percentile of the
    signal-bearing voxels pooled across animals.

    Pooled rather than per animal, and that is the whole point of the absolute view: a per-animal
    window would normalise away exactly the between-animal differences it exists to show. The
    subsample keeps the percentile cheap on a full atlas grid and is seeded, so the window does not
    move between runs.
    """
    rng = np.random.default_rng(0)
    sample = []
    for animal in sum(groups.values(), []):
        values = raw[(animal, tracer)].ravel()
        values = values[values > 0]
        if mode == 'relative':
            values = (values - background[(animal, tracer)]) / background[(animal, tracer)]
        if values.size:
            sample.append(rng.choice(values, size=min(n_voxels, values.size), replace=False))
    return 0.0, float(np.percentile(np.concatenate(sample), MONTAGE_PERCENTILE))


def per_animal_maps(raw, covered, background, groups, tracer, mode, planes):
    """
    One row per animal, grouped by treatment, one column per coronal plane.

    Two windowings of the same volumes, because they answer different questions:

      absolute   one intensity window for every animal. Acquisition settings were identical across
                 the cohort and the dose groups were randomised within imaging day, so a brightness
                 difference between rows is a difference in how much tracer arrived - the view that
                 corresponds to what the second modality measures, and the one every normalised
                 metric removes.
      relative   each animal divided by its own background, (intensity - background) / background,
                 then one common window. This takes out any residual per-animal difference in laser,
                 detector and tissue autofluorescence and leaves only the distribution within each
                 brain.

    Where the two disagree is where "more tracer" and "tracer arranged differently" part company.
    Drawn without the atlas outline, deliberately: the outline sits on the surface rim that carries
    the effect, and over these sections it obscures the thing being shown.
    """
    animals = sum((groups[t] for t in config.GROUP_ORDER), [])
    treatment_of = {a: t for t, members in groups.items() for a in members}
    vmin, vmax = montage_window(raw, background, groups, tracer, mode)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad(NOT_IMAGED)
    fig, axes = plt.subplots(len(animals), len(planes),
                             figsize=(2.05 * len(planes), 1.55 * len(animals)), squeeze=False)
    for row, animal in enumerate(animals):
        bg = background[(animal, tracer)]
        for col in range(len(planes)):
            ax = axes[row, col]
            image = raw[(animal, tracer)][col]
            if mode == 'relative':
                image = (image - bg) / bg
            ax.imshow(np.where(covered[animal][col], image, np.nan), cmap=cmap, vmin=vmin,
                      vmax=vmax, interpolation='nearest')
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'{planes[col] * config.ATLAS_SPACING_MM:.1f} mm', fontsize=10)
        axes[row, 0].text(-0.05, 0.5, animal, transform=axes[row, 0].transAxes, ha='right',
                          va='center', fontsize=10, color=config.GROUP_COLORS[treatment_of[animal]],
                          fontweight='bold')

    # one label and one rule per treatment block, so where a group starts is not something the
    # reader has to infer from the row labels
    y_top = 0.945
    for treatment in config.GROUP_ORDER:
        n = len(groups[treatment])
        height = n / len(animals) * 0.95
        fig.text(0.022, y_top - height / 2, f'{config.short(treatment)}  (n = {n})', rotation=90,
                 va='center', ha='center', fontsize=12, color=config.GROUP_COLORS[treatment],
                 fontweight='bold')
        y_top -= height
        if treatment != config.GROUP_ORDER[-1]:
            fig.add_artist(Line2D([0.06, 0.93], [y_top, y_top], color='#cccccc', linewidth=1))

    label = ('raw intensity (one window for every animal)' if mode == 'absolute'
             else '(intensity − background) / background, per animal')
    fig.subplots_adjust(left=0.10, right=0.93, top=0.945, bottom=0.01, wspace=0.02, hspace=0.05)
    cax = fig.add_axes((0.94, 0.35, 0.012, 0.3))
    # the window itself, not the last artist drawn: every panel was drawn in it, and saying so is
    # what the colour bar is for
    fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), cax=cax, label=label)
    fig.suptitle(f'{tracer}: coronal sections — '
                 f'{"ABSOLUTE" if mode == "absolute" else "PER-ANIMAL"} scaling\n'
                 f'{label}; grey = not imaged in that animal',
                 x=0.10, y=0.997, va='top', ha='left', fontsize=13)
    path = os.path.join(config.FIG_DIR, f'{tracer}_per_animal_{mode}.png')
    fig.savefig(path, dpi=MONTAGE_DPI)
    plt.close(fig)
    print(f'wrote {path}')


def ratio_to_reference(means, group, index):
    """
    One group's mean map over the reference group's, on one display plane, as log2.

    `index` is the position among the drawn planes, not the plane's index on the brain grid: means[] is
    built from the drawn planes only.

    Over the reference group and not over whichever group is listed first. For the three-group cohorts
    those were the same group; for a factorial cohort they need not be, and dividing by the wrong one
    would draw a perfectly plausible map of the wrong comparison.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.log2((means[group][index] + 1) / (means[config.REFERENCE_GROUP][index] + 1))


def compartment_maps(signal, covered, groups, brain, compartment_masks, tracer):
    """
    Group-mean signal across the brain, with the surface / deep boundary drawn on.

    One map per tracer rather than one per compartment: the compartments are contiguous, so a dashed
    line at SURFACE_MM below the surface shows the split without cutting the brain into two figures.
    Above-threshold voxels are sparse at this resolution, so the group mean is lightly smoothed
    within the brain - dividing by the smoothed validity mask keeps the edges from being pulled
    towards zero. It is a local density of tracer signal, not a new quantity.
    """
    planes = display_planes(brain)
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
    cmap.set_bad(NOT_IMAGED)
    # the ratio row goes to the reference group, not to whichever group happens to be listed first:
    # for a three-group cohort those were the same thing by luck, and for a 2x2 they need not be
    others = [t for t in config.GROUP_ORDER if t != config.REFERENCE_GROUP]
    n_rows = len(config.GROUP_ORDER) + len(others)
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
    for k, treatment in enumerate(others):
        for col, plane in enumerate(planes):
            ax = axes[len(config.GROUP_ORDER) + k, col]
            rim = ax.imshow(ratio_to_reference(means, treatment, col), cmap=DIVERGING, vmin=-2,
                            vmax=2, interpolation='nearest')
            boundary(ax, plane)
            ax.set_axis_off()
        axes[len(config.GROUP_ORDER) + k, 0].text(
            -0.05, 0.5, f'log2 {config.short(treatment)}\n/ {config.short(config.REFERENCE_GROUP)}',
            transform=axes[len(config.GROUP_ORDER) + k, 0].transAxes, rotation=90, ha='right',
            va='center', fontsize=9)

    fig.colorbar(im, ax=axes[:len(config.GROUP_ORDER), :], fraction=0.012, pad=0.01,
                 label='mean tracer signal above background')
    fig.colorbar(rim, ax=axes[len(config.GROUP_ORDER):, :], fraction=0.012, pad=0.01,
                 label=f'log$_2$ ratio to {config.short(config.REFERENCE_GROUP)}')
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

    fig, axes = plt.subplots(len(config.CHANNELS), len(compartments()), figsize=(14, 8.5),
                             sharex=True, squeeze=False)
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
    args = parser.parse_args()
    for line in config.validate():
        print(line)
    print(config.describe(), '\n', flush=True)

    if not os.path.isfile(config.THRESHOLD_SUMMARY) or not os.path.isdir(config.ATLAS_SPACE_DIR):
        raise SystemExit(
            f'this dataset has no atlas-space volumes yet, and this stage needs them.\n'
            f'  expected  {config.ATLAS_SPACE_DIR}/*.npz and {config.THRESHOLD_SUMMARY}\n'
            f'Run step 05 first:  python 05_atlas_space.py\n'
            f'It needs the brainglobe annotation of {config.ATLAS} (config.ATLAS_ANNOTATION) and '
            f'this dataset\'s brainreg registration_dir/. Steps 01-04 need no atlas and are '
            f'unaffected - this stage is optional.')
    volumes = [f for f in os.listdir(config.ATLAS_SPACE_DIR) if f.endswith('.npz')]
    if not volumes:
        raise SystemExit(f'{config.ATLAS_SPACE_DIR} holds no .npz volumes; step 05 has not produced '
                         f'anything for {config.DATASET_NAME}')

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

    # the per-animal montage, from the same volumes read raw: one window for the whole cohort in the
    # absolute view, each animal against its own background in the relative one
    planes = display_planes(brain)
    raw, plane_coverage, background = load_raw_planes(groups, planes)
    for tracer in config.CHANNELS:
        for mode in ('absolute', 'relative'):
            per_animal_maps(raw, plane_coverage, background, groups, tracer, mode, planes)
    print(f'manifest: {config.write_manifest("05_atlas_maps", {"groups": {t: len(m) for t, m in groups.items()}})}')


if __name__ == '__main__':
    main()