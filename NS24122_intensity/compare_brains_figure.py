r"""
Coronal panels of every animal's raw tracer signal in atlas space, grouped by treatment.

Two windowings of the same data, because they answer different questions:

  absolute   one intensity window for every animal. Acquisition settings were identical across the
             cohort and Vehicle/30 mg animals were randomised within imaging day, so brightness
             differences here are differences in how much tracer is present. This is the view that
             corresponds to IVIS radiant efficiency, and the one every metric in the other notebooks
             normalises away.
  relative   each animal divided by its own background, (intensity - background) / background, then
             one common window. Removes any residual per-animal differences in laser/detector and
             in tissue autofluorescence, and shows only the distribution within each brain.

Where the two disagree is where "more tracer" and "tracer arranged differently" part company.

Voxels no sample reached (caudal brain outside an animal's imaged field, mostly) are drawn in grey,
not black, so missing data is not read as absence of signal.

Run (after atlas_space_images.py):
  python compare_brains_figure.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from fixed_threshold_intensity import PARENT, DATA_MAP, ATLAS_DIR, TRACER_DIRS, EXCLUDE_ANIMALS
from atlas_space_images import OUT_DIR, BIN, VOXEL_MM

import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
PLANES_MM = [1.5, 3.0, 4.5, 6.0, 7.2, 8.4, 9.4]      # olfactory bulb -> caudal midbrain, imaged in every animal
TREATMENT_ORDER = ['Vehicle', 'NS24122_10mg', 'NS24122_30mg']
GROUP_COLORS = {'Vehicle': '#7f7f7f', 'NS24122_10mg': '#4C72B0', 'NS24122_30mg': '#C1666B'}
DIVERGING = LinearSegmentedColormap.from_list('warm_cool', ['#4C72B0', '#F2F2F2', '#C1666B'])


def short(treatment):
    return treatment.replace('NS24122_', '')


def load_all():
    """Every animal's atlas-space volumes, plus the atlas outline on the same grid."""
    data_map = (pd.read_csv(DATA_MAP, encoding='utf-8-sig')
                  .rename(columns={'blinded_number': 'animal_number'}).set_index('animal_number'))
    animals = sorted([f[:-4] for f in os.listdir(OUT_DIR)
                      if f.endswith('.npz') and f[:-4] not in EXCLUDE_ANIMALS], key=lambda a: int(a[2:]))
    volumes = {a: dict(np.load(os.path.join(OUT_DIR, f'{a}.npz'))) for a in animals}
    treatment_of = {a: data_map.loc[a, 'treatment'] for a in animals}
    groups = {t: [a for a in animals if treatment_of[a] == t] for t in TREATMENT_ORDER}
    annotation = tifffile.imread(os.path.join(ATLAS_DIR, 'annotation.tiff'))
    outline = annotation[::BIN, ::BIN, ::BIN] > 0
    return volumes, treatment_of, groups, outline


def plane_image(volume, plane, mode, bg):
    """One coronal plane, as absolute intensity or as (intensity - background) / background."""
    img = volume[plane].astype(np.float32)
    missing = volume[plane] == 0
    if mode == 'relative':
        img = (img - bg) / bg
    img[missing] = np.nan
    return img


def window(volumes, groups, tracer, mode, n_voxels=400000):
    """Common display window across all animals: 0 to the 99.5th percentile of signal-bearing voxels."""
    rng = np.random.default_rng(0)
    sample = []
    for animal in sum(groups.values(), []):
        v = volumes[animal][tracer]
        vals = v[v > 0].astype(np.float32)
        if mode == 'relative':
            vals = (vals - volumes[animal][f'bg_{tracer}']) / volumes[animal][f'bg_{tracer}']
        sample.append(rng.choice(vals, size=min(n_voxels // 20, vals.size), replace=False))
    pooled = np.concatenate(sample)
    return (0.0, float(np.percentile(pooled, 99.5)))


def per_animal_figure(volumes, groups, outline, tracer, mode, planes):
    """Rows = animals grouped by treatment, columns = coronal planes."""
    animals = sum((groups[t] for t in TREATMENT_ORDER), [])
    vmin, vmax = window(volumes, groups, tracer, mode)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#3a3a3a')

    fig, axes = plt.subplots(len(animals), len(planes),
                             figsize=(2.05 * len(planes), 1.55 * len(animals)), squeeze=False)
    for row, animal in enumerate(animals):
        bg = volumes[animal][f'bg_{tracer}']
        for col, plane in enumerate(planes):
            ax = axes[row, col]
            im = ax.imshow(plane_image(volumes[animal][tracer], plane, mode, bg), cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation='nearest')
            if SHOW_OUTLINE:
                ax.contour(outline[plane], levels=[0.5], colors='#66CCFF', linewidths=0.35)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'{plane * BIN * VOXEL_MM:.1f} mm', fontsize=10)
        treatment = next(t for t in TREATMENT_ORDER if animal in groups[t])
        axes[row, 0].text(-0.05, 0.5, animal, transform=axes[row, 0].transAxes, ha='right', va='center',
                          fontsize=10, color=GROUP_COLORS[treatment], fontweight='bold')

    # a label and a rule for each treatment block
    y_top = 0.945
    for treatment in TREATMENT_ORDER:
        n = len(groups[treatment])
        height = n / len(animals) * 0.95
        fig.text(0.022, y_top - height / 2, f'{short(treatment)}  (n = {n})', rotation=90, va='center',
                 ha='center', fontsize=12, color=GROUP_COLORS[treatment], fontweight='bold')
        y_top -= height
        if treatment != TREATMENT_ORDER[-1]:
            fig.add_artist(plt.Line2D([0.06, 0.93], [y_top, y_top], color='#cccccc', linewidth=1))

    label = ('raw intensity (one window for every animal)' if mode == 'absolute'
             else '(intensity − background) / background, per animal')
    fig.subplots_adjust(left=0.10, right=0.93, top=0.945, bottom=0.01, wspace=0.02, hspace=0.05)
    cax = fig.add_axes([0.94, 0.35, 0.012, 0.3])
    fig.colorbar(im, cax=cax, label=label)
    fig.suptitle(f'{tracer}: coronal sections — {"ABSOLUTE" if mode == "absolute" else "PER-ANIMAL"} scaling\n'
                 f'{label}; grey = not imaged in that animal', x=0.10, y=0.997, va='top', ha='left', fontsize=13)
    path = os.path.join(FIG_DIR, f'{tracer}_per_animal_{mode}.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'wrote {path}')


def group_mean_figure(volumes, groups, outline, tracer, planes):
    """Group means under both windowings, plus the ratio of each dose to Vehicle."""
    means = {}
    for mode in ('absolute', 'relative'):
        for treatment in TREATMENT_ORDER:
            stack = []
            for animal in groups[treatment]:
                v = volumes[animal][tracer].astype(np.float32)
                img = np.stack([plane_image(volumes[animal][tracer], p, mode, volumes[animal][f'bg_{tracer}'])
                                for p in planes])
                stack.append(img)
            means[(mode, treatment)] = np.nanmean(np.stack(stack), axis=0)

    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#3a3a3a')
    rows = [('absolute', t) for t in TREATMENT_ORDER] + [('relative', t) for t in TREATMENT_ORDER]
    fig, axes = plt.subplots(len(rows) + 2, len(planes), figsize=(2.05 * len(planes), 1.7 * (len(rows) + 2)), squeeze=False)

    for row, (mode, treatment) in enumerate(rows):
        block = np.concatenate([means[(mode, t)].ravel() for t in TREATMENT_ORDER])
        vmax = float(np.nanpercentile(block, 99.5))
        for col, plane in enumerate(planes):
            ax = axes[row, col]
            im = ax.imshow(means[(mode, treatment)][col], cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            if SHOW_OUTLINE:
                ax.contour(outline[plane], levels=[0.5], colors='#66CCFF', linewidths=0.35)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'{plane * BIN * VOXEL_MM:.1f} mm', fontsize=10)
        axes[row, 0].text(-0.06, 0.5, f'{short(treatment)}\n{mode}', transform=axes[row, 0].transAxes,
                          ha='right', va='center', fontsize=9, color=GROUP_COLORS[treatment])
        if col == len(planes) - 1 and mode == 'absolute' and treatment == TREATMENT_ORDER[-1]:
            fig.colorbar(im, ax=axes[:3, :], fraction=0.012, pad=0.01, label='mean raw intensity')

    # ratio rows: each dose against Vehicle, on the absolute means
    for k, treatment in enumerate(TREATMENT_ORDER[1:]):
        for col, plane in enumerate(planes):
            ax = axes[len(rows) + k, col]
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.log2(means[('absolute', treatment)][col] / means[('absolute', 'Vehicle')][col])
            rim = ax.imshow(ratio, cmap=DIVERGING, vmin=-1.5, vmax=1.5, interpolation='nearest')
            if SHOW_OUTLINE:
                ax.contour(outline[plane], levels=[0.5], colors='#444444', linewidths=0.35)
            ax.set_axis_off()
        axes[len(rows) + k, 0].text(-0.06, 0.5, f'log2 {short(treatment)}\n/ Vehicle',
                                    transform=axes[len(rows) + k, 0].transAxes, ha='right', va='center', fontsize=9)
    fig.colorbar(rim, ax=axes[len(rows):, :], fraction=0.012, pad=0.01, label='log$_2$ ratio to Vehicle (absolute)')
    fig.suptitle(f'{tracer}: group means in atlas space — absolute scaling (top 3), per-animal scaling (middle 3), '
                 f'and the dose/Vehicle ratio (bottom 2)', x=0.02, ha='left', fontsize=13)
    path = os.path.join(FIG_DIR, f'{tracer}_group_means.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


SHOW_OUTLINE = True


def main():
    global SHOW_OUTLINE, FIG_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-outline', action='store_true', help='draw the sections without the atlas outline')
    parser.add_argument('--out', help='directory for the figures (default: figures/ next to this script)')
    args = parser.parse_args()
    SHOW_OUTLINE = not args.no_outline
    if args.out:
        FIG_DIR = args.out
    os.makedirs(FIG_DIR, exist_ok=True)
    volumes, treatment_of, groups, outline = load_all()
    planes = [int(round(mm / (BIN * VOXEL_MM))) for mm in PLANES_MM]
    print(f'{sum(len(v) for v in groups.values())} animals; planes {PLANES_MM} mm -> indices {planes}')
    for tracer in TRACER_DIRS:
        for mode in ('absolute', 'relative'):
            per_animal_figure(volumes, groups, outline, tracer, mode, planes)
        group_mean_figure(volumes, groups, outline, tracer, planes)


if __name__ == '__main__':
    main()
