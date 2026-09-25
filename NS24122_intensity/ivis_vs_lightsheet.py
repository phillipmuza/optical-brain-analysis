r"""
Does the light-sheet data agree with IVIS, animal by animal?

IVIS images the dorsal and the ventral surface of each fresh brain and reports one average radiant
efficiency per surface. Every light-sheet metric used so far is relative (share of an animal's total,
% of a region's volume), which cannot see a change in how much tracer arrived. Here the light-sheet
data is reduced the same way IVIS reduces it:

  take the atlas-space volumes (raw intensity, identical acquisition settings across the cohort),
  keep voxels within BAND_MM of the brain surface, split them into the dorsal and the ventral half
  by which side of the brain they face, and average - no thresholding, no region parcellation,
  no per-animal normalisation.

Then compare per animal against IVIS, which is the sharp test: group means can agree by coincidence,
but two independent modalities agreeing animal by animal cannot.

Outputs (outputs/):
  ivis_vs_lightsheet.csv          per animal x tracer x surface, both modalities
  ivis_vs_lightsheet_stats.csv    correlations and group comparisons
  ../figures/ivis_vs_lightsheet.png
"""
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fixed_threshold_intensity import PARENT, DATA_MAP, ATLAS_DIR, TRACER_DIRS
from atlas_space_images import OUT_DIR, BIN, VOXEL_MM
from penetration_intensity import envelope_and_depth

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
OUTPUT_DIR = os.path.join(HERE, 'outputs')
IVIS_CSV = os.path.join(os.path.dirname(PARENT), 'ivis_raw_data', 'ex_vivo', 'brain_data.csv')
BANDS_MM = (0.2, 0.5, 1.0)        # surface bands; IVIS light comes from a depth of order a millimetre
PRIMARY_BAND = 0.5
EDGE_SKIP_MM = 0.06               # drop the outermost voxels, which are part tissue and part air
LS_MEASURES = {'lightsheet_mean': 'avg_radiant_efficiency',          # light-sheet measure -> IVIS measure
               'lightsheet_above_bg': 'avg_radiant_efficiency',
               'lightsheet_p95': 'avg_radiant_efficiency',
               'lightsheet_integrated': 'total_radiant_efficiency',
               'lightsheet_in_mask_integrated': 'total_radiant_efficiency',
               'lightsheet_in_mask_fraction': 'avg_radiant_efficiency'}
TREATMENT_ORDER = ['Vehicle', 'NS24122_10mg', 'NS24122_30mg']
GROUP_COLORS = {'Vehicle': '#7f7f7f', 'NS24122_10mg': '#4C72B0', 'NS24122_30mg': '#C1666B'}


def short(treatment):
    return treatment.replace('NS24122_', '')


def surface_bands():
    """
    Dorsal and ventral surface masks on the binned atlas grid.

    A voxel belongs to the surface band if it lies within BAND_MM of the outer envelope. It is
    dorsal or ventral according to which side of its own column's mid-height it sits on, so the
    split follows the brain's shape rather than a single global plane (orientation 'asr': axis 1
    runs dorsal -> ventral).
    """
    annotation = tifffile.imread(os.path.join(ATLAS_DIR, 'annotation.tiff'))
    depth_full = envelope_and_depth(annotation > 0)
    depth = depth_full[::BIN, ::BIN, ::BIN]
    brain = (annotation > 0)[::BIN, ::BIN, ::BIN]

    # mid-height of the brain in each (anterior-posterior, left-right) column
    dv_index = np.arange(brain.shape[1])[None, :, None]
    masked = np.where(brain, dv_index, np.nan)
    with np.errstate(invalid='ignore'):
        top = np.nanmin(masked, axis=1)
        bottom = np.nanmax(masked, axis=1)
    mid = (top + bottom) / 2
    dorsal = brain & (dv_index < mid[:, None, :])
    ventral = brain & ~(dv_index < mid[:, None, :])
    return depth, brain, dorsal, ventral


def light_sheet_surface(volumes, depth, dorsal, ventral, band_mm):
    """
    Several reductions of the dorsal / ventral surface band, because they are not equivalent.

    `mean` is the raw average, which is what a naive reading of "average radiant efficiency" suggests
    - but tissue autofluorescence sets a floor of roughly the animal's own background, and the
    outermost voxels average tissue with the dark exterior, so a thin bright rim is diluted into
    that floor and the dynamic range collapses. Subtracting the animal's background (`above_bg`)
    removes the floor; summing rather than averaging (`integrated`) is the analogue of IVIS total
    radiant efficiency; and a high percentile (`p95`) measures the rim itself rather than the
    average of rim and parenchyma. The outermost EDGE_SKIP_MM is dropped from every measure to keep
    partial-volume voxels (half tissue, half air) out of the average.
    """
    band = (depth <= band_mm) & (depth > EDGE_SKIP_MM)
    thresholds = (pd.read_csv(os.path.join(HERE, 'threshold_summary.csv'))
                    .set_index(['animal_number', 'tracer'])['threshold'])
    rows = []
    for animal, data in volumes.items():
        covered = data['n'] > 0
        for tracer in TRACER_DIRS:
            values = data[tracer].astype(np.float32)
            bg = float(data[f'bg_{tracer}'])
            threshold = float(thresholds.loc[(animal, tracer)])
            for surface, mask in (('dorsal_brain', dorsal), ('ventral_brain', ventral)):
                sel = band & mask & covered
                v = values[sel]
                above = v - bg
                # the tracer-bearing voxels only: autofluorescence dominates the average, so a change
                # in dye is diluted ~20-fold in `mean`, while thresholding keeps the contrast that
                # IVIS has naturally (its baseline is dark, ours is autofluorescent tissue)
                in_mask = v > threshold
                rows.append({'blinded_number': animal, 'tracer': tracer, 'tissue': surface,
                             'band_mm': band_mm,
                             'lightsheet_mean': float(v.mean()),
                             'lightsheet_above_bg': float(above.mean()),
                             'lightsheet_integrated': float(above.sum()),
                             'lightsheet_p95': float(np.percentile(above, 95)),
                             'lightsheet_in_mask_integrated': float(above[in_mask].sum()),
                             'lightsheet_in_mask_fraction': float(in_mask.mean()),
                             'background': bg, 'threshold': threshold, 'n_voxels': int(sel.sum())})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    animals = sorted([f[:-4] for f in os.listdir(OUT_DIR) if f.endswith('.npz')], key=lambda a: int(a[2:]))
    volumes = {a: dict(np.load(os.path.join(OUT_DIR, f'{a}.npz'))) for a in animals}
    data_map = (pd.read_csv(DATA_MAP, encoding='utf-8-sig')
                  .rename(columns={'blinded_number': 'animal_number'}).set_index('animal_number'))

    depth, brain, dorsal, ventral = surface_bands()
    print(f'atlas grid {brain.shape}; dorsal half {dorsal.sum():,} voxels, ventral half {ventral.sum():,}')
    for band in BANDS_MM:
        sel = depth <= band
        print(f'  band <= {band} mm: {100 * (sel & brain).sum() / brain.sum():.0f}% of brain voxels')

    ls = pd.concat([light_sheet_surface(volumes, depth, dorsal, ventral, band) for band in BANDS_MM],
                   ignore_index=True)

    ivis = pd.read_csv(IVIS_CSV, encoding='utf-8-sig')
    ivis = ivis[ivis['tissue'].isin(['dorsal_brain', 'ventral_brain'])]
    merged = ls.merge(ivis[['blinded_number', 'tracer', 'tissue', 'avg_radiant_efficiency', 'total_radiant_efficiency']],
                      on=['blinded_number', 'tracer', 'tissue'], how='inner')
    merged['treatment'] = merged['blinded_number'].map(data_map['treatment'])
    merged.to_csv(os.path.join(OUTPUT_DIR, 'ivis_vs_lightsheet.csv'), index=False)
    n_animals = merged['blinded_number'].nunique()
    missing = sorted(set(animals) - set(merged['blinded_number']), key=lambda a: int(a[2:]))
    print(f'\n{n_animals} animals have both modalities' + (f'; no IVIS for {missing}' if missing else ''))

    # 1. do the two modalities agree animal by animal?
    rows = []
    for band in BANDS_MM:
        for tracer in TRACER_DIRS:
            for surface in ('dorsal_brain', 'ventral_brain'):
                d = merged[(merged['band_mm'] == band) & (merged['tracer'] == tracer) & (merged['tissue'] == surface)]
                for ls_measure, ivis_measure in LS_MEASURES.items():
                    rho, p_rho = stats.spearmanr(d[ls_measure], d[ivis_measure])
                    rows.append({'band_mm': band, 'tracer': tracer, 'surface': surface, 'n': len(d),
                                 'lightsheet_measure': ls_measure.replace('lightsheet_', ''),
                                 'ivis_measure': ivis_measure.replace('_radiant_efficiency', ''),
                                 'spearman_rho': rho, 'p_spearman': p_rho})
    correlations = pd.DataFrame(rows)
    print('\nAgreement between modalities, per animal:')
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200):
        print(correlations[correlations['band_mm'] == PRIMARY_BAND].to_string(index=False))
        print('\nbest band for each light-sheet measure (by mean |rho| over tracer x surface):')
        print(correlations.groupby(['lightsheet_measure', 'band_mm'])['spearman_rho'].mean().unstack().round(3).to_string())

    # 2. the same group comparison in both modalities, side by side
    rows = []
    for tracer in TRACER_DIRS:
        for surface in ('dorsal_brain', 'ventral_brain'):
            d = merged[(merged['band_mm'] == PRIMARY_BAND) & (merged['tracer'] == tracer) & (merged['tissue'] == surface)]
            for measure, column in (('IVIS avg radiant efficiency', 'avg_radiant_efficiency'),
                                    ('IVIS total radiant efficiency', 'total_radiant_efficiency'),
                                    ('light-sheet surface mean (raw)', 'lightsheet_mean'),
                                    ('light-sheet mean above background', 'lightsheet_above_bg'),
                                    ('light-sheet 95th pct above background', 'lightsheet_p95'),
                                    ('light-sheet integrated above background', 'lightsheet_integrated'),
                                    ('light-sheet integrated, tracer voxels only', 'lightsheet_in_mask_integrated'),
                                    ('light-sheet % of band above threshold', 'lightsheet_in_mask_fraction')):
                row = {'tracer': tracer, 'surface': surface, 'measure': measure}
                veh = d.loc[d['treatment'] == 'Vehicle', column]
                for treatment in TREATMENT_ORDER:
                    row[short(treatment)] = d.loc[d['treatment'] == treatment, column].mean()
                for treatment in TREATMENT_ORDER[1:]:
                    dose = d.loc[d['treatment'] == treatment, column]
                    row[f'ratio_{short(treatment)}'] = dose.mean() / veh.mean()
                    row[f'p_{short(treatment)}'] = stats.ttest_ind(dose, veh, equal_var=False)[1]
                rows.append(row)
    group_tests = pd.DataFrame(rows)
    print(f'\nGroup comparison in both modalities (light-sheet band = {PRIMARY_BAND} mm):')
    with pd.option_context('display.float_format', '{:.4g}'.format, 'display.width', 220):
        print(group_tests.to_string(index=False))

    pd.concat([correlations.assign(table='correlation'), group_tests.assign(table='group_test')],
              ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, 'ivis_vs_lightsheet_stats.csv'), index=False)

    # figure: per-animal agreement, and both modalities' group means
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for row, tracer in enumerate(TRACER_DIRS):
        for col, surface in enumerate(('dorsal_brain', 'ventral_brain')):
            ax = axes[row, col]
            d = merged[(merged['band_mm'] == PRIMARY_BAND) & (merged['tracer'] == tracer) & (merged['tissue'] == surface)]
            for treatment in TREATMENT_ORDER:
                g = d[d['treatment'] == treatment]
                ax.scatter(g['lightsheet_in_mask_integrated'], g['total_radiant_efficiency'], s=60, color=GROUP_COLORS[treatment],
                           edgecolor='black', linewidth=0.8, label=short(treatment), zorder=3)
            for _, r in d.iterrows():
                ax.annotate(r['blinded_number'], (r['lightsheet_in_mask_integrated'], r['total_radiant_efficiency']),
                            fontsize=6, color='#555555', xytext=(3, 3), textcoords='offset points')
            stat = correlations[(correlations['band_mm'] == PRIMARY_BAND) & (correlations['tracer'] == tracer) & (correlations['lightsheet_measure'] == 'in_mask_integrated')
                                & (correlations['surface'] == surface)].iloc[0]
            ax.set_xlabel(f'light-sheet: integrated tracer signal, {PRIMARY_BAND} mm {surface.split("_")[0]} band', fontsize=9)
            ax.set_ylabel('IVIS: total radiant efficiency', fontsize=9)
            ax.set_title(f'{tracer}, {surface.split("_")[0]}: Spearman ρ = {stat["spearman_rho"]:+.2f} '
                         f'(p = {stat["p_spearman"]:.3f}), n = {int(stat["n"])}', fontsize=10, loc='left')
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)

        # group means, both modalities, same panel pair
        for col, (measure, column, label) in enumerate([('IVIS', 'total_radiant_efficiency', 'total radiant efficiency'),
                                                        ('light-sheet', 'lightsheet_in_mask_integrated', f'{PRIMARY_BAND} mm band, tracer voxels')], start=2):
            ax = axes[row, col]
            d = merged[(merged['band_mm'] == PRIMARY_BAND) & (merged['tracer'] == tracer)]
            width = 0.35
            for k, surface in enumerate(('dorsal_brain', 'ventral_brain')):
                s = d[d['tissue'] == surface]
                means = [s.loc[s['treatment'] == t, column].mean() for t in TREATMENT_ORDER]
                sems = [s.loc[s['treatment'] == t, column].sem() for t in TREATMENT_ORDER]
                x = np.arange(len(TREATMENT_ORDER)) + (k - 0.5) * width
                ax.bar(x, means, width=width, yerr=sems, capsize=4,
                       color=['#cccccc', '#9ecae1'][k], edgecolor='#333333', linewidth=0.6,
                       label=surface.split('_')[0])
                for j, t in enumerate(TREATMENT_ORDER):
                    vals = s.loc[s['treatment'] == t, column]
                    ax.scatter(np.full(len(vals), x[j]), vals, s=18, color='#333333', zorder=3)
            ax.set_xticks(np.arange(len(TREATMENT_ORDER)))
            ax.set_xticklabels([short(t) for t in TREATMENT_ORDER], fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f'{tracer}: {measure}', fontsize=10, loc='left')
            ax.legend(frameon=False, fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('IVIS vs light-sheet, per animal and by group — both reduced the same way '
                 '(surface average, no normalisation)', x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(FIG_DIR, 'ivis_vs_lightsheet.png')
    plt.savefig(path, dpi=120)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
