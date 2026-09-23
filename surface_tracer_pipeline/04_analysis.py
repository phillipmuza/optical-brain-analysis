r"""
Does treatment change how much tracer reaches the brain surface? Measured without an atlas.

Reads the depth profiles from native_depth.py (signal binned by distance below each animal's own
brain surface, dorsal and ventral separately) and asks three things:

  1. absolute  - integrated tracer signal in the surface shells, compared between groups. This is
     the quantity IVIS measures and the one every atlas-based metric normalised away. It is
     meaningful here only because acquisition settings were identical across the cohort and Vehicle
     and 30 mg animals were randomised within imaging day.
  2. relative  - the shape of the depth profile (what fraction of an animal's signal is deep),
     which is the native-space version of the penetration question.
  3. agreement - the same per-animal comparison against IVIS that the atlas-space version failed
     ventrally; if the native measure is the better instrument it should agree at least as well.

Primary measure is the **thresholded** sum: voxels above that animal's fixed threshold, summed as
(raw - background). The all-voxel sum is kept for depths inside the brain only, because outside the
surface it is dominated by non-tissue voxels sitting far below the tissue background.

Run (after native_depth.py):  python native_depth_analysis.py
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

HERE = os.path.dirname(os.path.abspath(__file__))

DEPTH_DIR = config.DEPTH_DIR
FIG_DIR = config.FIG_DIR
OUT_DIR = config.OUT_DIR
IVIS_CSV = config.IVIS_CSV
TRACERS = list(config.CHANNELS)
SIDES = ['dorsal', 'ventral']
TREATMENT_ORDER = config.GROUP_ORDER
CONTRASTS = config.CONTRASTS
GROUP_COLORS = config.GROUP_COLORS
BANDS = {'surface_0_200um': (0.0, 0.2), 'surface_0_500um': (0.0, 0.5),
         'deep_500um_plus': (0.5, np.inf), 'inside_total': (0.0, np.inf)}


def short(treatment):
    return config.short(treatment)


def load():
    """Per animal x tracer x side: band sums, from the depth histograms."""
    animals = sorted([f[:-4] for f in os.listdir(DEPTH_DIR) if f.endswith('.npz')], key=config.sort_key)
    rows, profiles = [], {}
    for animal in animals:
        data = np.load(os.path.join(DEPTH_DIR, f'{animal}.npz'), allow_pickle=True)
        edges = data['edges']
        centres = (edges[:-1] + edges[1:]) / 2
        treatment = str(data['treatment'])
        for tracer, side in itertools.product(TRACERS, SIDES):
            signal = data[f'{tracer}_{side}_sum_in_mask']
            voxels = data[f'{tracer}_{side}_n_voxels']
            profiles[(animal, tracer, side)] = (centres, signal)
            row = {'animal': animal, 'treatment': treatment, 'tracer': tracer, 'side': side,
                   'background': float(data[f'{tracer}_background']),
                   'threshold': float(data[f'{tracer}_threshold'])}
            for name, (lo, hi) in BANDS.items():
                sel = (centres >= lo) & (centres < hi)
                row[name] = float(signal[sel].sum())
                row[f'{name}_voxels'] = float(voxels[sel].sum())
            row['fraction_deep'] = row['deep_500um_plus'] / row['inside_total'] if row['inside_total'] else np.nan
            row['surface_per_voxel'] = (row['surface_0_500um'] / row['surface_0_500um_voxels']
                                        if row['surface_0_500um_voxels'] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows), profiles, centres


def exact_p(values, group_a, group_b):
    """Two-sided exact permutation p over every re-assignment into groups of these sizes."""
    pooled = sorted(group_a + group_b, key=config.sort_key)
    observed = values[group_a].mean() - values[group_b].mean()
    null = [values[[x for x in pooled if x not in b]].mean() - values[list(b)].mean()
            for b in itertools.combinations(pooled, len(group_b))]
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))


def anova_and_posthoc(df, measures):
    """
    One-way ANOVA of treatment on each measure, per tracer x region, then Tukey post-hoc.

    Deliberately the same procedure as the IVIS analysis (`ivis_analysis_updated.ipynb`):
    statsmodels Type II ANOVA on the raw values, then pingouin's pairwise Tukey, which reports the
    group means, the difference, T, the Tukey-corrected p and Hedges' g. Keeping the two analyses on
    the same statistical footing is the point - it is what makes the modalities comparable.
    """
    import pingouin as pg
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    anova_rows, posthoc = [], []
    for tracer, side, measure in itertools.product(TRACERS, SIDES, measures):
        d = (df[(df['tracer'] == tracer) & (df['side'] == side)][['animal', 'treatment', measure]]
             .rename(columns={measure: 'value'}))
        table = sm.stats.anova_lm(ols('value ~ C(treatment)', data=d).fit(), typ=2)
        ss_between, ss_total = float(table['sum_sq'].iloc[0]), float(table['sum_sq'].sum())
        anova_rows.append({'tracer': tracer, 'region': side, 'measure': measure,
                           'F': float(table['F'].iloc[0]), 'p_anova': float(table['PR(>F)'].iloc[0]),
                           'df_between': float(table['df'].iloc[0]), 'df_resid': float(table['df'].iloc[1]),
                           'eta_sq': ss_between / ss_total if ss_total else np.nan,
                           **{short(t): d.loc[d['treatment'] == t, 'value'].mean() for t in TREATMENT_ORDER}})
        pairs = pg.pairwise_tukey(data=d, dv='value', between='treatment')
        pairs.insert(0, 'measure', measure)
        pairs.insert(0, 'region', side)
        pairs.insert(0, 'tracer', tracer)
        posthoc.append(pairs)
    return pd.DataFrame(anova_rows), pd.concat(posthoc, ignore_index=True)


def tukey_p(posthoc, tracer, side, measure, treatment, reference=config.REFERENCE_GROUP):
    """The Tukey p for one pair, whichever way round pingouin listed it."""
    d = posthoc[(posthoc['tracer'] == tracer) & (posthoc['region'] == side) & (posthoc['measure'] == measure)]
    hit = d[((d['A'] == treatment) & (d['B'] == reference)) | ((d['A'] == reference) & (d['B'] == treatment))]
    return float(hit['p-tukey'].iloc[0]) if len(hit) else np.nan


def ivis_comparison(df):
    """
    Per-animal agreement with the second modality, using the surface band.

    Skipped (returns None) when config.IVIS_CSV is None or missing, which is the normal case for a
    dataset without a second measurement. Everything else in this script still runs.
    """
    if not IVIS_CSV or not os.path.exists(IVIS_CSV):
        print('no second-modality table configured; skipping the cross-validation')
        return None, None
    ivis = pd.read_csv(IVIS_CSV, encoding='utf-8-sig')
    ivis = ivis[ivis['tissue'].isin(['dorsal_brain', 'ventral_brain'])].copy()
    ivis['side'] = ivis['tissue'].str.replace('_brain', '', regex=False)
    animal_column = config.IVIS_ANIMAL_COLUMN
    merged = df.merge(ivis[[animal_column, 'tracer', 'side', 'avg_radiant_efficiency', 'total_radiant_efficiency']],
                      left_on=['animal', 'tracer', 'side'], right_on=[animal_column, 'tracer', 'side'], how='inner')
    rows = []
    for tracer, side in itertools.product(TRACERS, SIDES):
        d = merged[(merged['tracer'] == tracer) & (merged['side'] == side)]
        for measure, ivis_measure in (('surface_0_500um', 'total_radiant_efficiency'),
                                      ('surface_per_voxel', 'avg_radiant_efficiency')):
            rho, p = stats.spearmanr(d[measure], d[ivis_measure])
            rows.append({'tracer': tracer, 'side': side, 'native_measure': measure,
                         'ivis_measure': ivis_measure.replace('_radiant_efficiency', ''),
                         'n': len(d), 'spearman_rho': rho, 'p': p})
    return merged, pd.DataFrame(rows)


def figures(df, profiles, centres, merged, correlations, anova_table, posthoc):
    os.makedirs(FIG_DIR, exist_ok=True)
    groups = {t: sorted(df.loc[df['treatment'] == t, 'animal'].unique(), key=config.sort_key)
              for t in TREATMENT_ORDER}

    # 1. depth profiles by group
    inside = centres >= 0          # negative-depth bins are empty by construction (see the QC note)
    fig, axes = plt.subplots(len(TRACERS), len(SIDES), figsize=(13, 9), sharex=True)
    for (r, tracer), (c, side) in itertools.product(enumerate(TRACERS), enumerate(SIDES)):
        ax = axes[r, c]
        top = 0.0
        for treatment in TREATMENT_ORDER:
            stack = np.array([profiles[(a, tracer, side)][1][inside] for a in groups[treatment]])
            mean, sem = stack.mean(axis=0), stack.std(axis=0, ddof=1) / np.sqrt(len(stack))
            ax.plot(centres[inside], mean, color=GROUP_COLORS[treatment], linewidth=2, label=short(treatment))
            ax.fill_between(centres[inside], np.maximum(mean - sem, 1), mean + sem,
                            color=GROUP_COLORS[treatment], alpha=0.2, linewidth=0)
            top = max(top, float((mean + sem).max()))
        ax.set_yscale('log')
        ax.set_ylim(top / 3e3, top * 1.6)
        ax.set_xlim(0, 2.0)
        ax.set_title(f'{tracer}, {side}', fontsize=11, loc='left')
        if r == len(TRACERS) - 1:
            ax.set_xlabel('depth below the brain surface (mm)', fontsize=10)
        if c == 0:
            ax.set_ylabel('integrated tracer signal', fontsize=10)
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Tracer signal vs depth below each animal\'s own brain surface (group mean ± SEM)',
                 x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(FIG_DIR, 'native_depth_profiles.png'), dpi=120)
    plt.close(fig)

    # 2. headline: the two compartments, both as absolute integrated signal, with ANOVA and Tukey
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    rng = np.random.default_rng(0)
    for row, measure, label in ((0, 'surface_0_500um', 'Integrated signal from brain surface to 0.5 mm depth'),
                                (1, 'deep_500um_plus', 'Integrated signal deeper than 0.5 mm')):
        for col, (tracer, side) in enumerate(itertools.product(TRACERS, SIDES)):
            ax = axes[row, col]
            d = df[(df['tracer'] == tracer) & (df['side'] == side)].set_index('animal')[measure]
            top = 0.0
            for x, treatment in enumerate(TREATMENT_ORDER):
                v = d[groups[treatment]]
                ax.bar(x, v.mean(), width=0.6, color=GROUP_COLORS[treatment], alpha=0.3,
                       yerr=v.sem(), capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#444444'})
                ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=45, color=GROUP_COLORS[treatment],
                           edgecolor='black', linewidth=0.7, zorder=3)
                top = max(top, float(v.max()))
            # Tukey brackets for each dose against Vehicle, and the ANOVA p in the title
            step = top * 0.09
            for k, (treatment, reference) in enumerate(CONTRASTS):
                p = tukey_p(posthoc, tracer, side, measure, treatment, reference)
                if not np.isfinite(p):
                    continue
                y = top * 1.06 + k * step
                x0, x1 = TREATMENT_ORDER.index(reference), TREATMENT_ORDER.index(treatment)
                ax.plot([x0, x0, x1, x1], [y, y + step * 0.18, y + step * 0.18, y], color='#444444', linewidth=1)
                mark = '*' if p < 0.05 else f'p = {p:.3f}'
                ax.text((x0 + x1) / 2, y + step * 0.22, mark, ha='center', va='bottom',
                        fontsize=12 if mark == '*' else 8.5, color='#222222')
            ax.set_ylim(0, top * 1.06 + len(CONTRASTS) * step + step * 0.6)
            anova = anova_table[(anova_table['tracer'] == tracer) & (anova_table['region'] == side)
                                & (anova_table['measure'] == measure)]
            p_anova = float(anova['p_anova'].iloc[0]) if len(anova) else np.nan
            ax.set_xticks(range(len(TREATMENT_ORDER)))
            ax.set_xticklabels([short(t) for t in TREATMENT_ORDER], fontsize=9)
            ax.set_title(f'{tracer}, {side}\nANOVA p = {p_anova:.3f}', fontsize=10, loc='left')
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Integrated tracer signal by compartment: one-way ANOVA of treatment, Tukey post-hoc',
                 x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(FIG_DIR, 'native_depth_headline.png'), dpi=120)
    plt.close(fig)

    if merged is None:            # no second modality configured
        print(f'wrote 2 figures to {FIG_DIR}')
        return

    # 3 and 4. agreement with IVIS, against its total and its average radiant efficiency.
    # The two IVIS measures differ only by ROI area, so the matched light-sheet quantity differs too:
    # an integral for the total, a per-voxel average for the average.
    for native_measure, ivis_measure, ivis_label, filename in (
            ('surface_0_500um', 'total_radiant_efficiency', 'IVIS total radiant efficiency', 'native_vs_ivis.png'),
            ('surface_per_voxel', 'avg_radiant_efficiency', 'IVIS average radiant efficiency',
             'native_vs_ivis_average.png')):
        fig, axes = plt.subplots(len(TRACERS), len(SIDES), figsize=(13, 9))
        for (r, tracer), (c, side) in itertools.product(enumerate(TRACERS), enumerate(SIDES)):
            ax = axes[r, c]
            d = merged[(merged['tracer'] == tracer) & (merged['side'] == side)]
            for treatment in TREATMENT_ORDER:
                g = d[d['treatment'] == treatment]
                ax.scatter(g[native_measure], g[ivis_measure], s=55, color=GROUP_COLORS[treatment],
                           edgecolor='black', linewidth=0.7, label=short(treatment), zorder=3)
            for _, row in d.iterrows():
                ax.annotate(row['animal'], (row[native_measure], row[ivis_measure]),
                            fontsize=6, color='#555555', xytext=(3, 3), textcoords='offset points')
            stat = correlations[(correlations['tracer'] == tracer) & (correlations['side'] == side)
                                & (correlations['native_measure'] == native_measure)].iloc[0]
            label = ('native-space surface signal (0-0.5 mm)' if native_measure == 'surface_0_500um'
                     else 'native-space surface signal per voxel (0-0.5 mm)')
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel(ivis_label, fontsize=9)
            ax.set_title(f'{tracer}, {side}: Spearman ρ = {stat["spearman_rho"]:+.2f} '
                         f'(p = {stat["p"]:.3f}, n = {int(stat["n"])})', fontsize=10, loc='left')
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'Native-space surface signal vs {ivis_label.lower()}, per animal',
                     x=0.01, ha='left', fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(os.path.join(FIG_DIR, filename), dpi=120)
        plt.close(fig)
    print(f'wrote 4 figures to {FIG_DIR}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, profiles, centres = load()
    df.to_csv(os.path.join(OUT_DIR, 'native_depth_per_animal.csv'), index=False)
    print(f'{df["animal"].nunique()} animals')
    print('\nGroup means of the background and threshold (a technical check):')
    print(df.groupby(['treatment', 'tracer'])[['background', 'threshold']].mean()
          .reindex(TREATMENT_ORDER, level='treatment').round(1).to_string())

    measures = ['surface_0_500um', 'deep_500um_plus', 'surface_0_200um', 'inside_total',
                'surface_per_voxel', 'fraction_deep']
    anova_table, posthoc = anova_and_posthoc(df, measures)
    anova_table.to_csv(os.path.join(OUT_DIR, 'native_depth_anova.csv'), index=False)
    posthoc.to_csv(os.path.join(OUT_DIR, 'native_depth_posthoc_tukey.csv'), index=False)
    pd.set_option('display.width', 240)
    print('\nOne-way ANOVA of treatment, per tracer x region (same procedure as the IVIS analysis):')
    print(anova_table.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\nTukey post-hoc for the two compartment measures:')
    print(posthoc[posthoc['measure'].isin(['surface_0_500um', 'deep_500um_plus'])]
          [['tracer', 'region', 'measure', 'A', 'B', 'mean(A)', 'mean(B)', 'diff', 'T', 'p-tukey', 'hedges']]
          .to_string(index=False, float_format=lambda v: f'{v:.4g}'))

    merged, correlations = ivis_comparison(df)
    correlations.to_csv(os.path.join(OUT_DIR, 'native_vs_ivis_correlations.csv'), index=False)
    merged.to_csv(os.path.join(OUT_DIR, 'native_vs_ivis_per_animal.csv'), index=False)
    print(f'\nAgreement with IVIS ({merged["animal"].nunique()} animals with both):')
    print(correlations.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    figures(df, profiles, centres, merged, correlations, anova_table, posthoc)


if __name__ == '__main__':
    main()
