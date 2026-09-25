r"""
Does treatment change how much tracer reaches the brain surface? Measured without an atlas.

Reads the depth profiles from 03_depth_profiles.py (signal binned by distance below each animal's own
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
import argparse
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

# Every value this script needs comes from config.<name>, and config.select() resolves those in
# main() before any work starts. Nothing may be snapshotted here at import time, and no function
# may take one as a default argument: both would freeze whichever cohort was selected first.


# pingouin's output column names, and what this version of pingouin might call them instead. Only
# the display and the p-value lookup depend on these, but both are load-bearing.
TUKEY_COLUMNS = {'mean(A)': ('mean(A)', 'mean_A'), 'mean(B)': ('mean(B)', 'mean_B'),
                 'diff': ('diff',), 'T': ('T',), 'p-tukey': ('p-tukey', 'p_tukey'),
                 'hedges': ('hedges',)}


def short(treatment):
    return config.short(treatment)


def load():
    """Per animal x tracer x side: band sums, from the depth histograms."""
    animals = sorted([f[:-4] for f in os.listdir(config.DEPTH_DIR) if f.endswith('.npz')],
                     key=config.sort_key)
    rows, profiles = [], {}
    for animal in animals:
        data = np.load(os.path.join(config.DEPTH_DIR, f'{animal}.npz'), allow_pickle=True)
        edges = data['edges']
        centres = (edges[:-1] + edges[1:]) / 2
        treatment = str(data['treatment'])
        for tracer, side in itertools.product(config.CHANNELS, config.SIDES):
            signal = data[f'{tracer}_{side}_sum_in_mask']
            voxels = data[f'{tracer}_{side}_n_voxels']
            profiles[(animal, tracer, side)] = (centres, signal)
            row = {'animal': animal, 'treatment': treatment, 'tracer': tracer, 'side': side,
                   'background': float(data[f'{tracer}_background']),
                   'threshold': float(data[f'{tracer}_threshold'])}
            for name, (lo, hi) in config.BANDS.items():
                sel = (centres >= lo) & (centres < hi)
                row[name] = float(signal[sel].sum())
                row[f'{name}_voxels'] = float(voxels[sel].sum())
            row['fraction_deep'] = row['deep_500um_plus'] / row['inside_total'] if row['inside_total'] else np.nan
            row['surface_per_voxel'] = (row['surface_0_500um'] / row['surface_0_500um_voxels']
                                        if row['surface_0_500um_voxels'] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows), profiles, centres


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

    anova_rows, posthoc, skipped = [], [], []
    for tracer, side, measure in itertools.product(config.CHANNELS, config.SIDES, measures):
        d = (df[(df['tracer'] == tracer) & (df['side'] == side)][['animal', 'treatment', measure]]
             .rename(columns={measure: 'value'}))
        values = d['value'].to_numpy(dtype=float)
        # A measure that is undefined, or identical for every animal, cannot go into an ANOVA and
        # statsmodels fails on it with an error about asymptotic normality that says nothing about
        # the data. Skipping it - loudly - keeps one undefined measure from killing the run.
        if not np.isfinite(values).all():
            skipped.append(f'{tracer}/{side}/{measure}: undefined for {int(np.isnan(values).sum())} '
                           f'of {len(values)} rows')
            continue
        if len(values) < 2 or np.std(values, ddof=1) == 0:
            skipped.append(f'{tracer}/{side}/{measure}: identical for every animal, nothing to test')
            continue
        table = sm.stats.anova_lm(ols('value ~ C(treatment)', data=d).fit(), typ=2)
        ss_between, ss_total = float(table['sum_sq'].iloc[0]), float(table['sum_sq'].sum())
        anova_rows.append({'tracer': tracer, 'region': side, 'measure': measure,
                           'F': float(table['F'].iloc[0]), 'p_anova': float(table['PR(>F)'].iloc[0]),
                           'df_between': float(table['df'].iloc[0]), 'df_resid': float(table['df'].iloc[1]),
                           'eta_sq': ss_between / ss_total if ss_total else np.nan,
                           **{config.short(t): d.loc[d['treatment'] == t, 'value'].mean()
                              for t in config.GROUP_ORDER}})
        pairs = pg.pairwise_tukey(data=d, dv='value', between='treatment')
        pairs.insert(0, 'measure', measure)
        pairs.insert(0, 'region', side)
        pairs.insert(0, 'tracer', tracer)
        posthoc.append(pairs)
    for note in skipped:
        print(f'note: skipped {note}')
    if not posthoc:
        return pd.DataFrame(anova_rows), pd.DataFrame(columns=['tracer', 'region', 'measure', 'A', 'B'])
    return pd.DataFrame(anova_rows), pd.concat(posthoc, ignore_index=True)


def tukey_column(pairs, name):
    """
    The column of pingouin's Tukey output holding one quantity, whichever way this version spells it.

    pingouin 0.6 renamed mean(A) -> mean_A and p-tukey -> p_tukey, so a hardcoded list breaks on a
    fresh environment with a KeyError that looks like a data problem. Failing with the available
    columns instead says what to do about it.
    """
    for candidate in TUKEY_COLUMNS[name]:
        if candidate in pairs.columns:
            return candidate
    raise SystemExit(f"pingouin's Tukey output has no {name!r} column (looked for "
                     f"{TUKEY_COLUMNS[name]}); it has {list(pairs.columns)}. Add the new name to "
                     f'TUKEY_COLUMNS in 04_analysis.py.')


def tukey_p(posthoc, tracer, side, measure, treatment, reference=None):
    """The Tukey p for one pair, whichever way round pingouin listed it."""
    if reference is None:
        reference = config.REFERENCE_GROUP
    d = posthoc[(posthoc['tracer'] == tracer) & (posthoc['region'] == side) & (posthoc['measure'] == measure)]
    hit = d[((d['A'] == treatment) & (d['B'] == reference)) | ((d['A'] == reference) & (d['B'] == treatment))]
    return float(hit[tukey_column(posthoc, 'p-tukey')].iloc[0]) if len(hit) else np.nan


def ivis_comparison(df):
    """
    Per-animal agreement with the second modality, using the surface band.

    Skipped (returns None) when the cohort has no second modality configured, which is the normal case
    for a dataset without one. Everything else in this script still runs.

    Column names and the tissue -> side values come from config, because a second modality that is
    only *equivalent* to IVIS will not use IVIS's headings.
    """
    if not config.SECOND_MODALITY_CSV or not os.path.exists(config.SECOND_MODALITY_CSV):
        print('no second-modality table configured; skipping the cross-validation')
        return None, None
    columns = config.SECOND_MODALITY_COLUMNS
    tissue_to_side = config.SECOND_MODALITY_TISSUE_TO_SIDE
    ivis = pd.read_csv(config.SECOND_MODALITY_CSV, encoding='utf-8-sig')
    missing = [c for c in columns.values() if c not in ivis.columns]
    if missing:
        raise SystemExit(f'{config.SECOND_MODALITY_CSV} has no column {missing}; it has '
                         f'{list(ivis.columns)}. Set SECOND_MODALITY_COLUMNS in config.py to match.')
    ivis = ivis[ivis[columns['tissue']].isin(tissue_to_side)].copy()
    ivis['side'] = ivis[columns['tissue']].map(tissue_to_side)
    ivis['native_tracer'] = ivis[columns['tracer']].astype(str)
    merged = df.merge(ivis[[columns['animal'], 'native_tracer', 'side', columns['total'], columns['average']]],
                      left_on=['animal', 'tracer', 'side'],
                      right_on=[columns['animal'], 'native_tracer', 'side'], how='inner')
    if merged.empty:
        raise SystemExit(f'no animal in the depth profiles matches the second-modality table '
                         f'{config.SECOND_MODALITY_CSV}: check the animal ids in column '
                         f'{columns["animal"]!r} and the tracer names in {columns["tracer"]!r} '
                         f'(configured channels: {list(config.CHANNELS)})')
    rows = []
    for tracer, side in itertools.product(config.CHANNELS, config.SIDES):
        d = merged[(merged['tracer'] == tracer) & (merged['side'] == side)]
        for native_measure, column_key in config.SECOND_MODALITY_PAIRS:
            rho, p = stats.spearmanr(d[native_measure], d[columns[column_key]])
            rows.append({'tracer': tracer, 'side': side, 'native_measure': native_measure,
                         'ivis_measure': column_key,
                         'n': len(d), 'spearman_rho': rho, 'p': p})
    return merged, pd.DataFrame(rows)


def figures(df, profiles, centres, merged, correlations, anova_table, posthoc):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    groups = {t: sorted(df.loc[df['treatment'] == t, 'animal'].unique(), key=config.sort_key)
              for t in config.GROUP_ORDER}

    # 1. depth profiles by group
    # Only depth >= 0 is drawn: this is the penetration profile. The negative bins are NOT empty -
    # they hold the outside-surface compartment that 03 measures (up to ~17% of the thresholded
    # signal) and that no entry in config.BANDS covers, so it appears in 03's QC printout and in the
    # per-animal table but in none of the statistics. See config.OUTSIDE_BAND.
    inside = centres >= 0
    fig, axes = plt.subplots(len(config.CHANNELS), len(config.SIDES), figsize=(13, 9), sharex=True)
    for (r, tracer), (c, side) in itertools.product(enumerate(config.CHANNELS), enumerate(config.SIDES)):
        ax = axes[r, c]
        top = 0.0
        for treatment in config.GROUP_ORDER:
            stack = np.array([profiles[(a, tracer, side)][1][inside] for a in groups[treatment]])
            mean, sem = stack.mean(axis=0), stack.std(axis=0, ddof=1) / np.sqrt(len(stack))
            ax.plot(centres[inside], mean, color=config.GROUP_COLORS[treatment], linewidth=2,
                    label=config.short(treatment))
            ax.fill_between(centres[inside], np.maximum(mean - sem, 1), mean + sem,
                            color=config.GROUP_COLORS[treatment], alpha=0.2, linewidth=0)
            top = max(top, float((mean + sem).max()))
        ax.set_yscale('log')
        ax.set_ylim(top / 3e3, top * 1.6)
        ax.set_xlim(0, 2.0)
        ax.set_title(f'{tracer}, {side}', fontsize=11, loc='left')
        if r == len(config.CHANNELS) - 1:
            ax.set_xlabel('depth below the brain surface (mm)', fontsize=10)
        if c == 0:
            ax.set_ylabel('integrated tracer signal', fontsize=10)
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Tracer signal vs depth below each animal\'s own brain surface (group mean ± SEM)',
                 x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(config.FIG_DIR, 'native_depth_profiles.png'), dpi=120)
    plt.close(fig)

    # 2. headline: the two compartments, both as absolute integrated signal, with ANOVA and Tukey
    compartments = ((config.SURFACE_BAND,
                     f'Integrated signal from the brain surface to {config.SURFACE_MM:g} mm depth'),
                    (config.DEEP_BAND,
                     f'Integrated signal deeper than {config.SURFACE_MM:g} mm'))
    panels = list(itertools.product(config.CHANNELS, config.SIDES))
    fig, axes = plt.subplots(len(compartments), len(panels), figsize=(4.5 * len(panels), 8.5), squeeze=False)
    rng = np.random.default_rng(0)
    for row, (measure, label) in enumerate(compartments):
        for col, (tracer, side) in enumerate(panels):
            ax = axes[row, col]
            d = df[(df['tracer'] == tracer) & (df['side'] == side)].set_index('animal')[measure]
            top = 0.0
            for x, treatment in enumerate(config.GROUP_ORDER):
                v = d[groups[treatment]]
                ax.bar(x, v.mean(), width=0.6, color=config.GROUP_COLORS[treatment], alpha=0.3,
                       yerr=v.sem(), capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#444444'})
                ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=45,
                           color=config.GROUP_COLORS[treatment],
                           edgecolor='black', linewidth=0.7, zorder=3)
                top = max(top, float(v.max()))
            # Tukey brackets for each dose against the reference group, and the ANOVA p in the title
            step = top * 0.09
            for k, (treatment, reference) in enumerate(config.CONTRASTS):
                p = tukey_p(posthoc, tracer, side, measure, treatment, reference)
                if not np.isfinite(p):
                    continue
                y = top * 1.06 + k * step
                x0, x1 = config.GROUP_ORDER.index(reference), config.GROUP_ORDER.index(treatment)
                ax.plot([x0, x0, x1, x1], [y, y + step * 0.18, y + step * 0.18, y], color='#444444', linewidth=1)
                mark = '*' if p < 0.05 else f'p = {p:.3f}'
                ax.text((x0 + x1) / 2, y + step * 0.22, mark, ha='center', va='bottom',
                        fontsize=12 if mark == '*' else 8.5, color='#222222')
            ax.set_ylim(0, top * 1.06 + len(config.CONTRASTS) * step + step * 0.6)
            anova = anova_table[(anova_table['tracer'] == tracer) & (anova_table['region'] == side)
                                & (anova_table['measure'] == measure)]
            p_anova = float(anova['p_anova'].iloc[0]) if len(anova) else np.nan
            ax.set_xticks(range(len(config.GROUP_ORDER)))
            ax.set_xticklabels([config.short(t) for t in config.GROUP_ORDER], fontsize=9)
            ax.set_title(f'{tracer}, {side}\nANOVA p = {p_anova:.3f}', fontsize=10, loc='left')
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Integrated tracer signal by compartment: one-way ANOVA of treatment, Tukey post-hoc',
                 x=0.01, ha='left', fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(config.FIG_DIR, 'native_depth_headline.png'), dpi=120)
    plt.close(fig)

    if merged is None:            # no second modality configured
        print(f'wrote 2 figures to {config.FIG_DIR}')
        return

    # 3 and 4. agreement with the second modality, against its total and its average.
    # The two columns differ only by ROI area, so the matched light-sheet quantity differs too:
    # an integral for the total, a per-voxel average for the average.
    modality = config.SECOND_MODALITY_NAME
    columns = config.SECOND_MODALITY_COLUMNS
    for native_measure, column_key, filename in (
            (config.SURFACE_BAND, 'total', 'native_vs_ivis.png'),
            ('surface_per_voxel', 'average', 'native_vs_ivis_average.png')):
        ivis_measure = columns[column_key]
        ivis_label = f'{modality} {column_key}'
        fig, axes = plt.subplots(len(config.CHANNELS), len(config.SIDES), figsize=(13, 9))
        for (r, tracer), (c, side) in itertools.product(enumerate(config.CHANNELS), enumerate(config.SIDES)):
            ax = axes[r, c]
            d = merged[(merged['tracer'] == tracer) & (merged['side'] == side)]
            for treatment in config.GROUP_ORDER:
                g = d[d['treatment'] == treatment]
                ax.scatter(g[native_measure], g[ivis_measure], s=55, color=config.GROUP_COLORS[treatment],
                           edgecolor='black', linewidth=0.7, label=config.short(treatment), zorder=3)
            for _, row in d.iterrows():
                ax.annotate(row['animal'], (row[native_measure], row[ivis_measure]),
                            fontsize=6, color='#555555', xytext=(3, 3), textcoords='offset points')
            stat = correlations[(correlations['tracer'] == tracer) & (correlations['side'] == side)
                                & (correlations['native_measure'] == native_measure)].iloc[0]
            label = (f'native-space surface signal (0-{config.SURFACE_MM * 1000:.0f} um)'
                     if native_measure == config.SURFACE_BAND
                     else f'native-space surface signal per voxel (0-{config.SURFACE_MM * 1000:.0f} um)')
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel(ivis_label, fontsize=9)
            ax.set_title(f'{tracer}, {side}: Spearman ρ = {stat["spearman_rho"]:+.2f} '
                         f'(p = {stat["p"]:.3f}, n = {int(stat["n"])})', fontsize=10, loc='left')
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'Native-space surface signal vs {ivis_label}, per animal',
                     x=0.01, ha='left', fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(os.path.join(config.FIG_DIR, filename), dpi=120)
        plt.close(fig)
    print(f'wrote 4 figures to {config.FIG_DIR}')


def check_depth_provenance():
    """
    Refuse to analyse depth profiles produced under a different configuration.

    A mismatch is a hard stop: it means these files belong to another cohort, or were written with
    different threshold, envelope or binning parameters, and either way any figure built on them
    would be wrong in a way nothing downstream could detect. A file with *no* recorded provenance
    predates this check, so it is a warning rather than a stop - the NS24122 numbers on disk today
    are exactly that, and re-running 03 is a 90-minute decision, not one to make silently.
    """
    mine = config.provenance()
    stale, unknown = [], []
    for name in sorted(os.listdir(config.DEPTH_DIR)):
        if not name.endswith('.npz'):
            continue
        with np.load(os.path.join(config.DEPTH_DIR, name), allow_pickle=True) as data:
            recorded = str(data['provenance']) if 'provenance' in data else ''
            cohort = str(data['cohort']) if 'cohort' in data else 'unrecorded'
        if not recorded:
            unknown.append(name[:-4])
        elif recorded != mine:
            stale.append(f'{name[:-4]}: cohort {cohort}, fingerprint {recorded}')
    if stale:
        raise SystemExit(f'{len(stale)} depth profile(s) in {config.DEPTH_DIR} were written under a '
                         f'different configuration (this is cohort {config.COHORT}, fingerprint '
                         f'{mine}):\n  - ' + '\n  - '.join(stale) +
                         f'\nPoint --cohort at the right cohort, or delete {config.DEPTH_DIR} and '
                         f're-run 03_depth_profiles.py.')
    if unknown:
        print(f'warning: {len(unknown)} depth profile(s) carry no provenance and may predate this '
              f'configuration: {unknown}. Re-run 03_depth_profiles.py to be certain.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cohort', help='cohort from config.COHORTS (default: $NATIVE_DEPTH_COHORT)')
    args = parser.parse_args()
    config.select(args.cohort)
    for line in config.validate():
        print(line)
    print(config.describe(), '\n', flush=True)
    check_depth_provenance()

    os.makedirs(config.OUT_DIR, exist_ok=True)
    df, profiles, centres = load()
    df.to_csv(os.path.join(config.OUT_DIR, 'native_depth_per_animal.csv'), index=False)
    print(f'{df["animal"].nunique()} animals')
    print('\nGroup means of the background and threshold (a technical check):')
    print(df.groupby(['treatment', 'tracer'])[['background', 'threshold']].mean()
          .reindex(config.GROUP_ORDER, level='treatment').round(1).to_string())

    measures = [config.SURFACE_BAND, config.DEEP_BAND, config.SHALLOW_BAND, config.INSIDE_BAND,
                'surface_per_voxel', 'fraction_deep']
    anova_table, posthoc = anova_and_posthoc(df, measures)
    anova_table.to_csv(os.path.join(config.OUT_DIR, 'native_depth_anova.csv'), index=False)
    posthoc.to_csv(os.path.join(config.OUT_DIR, 'native_depth_posthoc_tukey.csv'), index=False)
    pd.set_option('display.width', 240)
    print('\nOne-way ANOVA of treatment, per tracer x region (same procedure as the '
          f'{config.SECOND_MODALITY_NAME} analysis):')
    print(anova_table.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    compartments = posthoc[posthoc['measure'].isin([config.SURFACE_BAND, config.DEEP_BAND])]
    if len(compartments):
        show = (['tracer', 'region', 'measure', 'A', 'B']
                + [tukey_column(posthoc, name) for name in ('mean(A)', 'mean(B)', 'diff', 'T',
                                                            'p-tukey', 'hedges')])
        print('\nTukey post-hoc for the two compartment measures:')
        print(compartments[show].to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    else:
        print('\nno Tukey post-hoc table for the compartments - see the notes above')

    merged, correlations = ivis_comparison(df)
    if correlations is None:        # no second modality configured; figures() handles this too
        print('skipping the second-modality correlation outputs and figures')
    else:
        correlations.to_csv(os.path.join(config.OUT_DIR, 'native_vs_ivis_correlations.csv'), index=False)
        merged.to_csv(os.path.join(config.OUT_DIR, 'native_vs_ivis_per_animal.csv'), index=False)
        print(f'\nAgreement with {config.SECOND_MODALITY_NAME} '
              f'({merged["animal"].nunique()} animals with both):')
        print(correlations.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    figures(df, profiles, centres, merged, correlations, anova_table, posthoc)
    print(f'manifest: {config.write_manifest("04_analysis", {"animals_in_profiles": int(df["animal"].nunique())})}')


if __name__ == '__main__':
    main()
