"""
The factorial design: a group is a cell of a factor cross, and step 04 tests the factors.

Two things are being tested here and they are different in kind. The first is that the config layer
crosses the factors and names the cells the way DATASET says it does, and that every way of getting
that wrong - a level that is not declared, a group that is not a combination, a factor column that
does not exist - stops the run with a list rather than reaching a figure.

The second is the reason for all of it: a one-way test over four cells reports that the cells differ
and says nothing about either factor. When an effect is present in one phase and reverses in the
other, the pooled effect of the drug comes to nothing - correctly, because averaging the phases is
the wrong operation - and the four-cell test is significant regardless. The interaction term is what
answers the design's question, and test_the_pooled_drug_effect_is_null_when_the_effect_reverses_between_phases
pins that down.

Neither test runs the pipeline on real data, and the synthetic image is a sphere with a bright rim,
so nothing here is a statement about the science.
"""
import importlib
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest

import config

build_masks = importlib.import_module('02_build_masks')
depth_profiles = importlib.import_module('03_depth_profiles')
analysis = importlib.import_module('04_analysis')


def run_step(module, monkeypatch):
    """Call a step's main() as the command line would."""
    monkeypatch.setattr(sys, 'argv', [module.__name__])
    module.main()


def cell_frame(effect_on=0.0, effect_off=0.0, per_cell=10, seed=0, spread=3.0,
               measure=None, factor_columns=('drug', 'light')):
    """
    A per-animal table with a chosen cell structure.

    Vehicle cells sit at a baseline and the drug cells are offset by effect_on / effect_off, with
    within-cell noise - so equal effects, no effect, and an effect that reverses between the phases
    are all constructible. Ten animals per cell at a spread of 3 makes the null terms genuinely null
    rather than null-by-luck, which is what the assertions about them rest on.

    Hand-built rather than pushed through 02 and 03 because the numbers under test are the model's,
    and a fabricated rim brightness would make this a test of the fixture.
    """
    import synthetic_cohort

    measure = measure or config.SURFACE_BAND
    rng = np.random.default_rng(seed)
    rows = []
    for first, second, _ in synthetic_cohort.FACTORIAL_CELLS:
        offset = 0.0 if first == 'Vehicle' else (effect_on if second == 'LightsON' else effect_off)
        for k in range(per_cell):
            rows.append({'animal': f'{first[:3]}_{second[-2:]}_{k}',
                         'treatment': config.cell_name((first, second)),
                         factor_columns[0]: first, factor_columns[1]: second,
                         'tracer': 'FITC', 'side': 'dorsal',
                         measure: 100.0 + offset + rng.normal(0, spread)})
    return pd.DataFrame(rows)


def one_way_p(df, measure=None):
    """The p a one-way ANOVA of the four cells would report for the same frame."""
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    measure = measure or config.SURFACE_BAND
    d = df[['treatment', measure]].rename(columns={measure: 'value'})
    return float(sm.stats.anova_lm(ols('value ~ C(treatment)', data=d).fit(), typ=2)['PR(>F)'].iloc[0])


@pytest.fixture
def custom_cells(tmp_path, monkeypatch):
    """Point config at a factorial cohort built from whatever cells the test asks for."""
    import synthetic_cohort

    def _point(cells, **kwargs):
        data_dir, data_map, animals = synthetic_cohort.build_factorial_cohort(tmp_path, cells)
        monkeypatch.setenv('NATIVE_DEPTH_RESULTS', str(tmp_path / 'results'))
        monkeypatch.setattr(config, 'DATASET', synthetic_cohort.factorial_dataset_entry(
            data_dir, data_map, cells, **kwargs))
        config.resolve()
        return data_dir, data_map, animals

    yield _point
    config.resolve()


@pytest.fixture
def factorial_pipeline(tmp_path, factorial_cohort_fixture, monkeypatch):
    """02 and 03 run on the 2x2 cohort, with the paths they wrote."""
    run_step(build_masks, monkeypatch)
    run_step(depth_profiles, monkeypatch)
    return {'results': config.RESULTS_DIR, 'depth': config.DEPTH_DIR, 'out': config.OUT_DIR,
            'figures': config.FIG_DIR, 'animals': factorial_cohort_fixture[2]}


class TestTheDesign:
    def test_a_factorial_dataset_crosses_its_factor_columns(self, factorial_cohort_fixture):
        """The cells are the cross of the declared levels, in the declared order, not typed names."""
        assert config.FACTOR_COLUMNS == ['drug', 'light']
        assert config.FACTOR_LEVELS == {'drug': ['Vehicle', 'DRUG'],
                                        'light': ['LightsON', 'LightsOFF']}
        assert config.CELLS == [config.cell_name(('Vehicle', 'LightsON')),
                                config.cell_name(('Vehicle', 'LightsOFF')),
                                config.cell_name(('DRUG', 'LightsON')),
                                config.cell_name(('DRUG', 'LightsOFF'))]
        assert set(config.GROUP_ORDER) == set(config.CELLS)
        assert config.treatment_of('an1') == config.cell_name(('Vehicle', 'LightsON'))
        assert config.treatment_of('an7') == config.cell_name(('DRUG', 'LightsOFF'))
        assert config.factor_levels_of(config.treatment_of('an3')) == {'drug': 'DRUG',
                                                                      'light': 'LightsON'}

    def test_the_contrasts_are_the_four_simple_effects_from_the_dataset(self, factorial_cohort_fixture):
        """
        A factorial design names its own pairs. The default - every cell against the reference - is
        not the same set of questions, and would leave the interaction to be inferred.
        """
        assert len(config.CONTRASTS) == 4
        assert len(set(config.CONTRASTS)) == 4, 'the four simple effects must be distinct pairs'
        assert all(len(pair) == 2 for pair in config.CONTRASTS)
        assert all(group in config.GROUP_ORDER for pair in config.CONTRASTS for group in pair)

    def test_a_single_factor_dataset_keeps_the_default_contrasts(self, synthetic_cohort_fixture):
        assert config.FACTOR_COLUMNS == ['treatment']
        assert config.FACTOR_LEVELS == {'treatment': ['Vehicle', 'Medetomidine', 'K/X']}
        assert config.CELLS == config.GROUP_ORDER == ['Vehicle', 'Medetomidine', 'K/X']
        assert config.CONTRASTS == [('Medetomidine', 'Vehicle'), ('K/X', 'Vehicle')]

    def test_the_worked_2x2_example_in_config_is_internally_consistent(self):
        """
        The block a future cohort is copied from. It cannot be validate()d without its data, but its
        cells, contrasts and reference group can all be checked against each other here - which is
        where a typo in a name nobody has run yet would otherwise sit until the day it is used.
        """
        spec = config.DRUG_2X2_DATASET
        cells = [config.cell_name(levels) for levels in itertools.product(
            *(spec['factor_levels'][column] for column in spec['factor_columns']))]
        assert set(g['name'] for g in spec['groups']) == set(cells)
        assert len(spec['groups']) == 4
        assert spec['reference_group'] in cells
        assert len(spec['contrasts']) == 4
        assert len(set(spec['contrasts'])) == 4
        assert all(group in cells for pair in spec['contrasts'] for group in pair)
        # each factor level appears in exactly half the cells, which is what makes the four
        # contrasts the four simple effects rather than an arbitrary set of pairs
        for column in spec['factor_columns']:
            for level in spec['factor_levels'][column]:
                assert sum(1 for cell in cells if level in cell) == 2


class TestValidate:
    def test_rejects_a_level_that_is_not_declared(self, factorial_cohort_fixture):
        """
        The failure this catches is a typo or an animal from another cohort: undeclared, the value
        becomes a group of its own in one figure and vanishes from the next.
        """
        table = pd.read_csv(factorial_cohort_fixture[1])
        table.loc[0, 'light'] = 'Dusk'
        table.to_csv(factorial_cohort_fixture[1], index=False)
        config.resolve()
        with pytest.raises(SystemExit, match='not declared levels'):
            config.validate()

    def test_rejects_a_group_that_is_not_a_combination(self, factorial_cohort_fixture):
        config.DATASET['groups'][1]['name'] = config.cell_name(('DRUG', 'Dusk'))
        config.resolve()
        with pytest.raises(SystemExit, match='not a combination'):
            config.validate()

    def test_rejects_a_missing_factor_column(self, factorial_cohort_fixture):
        table = pd.read_csv(factorial_cohort_fixture[1]).rename(columns={'light': 'phase'})
        table.to_csv(factorial_cohort_fixture[1], index=False)
        config.resolve()
        with pytest.raises(SystemExit, match="no factor column 'light'"):
            config.validate()

    def test_rejects_a_factor_column_called_treatment(self, factorial_cohort_fixture):
        """
        'treatment' is the group column in the pipeline's own tables, so a factor of that name would
        be written over the cell it is supposed to explain.
        """
        config.DATASET['factor_columns'] = ['treatment', 'light']
        config.resolve()
        with pytest.raises(SystemExit, match='may not include'):
            config.validate()

    def test_rejects_a_level_containing_the_separator(self, factorial_cohort_fixture):
        """
        'A x B' would be ambiguous between a single level of that name and the cell (A, B), and the
        group name is built by joining levels, so a level may not contain the separator.
        """
        config.DATASET['factor_levels'] = {'drug': ['Vehicle', 'DRUG x 10mg'],
                                          'light': ['LightsON', 'LightsOFF']}
        config.resolve()
        with pytest.raises(SystemExit, match='containing the factor separator'):
            config.validate()

    def test_a_group_that_is_not_a_factor_combination_is_reported_not_split(self, factorial_cohort_fixture):
        with pytest.raises(SystemExit, match='factor level'):
            config.factor_levels_of('Vehicle')

    def test_rejects_a_contrast_that_names_no_group(self, factorial_cohort_fixture):
        """A typo here draws no bracket and raises nothing: a silent gap in a figure."""
        config.DATASET['contrasts'] = [('DRUG x LightsON', 'Vehicle x Dusk')]
        config.resolve()
        with pytest.raises(SystemExit, match='names'):
            config.validate()

    def test_warns_about_a_cell_too_small_for_an_interaction(self, custom_cells):
        cells = (('Vehicle', 'LightsON', 3), ('DRUG', 'LightsON', 1),
                 ('Vehicle', 'LightsOFF', 3), ('DRUG', 'LightsOFF', 3))
        custom_cells(cells)
        lines = config.validate()
        assert any('warning' in line and 'too few' in line for line in lines), lines

    def test_rejects_a_missing_combination_of_declared_levels(self, custom_cells):
        """
        No entry for a cell that the declared levels produce. It would otherwise be crossed, named,
        and then missing from every figure - the same failure as an empty group, one step earlier.
        """
        cells = (('Vehicle', 'LightsON', 2), ('DRUG', 'LightsON', 2), ('Vehicle', 'LightsOFF', 2))
        custom_cells(cells)
        with pytest.raises(SystemExit, match='no group entry|has no animals'):
            config.validate()

    def test_the_census_names_every_cell_and_both_factor_marginals(self, factorial_cohort_fixture):
        lines = config.validate()
        for cell in config.GROUP_ORDER:
            assert any(cell in line and 'n=' in line for line in lines), cell
        assert any(line.startswith('note: drug:') for line in lines), lines
        assert any(line.startswith('note: light:') for line in lines), lines


class TestTheStatistics:
    def test_the_two_way_model_reports_both_factors_and_the_interaction(self, factorial_cohort_fixture):
        """A drug effect in both phases: the drug term carries it, and the phases do not differ."""
        df = cell_frame(effect_on=60.0, effect_off=60.0)
        anova, posthoc = analysis.anova_and_posthoc(df, [config.SURFACE_BAND])
        assert set(anova['term']) == {'drug', 'light', 'drug:light'}
        assert analysis.anova_p(anova, 'FITC', 'dorsal', config.SURFACE_BAND, 'drug') < 0.001
        assert analysis.anova_p(anova, 'FITC', 'dorsal', config.SURFACE_BAND, 'drug:light') > 0.05
        # the four cells still get a post-hoc, so the two tables answer the two questions
        assert len(posthoc) == 6, 'four cells have six pairs'
        assert set(posthoc['A']) | set(posthoc['B']) == set(config.GROUP_ORDER)

    def test_the_pooled_drug_effect_is_null_when_the_effect_reverses_between_phases(
            self, factorial_cohort_fixture):
        """
        What the interaction term is for, and what a pooled drug comparison would say.

        The drug helps in LightsON and hurts by the same amount in LightsOFF. The interaction term
        sees it; the pooled drug effect - what "did the drug work?" asks - comes to zero, correctly,
        because averaging the two phases is the wrong thing to do to this result. Note that a one-way
        ANOVA of the same four cells is highly significant: "the groups differ" is true here and is
        not evidence about the drug either way, which is why the drug question has to be asked as a
        term in the model rather than as a difference between groups.
        """
        df = cell_frame(effect_on=60.0, effect_off=-60.0)
        anova, _ = analysis.anova_and_posthoc(df, [config.SURFACE_BAND])
        p_interaction = analysis.anova_p(anova, 'FITC', 'dorsal', config.SURFACE_BAND, 'drug:light')
        assert p_interaction < 1e-6
        assert analysis.anova_p(anova, 'FITC', 'dorsal', config.SURFACE_BAND, 'drug') > 0.05, \
            'the pooled drug effect must be null for this construction'
        assert one_way_p(df) < 0.001, \
            'the four cells differ, so a one-way test is significant without saying anything about the drug'

    def test_the_single_factor_model_is_still_the_one_way_anova(self, synthetic_cohort_fixture):
        """Parity: one factor column must give exactly the one-way fit this pipeline has always run."""
        rng = np.random.default_rng(0)
        rows = []
        for k, cell in enumerate(config.GROUP_ORDER):
            for j in range(4):
                rows.append({'animal': f'{cell[:4]}{j}', 'treatment': cell,
                             'tracer': 'FITC', 'side': 'dorsal',
                             config.SURFACE_BAND: 100.0 + 30.0 * k + rng.normal(0, 8)})
        df = pd.DataFrame(rows)
        anova, _ = analysis.anova_and_posthoc(df, [config.SURFACE_BAND])
        assert list(anova['term']) == ['treatment']
        assert len(anova) == 1, 'one row per measure, as before'
        assert analysis.anova_p(anova, 'FITC', 'dorsal', config.SURFACE_BAND) == pytest.approx(
            one_way_p(df), rel=1e-12)
        # and the eta squared is the same number as before the change: for one factor the total is
        # the between-group sum of squares plus the residual
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        d = df[['treatment', config.SURFACE_BAND]].rename(columns={config.SURFACE_BAND: 'value'})
        table = sm.stats.anova_lm(ols('value ~ C(treatment)', data=d).fit(), typ=2)
        assert float(anova['eta_sq'].iloc[0]) == pytest.approx(
            float(table['sum_sq'].iloc[0]) / float(table['sum_sq'].sum()), rel=1e-12)

    def test_the_means_are_named_by_the_short_labels(self, factorial_cohort_fixture):
        """The ANOVA table's group-mean columns are what the figures and the report read."""
        df = cell_frame()
        anova, _ = analysis.anova_and_posthoc(df, [config.SURFACE_BAND])
        assert set(config.short(cell) for cell in config.GROUP_ORDER) <= set(anova.columns)


class TestEndToEnd:
    def test_the_factorial_cohort_runs_through_02_03_and_04(self, factorial_pipeline, monkeypatch, capsys):
        run_step(analysis, monkeypatch)
        printed = capsys.readouterr().out

        per_animal = pd.read_csv(os.path.join(factorial_pipeline['out'], 'native_depth_per_animal.csv'))
        assert set(per_animal['treatment']) == set(config.GROUP_ORDER)
        assert len(per_animal) == len(factorial_pipeline['animals']) * len(config.CHANNELS) * len(config.SIDES)
        # one column per factor, so the table says which cell and why it is that cell
        assert set(per_animal['drug']) == {'Vehicle', 'DRUG'}
        assert set(per_animal['light']) == {'LightsON', 'LightsOFF'}

        anova = pd.read_csv(os.path.join(factorial_pipeline['out'], 'native_depth_anova.csv'))
        assert set(anova['term']) == {'drug', 'light', 'drug:light'}
        assert len(anova) == len(config.CHANNELS) * len(config.SIDES) * 6 * 3, \
            'two tracers x two regions x six measures x three terms'
        posthoc = pd.read_csv(os.path.join(factorial_pipeline['out'], 'native_depth_posthoc_tukey.csv'))
        assert set(posthoc['A']) | set(posthoc['B']) == set(config.GROUP_ORDER)

        for figure in ('native_depth_profiles.png', 'native_depth_headline.png',
                       'native_depth_interaction.png'):
            assert os.path.getsize(os.path.join(factorial_pipeline['figures'], figure)) > 0, figure
        assert 'drug x light two-way ANOVA' in printed, 'the run should say which model it fitted'

    def test_the_depth_profiles_record_the_cell(self, factorial_pipeline):
        path = os.path.join(factorial_pipeline['depth'], f'{factorial_pipeline["animals"][0]}.npz')
        with np.load(path, allow_pickle=True) as data:
            assert str(data['treatment']) == config.cell_name(('Vehicle', 'LightsON'))

    def test_the_analysis_reads_the_group_from_the_data_map_not_the_profile(
            self, factorial_pipeline, monkeypatch, capsys):
        """
        A group is which animals were compared, not how one was measured, so the profile's recorded
        group is not authoritative - which is what lets a cohort be relabelled, or its factors
        declared differently, without re-running the 90-minute measurement.
        """
        animal = factorial_pipeline['animals'][0]
        path = os.path.join(factorial_pipeline['depth'], f'{animal}.npz')
        with np.load(path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload['treatment'] = 'an_old_labelling'
        np.savez_compressed(path, **payload)

        run_step(analysis, monkeypatch)
        printed = capsys.readouterr().out
        assert 'record a group other than the data map' in printed
        per_animal = pd.read_csv(os.path.join(factorial_pipeline['out'], 'native_depth_per_animal.csv'))
        assert set(per_animal.loc[per_animal['animal'] == animal, 'treatment']) == \
            {config.cell_name(('Vehicle', 'LightsON'))}
