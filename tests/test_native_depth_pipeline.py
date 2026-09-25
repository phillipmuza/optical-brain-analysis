"""
The pipeline, run end to end on a fabricated cohort.

This is the test the config work exists for: a dataset that has never been seen is a COHORTS entry,
and 02 -> 03 -> 04 produce masks, depth profiles, statistics and figures from it with no code change.
It also holds the guard that matters most in practice - two cohorts with the same animal ids must not
be able to analyse each other's files.

It is not a test of the science. The synthetic cohort is a sphere with a bright rim; the numbers it
produces are meaningless, and the assertions here are about which files exist, which columns they
carry, and which errors are raised.
"""
import importlib
import json
import os
import sys

import numpy as np
import pytest

import config

build_masks = importlib.import_module('02_build_masks')
depth_profiles = importlib.import_module('03_depth_profiles')
analysis = importlib.import_module('04_analysis')


def run_step(module, monkeypatch):
    """Call a step's main() as the command line would."""
    monkeypatch.setattr(sys, 'argv', [module.__name__])
    module.main()


@pytest.fixture
def pipeline_results(tmp_path, synthetic_cohort_fixture, monkeypatch):
    """Run 02 and 03 on the synthetic cohort and hand back the paths they wrote."""
    run_step(build_masks, monkeypatch)
    run_step(depth_profiles, monkeypatch)
    return {
        'results': config.RESULTS_DIR,
        'masks': config.MASK_DIR,
        'depth': config.DEPTH_DIR,
        'out': config.OUT_DIR,
        'figures': config.FIG_DIR,
        'animals': synthetic_cohort_fixture[2],
    }


def test_masks_are_written_per_animal_with_their_provenance(pipeline_results):
    animals = pipeline_results['animals']
    written = sorted(f for f in os.listdir(pipeline_results['masks']) if f.endswith('.npz'))
    assert written == [f'{a}.npz' for a in animals]
    with np.load(os.path.join(pipeline_results['masks'], f'{animals[0]}.npz')) as data:
        assert str(data['dataset']) == 'synthetic'
        assert str(data['provenance']) == config.provenance()
        assert tuple(int(s) for s in data['shape']) == (64, 64, 64)
        bitmask = np.unpackbits(data['packed']).astype(bool)
        assert 0.05 < bitmask.mean() < 1.0, 'the Otsu mask should be a brain-sized fraction of the volume'


def test_depth_profiles_carry_the_bands_the_analysis_needs(pipeline_results):
    animals = pipeline_results['animals']
    path = os.path.join(pipeline_results['depth'], f'{animals[0]}.npz')
    with np.load(path, allow_pickle=True) as data:
        assert str(data['dataset']) == 'synthetic'
        assert str(data['provenance']) == config.provenance()
        assert str(data['treatment']) == 'Vehicle'
        edges = data['edges']
        centres = (edges[:-1] + edges[1:]) / 2
        assert centres.min() < 0 < centres.max(), 'signed depth must span both sides of the surface'
        for tracer in config.CHANNELS:
            for side in config.SIDES:
                assert data[f'{tracer}_{side}_sum_in_mask'].shape == centres.shape
                assert data[f'{tracer}_{side}_n_voxels'].shape == centres.shape
        core = data['FITC_dorsal_sum_in_mask'] + data['FITC_ventral_sum_in_mask']
        assert core[centres >= 0].sum() > 0, 'the tracer rim should put signal above the threshold'


def test_analysis_produces_the_tables_and_figures(pipeline_results, monkeypatch, capsys):
    run_step(analysis, monkeypatch)
    capsys.readouterr()

    import pandas as pd
    animals = pipeline_results['animals']
    per_animal = pd.read_csv(os.path.join(pipeline_results['out'], 'native_depth_per_animal.csv'))
    # one row per animal x tracer x side
    assert len(per_animal) == len(animals) * len(config.CHANNELS) * len(config.SIDES)
    assert set(per_animal['treatment']) == set(config.GROUP_ORDER)
    for band in (config.SURFACE_BAND, config.DEEP_BAND, config.SHALLOW_BAND, config.INSIDE_BAND):
        assert band in per_animal.columns, f'{band} missing from the per-animal table'
    anova = pd.read_csv(os.path.join(pipeline_results['out'], 'native_depth_anova.csv'))
    assert set(anova['measure']) >= {config.SURFACE_BAND, config.DEEP_BAND}
    assert set(anova['region']) == set(config.SIDES)
    for figure in ('native_depth_profiles.png', 'native_depth_headline.png'):
        assert os.path.getsize(os.path.join(pipeline_results['figures'], figure)) > 0
    assert not os.path.exists(os.path.join(pipeline_results['figures'], 'native_vs_ivis.png')), \
        'a cohort with no second modality must not get second-modality figures'


def test_run_manifest_records_every_stage(pipeline_results, monkeypatch):
    run_step(analysis, monkeypatch)
    with open(os.path.join(pipeline_results['results'], 'run_manifest.json')) as handle:
        manifest = json.load(handle)
    assert {'02_build_masks', '03_depth_profiles', '04_analysis'} <= set(manifest)
    for stage in manifest.values():
        assert stage['dataset'] == 'synthetic'
        assert stage['provenance'] == config.provenance()
        assert 'Vehicle' in stage['animals']


def test_analysis_refuses_depth_profiles_from_another_configuration(pipeline_results, monkeypatch):
    """
    The failure this guards against: same animal ids, same filenames, different cohort. Without the
    fingerprint check, 04 would happily analyse them and produce a plausible figure.
    """
    animal = pipeline_results['animals'][0]
    path = os.path.join(pipeline_results['depth'], f'{animal}.npz')
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload['provenance'] = 'deadbeefdeadbeef'
    payload['cohort'] = 'anaesthetic'
    np.savez_compressed(path, **payload)

    with pytest.raises(SystemExit, match='different configuration'):
        run_step(analysis, monkeypatch)


def test_analysis_only_warns_about_files_that_predate_the_check(pipeline_results, monkeypatch, capsys):
    """The existing NS24122 depth profiles are exactly this case: usable, but flagged."""
    animal = pipeline_results['animals'][0]
    path = os.path.join(pipeline_results['depth'], f'{animal}.npz')
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files if key != 'provenance'}
    np.savez_compressed(path, **payload)

    run_step(analysis, monkeypatch)
    printed = capsys.readouterr().out
    assert 'no provenance' in printed
    assert animal in printed


def test_an_excluded_animal_does_not_reach_the_statistics(pipeline_results, monkeypatch, capsys):
    """
    Masks and depth profiles outlive an animal's place in the cohort: they were written before it was
    excluded, or by a run with a different exclusion list. The published exclusions were applied
    after the fact in the intensity arm, so this is not hypothetical.
    """
    import pandas as pd
    animal = pipeline_results['animals'][0]
    config.DATASET['exclude_animals'] = [animal]
    config.resolve()

    run_step(analysis, monkeypatch)
    printed = capsys.readouterr().out
    assert f'ignoring [\'{animal}\']' in printed
    per_animal = pd.read_csv(os.path.join(pipeline_results['out'], 'native_depth_per_animal.csv'))
    assert animal not in set(per_animal['animal'])
    assert len(per_animal) == (len(pipeline_results['animals']) - 1) * len(config.CHANNELS) * len(config.SIDES)


def test_a_changed_exclusion_list_does_not_invalidate_older_profiles(pipeline_results, monkeypatch, capsys):
    """
    The other half of the exclusion rule, and the reason it is a read-time filter rather than part of
    the fingerprint: excluding an animal says nothing about how the others were measured, so it must
    not force a 90-minute re-run. The profiles stay valid; the animal is dropped when they are read.
    """
    import numpy as np
    import pandas as pd
    animal = pipeline_results['animals'][0]
    with np.load(os.path.join(pipeline_results['depth'], f'{animal}.npz'), allow_pickle=True) as data:
        fingerprint = str(data['provenance'])
    assert fingerprint == config.provenance()

    config.DATASET['exclude_animals'] = [animal]
    config.resolve()
    assert config.provenance() == fingerprint, 'cohort membership is not a measurement parameter'

    run_step(analysis, monkeypatch)          # runs rather than refusing
    capsys.readouterr()
    per_animal = pd.read_csv(os.path.join(pipeline_results['out'], 'native_depth_per_animal.csv'))
    assert animal not in set(per_animal['animal'])


def test_a_cohort_entry_with_a_wrong_group_name_stops_before_any_work(synthetic_cohort_fixture, monkeypatch):
    """
    The other half of generalising: a typo in a group name must not reach the figures. It is the one
    mistake the code cannot detect later, because an empty group is a valid-looking DataFrame.
    """
    config.DATASET['groups'][2]['name'] = 'Drug_30'
    config.resolve()          # re-read the patched dict, as any script does when it imports config
    with pytest.raises(SystemExit) as raised:
        run_step(build_masks, monkeypatch)
    assert 'vanish from every figure' in str(raised.value)
    assert not os.listdir(config.MASK_DIR), 'nothing should have been written'


def test_the_other_stages_need_no_atlas(synthetic_cohort_fixture, monkeypatch, tmp_path):
    """01-04 run for a cohort with no atlas inputs configured at all - which is the normal case."""
    for module in (build_masks, depth_profiles, analysis):
        run_step(module, monkeypatch)
    assert os.path.isdir(config.DEPTH_DIR)