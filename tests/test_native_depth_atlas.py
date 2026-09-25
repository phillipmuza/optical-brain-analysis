"""
Steps 05 and 06: the atlas-space arm, run on the synthetic cohort.

The resampler is ported code and its failure mode is a plausible-looking volume in the wrong place,
so the fixture writes brainreg's own files with an identity registration: every voxel's atlas
coordinate is its own coordinate. That makes the expected resampled volume computable by hand - each
binned atlas voxel must hold the mean of the sample voxels in its block - which is the only way to
tell a correct resampler from one that happens to produce a brain-shaped array.

The thresholds are checked the same way, against the formula recomputed here rather than against the
code's own output.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

import config
import synthetic_cohort

atlas_space = importlib.import_module('05_atlas_space')
atlas_maps = importlib.import_module('06_atlas_maps')


def run_step(module, monkeypatch, cohort='synthetic'):
    monkeypatch.setattr(sys, 'argv', [module.__name__, '--cohort', cohort])
    module.main()


def reference_and_tissue(animal):
    """The tissue definition 05 uses, recomputed from the fixture for comparison."""
    reference = tifffile.imread(config.image_path(animal, config.REFERENCE_IMAGE)).astype(np.float64)
    labels = tifffile.imread(os.path.join(config.DATA_DIR, animal, 'registration_dir',
                                          'registered_atlas.tiff'))
    inside = labels > 0
    values = reference[inside]
    median = float(np.median(values))
    sd = 1.4826 * float(np.median(np.abs(values - median)))
    return reference, inside, reference > (median - config.TISSUE_K * sd), median, sd


@pytest.fixture
def atlas_space_results(synthetic_cohort_fixture, monkeypatch):
    """Run step 05 for the synthetic cohort and hand back the animal and the paths it wrote."""
    run_step(atlas_space, monkeypatch)
    return {'animals': synthetic_cohort_fixture[2],
            'dir': config.ATLAS_SPACE_DIR,
            'thresholds': config.THRESHOLD_SUMMARY}


def test_resampling_places_each_sample_voxel_in_its_atlas_voxel(atlas_space_results):
    """
    The resampler's contract, verified rather than assumed: each tissue voxel is dropped into the
    binned atlas voxel its deformation field points at (rint(coord / voxel_mm / bin)), and each atlas
    voxel holds the mean of the raw intensity of the voxels that landed in it, 0 where none did.

    The assignment is reimplemented here from the fixture's own fields, and the intensity check uses
    an off-centre bright marker in the fixture - a transposed, sign-flipped or off-by-one-strided
    mapping would put the marker in the wrong place, which a symmetric sphere could not detect.

    Note the binning is a nearest-atlas-voxel assignment, not an exact block average: for a
    perfectly aligned field, rint rounds the half-way voxel coordinates to even, so the blocks are
    not a clean 2x2x2. That is a property of the method, not a bug, and it is why this test re-derives
    the assignment instead of assuming one.
    """
    animal = atlas_space_results['animals'][0]
    raw = tifffile.imread(config.image_path(animal, 'fitc.tif')).astype(np.float64)
    _, inside, tissue, _, _ = reference_and_tissue(animal)
    fields = [tifffile.imread(os.path.join(config.DATA_DIR, animal, 'registration_dir',
                                           f'deformation_field_{a}.tiff')).astype(np.float64)
              for a in range(3)]

    shape = atlas_space.binned_shape(raw.shape)
    flat = np.flatnonzero(tissue.ravel())
    stride = [int(np.prod(shape[i + 1:])) for i in range(3)]
    index = np.zeros(len(flat), dtype=np.int64)
    keep = np.ones(len(flat), dtype=bool)
    for axis in range(3):
        coord = np.rint(fields[axis].ravel()[flat] / config.ATLAS_VOXEL_MM / config.ATLAS_BIN).astype(np.int64)
        keep &= (coord >= 0) & (coord < shape[axis])
        index += np.clip(coord, 0, shape[axis] - 1) * stride[axis]
    index = index[keep]
    n_bins = int(np.prod(shape))
    counts = np.bincount(index, minlength=n_bins)
    values = raw.ravel()[flat][keep]
    expected = np.where(counts > 0, np.bincount(index, weights=values, minlength=n_bins)
                        / np.maximum(counts, 1), 0.0).reshape(shape)

    with np.load(os.path.join(atlas_space_results['dir'], f'{animal}.npz')) as data:
        volume, coverage = data['FITC'].astype(np.float64), data['n']
    assert volume.shape == shape
    assert counts.reshape(shape).astype(np.int64).sum() == keep.sum()
    assert np.allclose(volume, expected, atol=1.5), (
        f'the placed intensities do not match the mapping (max difference '
        f'{np.abs(volume - expected).max():.2f}) - the saved volume is uint16, so one intensity unit '
        f'of truncation is expected')
    assert np.array_equal(coverage.astype(np.int64), np.minimum(counts.reshape(shape), 255))


def test_the_marker_lands_where_the_registration_puts_it(atlas_space_results):
    """
    The fixture's bright marker sits at one known off-centre voxel, so the atlas voxel holding it is
    known: it is whatever the deformation field points at, worked out here by evaluating the field at
    that voxel rather than by reusing any of the mapping arithmetic. This is the check that catches a
    transposed or sign-flipped mapping - the tracer rim is symmetric and could not.
    """
    animal = atlas_space_results['animals'][0]
    bin_size = config.ATLAS_BIN
    fields = [tifffile.imread(os.path.join(config.DATA_DIR, animal, 'registration_dir',
                                           f'deformation_field_{a}.tiff')).astype(np.float64)
              for a in range(3)]
    expected = tuple(int(np.rint(fields[a][synthetic_cohort.MARKER_VOXEL]
                                 / config.ATLAS_VOXEL_MM / bin_size)) for a in range(3))
    with np.load(os.path.join(atlas_space_results['dir'], f'{animal}.npz')) as data:
        volume = data['FITC'].astype(np.float64)
    brightest = np.unravel_index(np.argmax(volume), volume.shape)
    assert brightest == expected, f'the marker landed at {brightest}, the field says {expected}'
    assert volume[expected] == pytest.approx(synthetic_cohort.MARKER_VALUE, abs=2)
    # the atlas arm keeps tissue outside the registered atlas, so the volume is not just its outline
    assert (volume > 0).mean() > 0.05


def test_voxels_the_atlas_does_not_cover_are_still_resampled(atlas_space_results):
    """
    The pial rim and the cisterns sit outside the registered atlas, and keeping them is the whole
    point of this stage - so coverage outside the atlas outline must be non-zero.
    """
    animal = atlas_space_results['animals'][0]
    _, inside, _, _, _ = reference_and_tissue(animal)
    atlas_small = inside[::config.ATLAS_BIN, ::config.ATLAS_BIN, ::config.ATLAS_BIN]
    with np.load(os.path.join(atlas_space_results['dir'], f'{animal}.npz')) as data:
        coverage = data['n']
    assert (coverage[~atlas_small] > 0).sum() > 0
    assert coverage[~atlas_small].sum() < coverage[atlas_small].sum()


def test_thresholds_are_recomputed_independently(atlas_space_results):
    """
    One threshold per animal x tracer: background + K_MAD robust SDs, over the atlas voxels that hold
    tissue. Recomputed here from the fixture rather than trusted from the code's own output.
    """
    table = pd.read_csv(atlas_space_results['thresholds'])
    animals = atlas_space_results['animals']
    assert len(table) == len(animals) * len(config.CHANNELS)
    assert {'animal_number', 'tracer', 'threshold'} <= set(table.columns)
    assert set(table['animal_number']) == set(animals)

    for animal in animals[:2]:
        raw = tifffile.imread(config.image_path(animal, 'fitc.tif')).astype(np.float64)
        _, inside, tissue, _, _ = reference_and_tissue(animal)
        # the population is the atlas voxels that hold tissue, which is what 05 thresholds on
        measured = inside & tissue
        values = raw[measured]
        median = float(np.median(values))
        sd = 1.4826 * float(np.median(np.abs(values - median)))
        expected = median + config.K_MAD * sd
        got = table[(table['animal_number'] == animal) & (table['tracer'] == 'FITC')]['threshold'].iloc[0]
        assert got == pytest.approx(expected, rel=1e-6)
        # and that population really is a constrained one, not the whole volume
        assert measured.sum() < raw.size
        assert median > np.median(raw.ravel())


def test_saved_volumes_carry_their_cohort_and_provenance(atlas_space_results):
    for animal in atlas_space_results['animals']:
        with np.load(os.path.join(atlas_space_results['dir'], f'{animal}.npz')) as data:
            assert str(data['cohort']) == 'synthetic'
            assert str(data['provenance']) == config.provenance()
            assert data['FITC'].dtype == np.uint16 and data['n'].dtype == np.uint8
            assert {'FITC', 'TxR', 'n', 'bg_FITC', 'bg_TxR'} <= set(data.files)


def test_tissue_fraction_is_recorded_and_not_assumed(atlas_space_results):
    """
    The registered atlas overhangs the sample, and how much it overhangs varies with registration
    fit - which is group-related in these cohorts. The fraction of atlas voxels that hold tissue is
    what tells a reader how much of the atlas was actually measured, so it is reported.
    """
    table = pd.read_csv(atlas_space_results['thresholds'])
    animal = atlas_space_results['animals'][0]
    _, inside, tissue, _, _ = reference_and_tissue(animal)
    fraction = float(tissue[inside].mean())
    row = table[table['animal_number'] == animal].iloc[0]
    assert row['tissue_fraction_of_atlas'] == pytest.approx(fraction, rel=1e-6)
    # the fixture's atlas is offset from the brain, so part of it hangs over dark medium and is not
    # tissue: the fraction must be below 1, which is what makes it worth reporting
    assert 0.5 < fraction < 1.0


def test_atlas_stage_refuses_when_05_has_not_run(synthetic_cohort_fixture, monkeypatch, tmp_path):
    """06 needs volumes from 05; without them it must say so and name the command to run."""
    with pytest.raises(SystemExit, match='no atlas-space volumes yet'):
        run_step(atlas_maps, monkeypatch)


def test_atlas_stage_runs_once_05_has(synthetic_cohort_fixture, monkeypatch):
    """The whole atlas arm: 05 then 06, ending in the two figures and their tables."""
    run_step(atlas_space, monkeypatch)
    run_step(atlas_maps, monkeypatch)
    for figure in ('compartment_maps_FITC.png', 'compartment_maps_TxR.png', 'coronal_profile.png'):
        path = os.path.join(config.FIG_DIR, figure)
        assert os.path.getsize(path) > 0, f'{figure} was not written'
    assert os.path.getsize(os.path.join(config.OUT_DIR, 'coronal_profile_totals.csv')) > 0


def test_step_05_needs_the_atlas_annotation(synthetic_cohort_fixture, monkeypatch, tmp_path):
    config.COHORTS['synthetic']['atlas_annotation'] = str(tmp_path / 'no_such_annotation.tiff')
    with pytest.raises(SystemExit, match='needs the atlas annotation'):
        run_step(atlas_space, monkeypatch)