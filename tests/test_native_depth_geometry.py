"""
The native-depth pipeline's own tests: the geometry it measures with, and the config that decides
which dataset it measures.

The geometry tests use shapes whose depth values can be worked out by hand from a solid cube, so a
change to the envelope or the signed distance shows up as a wrong number rather than a plausible
figure. They were written against unchanged code first: any mismatch is a bug in the code, not a
test to adjust.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

PACKAGE = Path(__file__).resolve().parent.parent / 'native_depth_analysis'
sys.path.insert(0, str(PACKAGE))

import config                     # noqa: E402
import importlib                  # noqa: E402
depth_profiles = importlib.import_module('03_depth_profiles')


def solid_cube(shape=(20, 20, 20), lo=5, hi=15):
    """A solid cube of brain in the middle of an empty volume."""
    mask = np.zeros(shape, dtype=bool)
    mask[lo:hi, lo:hi, lo:hi] = True
    return mask


class TestEnvelope:
    def test_closing_radius_zero_leaves_a_convex_solid_alone(self):
        mask = solid_cube()
        envelope = depth_profiles.outer_envelope(mask, closing_radius=0)
        assert envelope.sum() == mask.sum()
        assert np.array_equal(envelope, mask)

    def test_a_closed_radius_does_not_grow_a_convex_solid(self):
        """Closing adds material in concavities; a cube has none, so nothing changes."""
        mask = solid_cube()
        envelope = depth_profiles.outer_envelope(mask, closing_radius=2)
        assert envelope.sum() == mask.sum()

    def test_internal_cavity_is_sealed(self):
        """The documented job of the envelope: ventricles must not count as surface."""
        mask = solid_cube()
        mask[8:12, 8:12, 8:12] = False        # a 4^3 ventricle, open to nothing
        assert mask.sum() == 1000 - 64
        envelope = depth_profiles.outer_envelope(mask, closing_radius=0)
        assert envelope.sum() == 1000          # the cavity is filled, the outline is unchanged
        assert not mask[10, 10, 10] and envelope[10, 10, 10]

    def test_cavity_open_to_the_outside_is_sealed_by_the_closing(self):
        """
        A tube through the brain would otherwise make every deep voxel 'close to the surface'. Two
        voxels of clearance is the smallest radius ENVELOPE_CLOSING_VOX can bridge, so this tests the
        mechanism rather than the 100 um default.
        """
        mask = solid_cube()
        mask[5:10, 9:11, 9:11] = False        # a 4 x 2 x 2 tunnel from the dorsal surface to the middle
        envelope = depth_profiles.outer_envelope(mask, closing_radius=2)
        assert envelope[7, 9, 9], 'the tunnel should be sealed by a 2-voxel closing'
        assert envelope[9, 10, 10]


class TestSignedDepth:
    def test_inside_is_positive_and_outside_negative(self):
        mask = solid_cube()
        depth = depth_profiles.signed_depth(mask)
        assert depth[10, 10, 10] > 0
        assert depth[0, 0, 0] < 0
        assert depth[mask].min() > 0
        assert depth[~mask].max() < 0

    def test_hand_computed_depth_at_the_centre_of_a_cube(self):
        """
        Mask indices 5..14 in a 20^3 volume at 20 um. From index 10 the nearest outside voxel is
        index 15, five voxels away, so 5 * 0.02 = 0.1 mm - and the cube is 10 voxels across, so no
        voxel can be deeper than that.
        """
        mask = solid_cube()
        depth = depth_profiles.signed_depth(mask)
        assert depth[10, 10, 10] == pytest.approx(5 * config.VOXEL_MM, abs=1e-6)
        assert depth.max() == pytest.approx(5 * config.VOXEL_MM, abs=1e-6)

    def test_depth_scales_with_the_voxel_size(self, monkeypatch):
        """A 25 um cohort must not be measured with 20 um distances."""
        mask = solid_cube()
        twenty = depth_profiles.signed_depth(mask).max()
        monkeypatch.setattr(config, 'VOXEL_MM', 0.025)
        assert depth_profiles.signed_depth(mask).max() == pytest.approx(twenty * 1.25, rel=1e-6)


class TestDorsalSplit:
    def test_an_even_cube_splits_in_half(self):
        mask = solid_cube()
        dorsal = depth_profiles.dorsal_mask(mask)
        assert dorsal[mask].mean() == pytest.approx(0.5, abs=0.01)
        # and it splits the brain, not the whole volume: columns outside the cube are not "dorsal"
        assert dorsal.sum() > mask.sum() / 2

    def test_the_split_follows_the_brain_not_a_global_plane(self):
        """
        Each column is split at its own mid-height, which is the point of doing it this way: a
        column of brain sitting entirely low in the volume is still half dorsal.
        """
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[:, 12:18, 5:15] = True           # brain only in the ventral part of the volume
        dorsal = depth_profiles.dorsal_mask(mask)
        assert dorsal[mask].mean() == pytest.approx(0.5, abs=0.01)

    def test_brain_free_columns_are_counted_dorsal(self):
        """
        Characterisation of a quirk, not an endorsement. A column with no envelope voxels has top=NaN,
        so mid becomes inf and `dv < mid` is True everywhere: every voxel in that column counts as
        dorsal. This is the opposite of what the `np.where(np.isnan(top), np.inf, ...)` line looks
        like it does, and it only touches voxels outside the brain - the thresholded measures are all
        inside the envelope, but the all-voxel (sum_above_bg) histograms are not. Recorded here so a
        change to it is deliberate rather than accidental.
        """
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[:, 5:15, 5:15] = True            # nothing at all in the first 5 and last 5 columns
        dorsal = depth_profiles.dorsal_mask(mask)
        assert dorsal[:, :, 0].all()
        assert dorsal[:, :, 19].all()
        assert dorsal[mask].mean() == pytest.approx(0.5, abs=0.01)


class TestHistogramBinning:
    def test_every_kept_voxel_lands_in_a_bin(self):
        """
        The bin index is computed from the float depth, and bincount silently returns a longer array
        if an index escapes the range - which would then either raise on the += or, worse, shift
        every bin. Whatever the depth, the index must be inside [0, n_bins).
        """
        edges = np.arange(config.DEPTH_LO, config.DEPTH_HI + config.DEPTH_STEP / 2,
                          config.DEPTH_STEP, dtype=np.float32)
        n_bins = len(edges) - 1
        mask = solid_cube()
        depth = depth_profiles.signed_depth(mask)
        keep = (depth >= config.DEPTH_LO) & (depth < config.DEPTH_HI)
        index = ((depth - config.DEPTH_LO) / config.DEPTH_STEP).astype(np.int32)[keep]
        assert index.min() >= 0
        assert index.max() < n_bins
        assert np.bincount(index, minlength=2 * n_bins).shape == (2 * n_bins,)


class TestConfigCohortSelection:
    """The whole point of the config work: a dataset is an entry, and a wrong entry stops the run."""

    def test_band_names_are_derived_from_surface_mm(self):
        """
        The boundary number must live in exactly one place. BANDS is built from SURFACE_MM at import,
        so the names, the bands and the number cannot drift apart the way a second hardcoded 0.5 in
        the analysis would.
        """
        assert config.SURFACE_BAND == f'surface_0_{int(round(config.SURFACE_MM * 1000))}um'
        assert config.DEEP_BAND == f'deep_{int(round(config.SURFACE_MM * 1000))}um_plus'
        assert config.BANDS[config.SURFACE_BAND] == (0.0, config.SURFACE_MM)
        assert config.BANDS[config.DEEP_BAND] == (config.SURFACE_MM, float('inf'))
        assert config.band_name('surface_0', 0.3) == 'surface_0_300um'

    def test_resolve_picks_up_a_changed_dataset(self, monkeypatch):
        """
        The mechanism every other test here relies on: config.resolve() re-reads DATASET, which is why
        no script may snapshot a config value at import time.
        """
        monkeypatch.setattr(config, 'DATASET', dict(config.DATASET, name='renamed',
                                                    reference_group='Vehicle'))
        assert config.resolve() == 'renamed'
        assert config.RESULTS_DIR.endswith(os.path.join('results', 'renamed'))

    def test_resolve_gives_the_dataset_names(self, tmp_path, synthetic_cohort_fixture):
        data_dir, data_map, animals = synthetic_cohort_fixture
        config.resolve()
        assert config.DATASET_NAME == 'synthetic'
        assert config.GROUP_ORDER == ['Vehicle', 'Medetomidine', 'K/X']
        assert config.REFERENCE_GROUP == 'Vehicle'
        assert config.CONTRASTS == [('Medetomidine', 'Vehicle'), ('K/X', 'Vehicle')]
        assert config.animals() == animals

    def test_results_are_namespaced_per_dataset(self, monkeypatch, synthetic_cohort_fixture):
        """
        Two datasets must never share a results directory: animal ids repeat between them, so a shared
        directory would half-overwrite and produce a figure mixing the two.
        """
        monkeypatch.delenv('NATIVE_DEPTH_RESULTS', raising=False)
        first = config.resolve()
        first_dir = config.RESULTS_DIR
        monkeypatch.setattr(config, 'DATASET', dict(config.DATASET, name='another-dataset'))
        second = config.resolve()
        second_dir = config.RESULTS_DIR
        assert first != second
        assert first_dir.endswith(os.path.join('results', first))
        assert second_dir.endswith(os.path.join('results', second))
        assert os.path.dirname(first_dir) == os.path.dirname(second_dir)

    def test_validate_rejects_a_group_name_that_is_not_in_the_data_map(self, tmp_path, synthetic_cohort_fixture):
        data_dir, data_map, animals = synthetic_cohort_fixture
        config.DATASET['groups'][1]['name'] = 'Drug_10mg_THAT_DOES_NOT_EXIST'
        with pytest.raises(SystemExit, match='vanish from every figure'):
            config.resolve()
            config.validate()

    def test_validate_rejects_an_empty_group(self, tmp_path, synthetic_cohort_fixture):
        data_dir, data_map, animals = synthetic_cohort_fixture
        config.DATASET['exclude_animals'] = ['an2', 'an1']     # all of Vehicle
        with pytest.raises(SystemExit, match='no animals'):
            config.resolve()
            config.validate()

    def test_validate_rejects_a_missing_image(self, tmp_path, synthetic_cohort_fixture):
        data_dir, data_map, animals = synthetic_cohort_fixture
        (Path(data_dir) / animals[0] / 'downsampled' / 'txr.tif').unlink()
        with pytest.raises(SystemExit, match='missing downsampled/txr.tif'):
            config.resolve()
            config.validate()

    def test_validate_reports_a_data_map_row_with_no_folder(self, synthetic_cohort_fixture):
        data_dir, data_map, animals = synthetic_cohort_fixture
        import pandas as pd
        table = pd.read_csv(data_map)
        table.loc[len(table)] = {'blinded_number': 'an999', 'treatment': 'Vehicle'}
        table.to_csv(data_map, index=False)
        config.resolve()
        lines = config.validate()
        assert any('an999' in line and 'warning' in line for line in lines)

    def test_provenance_changes_with_a_parameter_but_not_with_a_path(self, synthetic_cohort_fixture):
        config.resolve()
        before = config.provenance()
        config.DATASET['data_dir'] = 'elsewhere'
        config.resolve()
        assert config.provenance() == before, 'moving a data directory is not a measurement change'
        config.K_MAD = config.K_MAD + 1
        try:
            assert config.provenance() != before, 'the threshold rule is part of the measurement'
        finally:
            config.K_MAD = config.K_MAD - 1