"""Tests for signal_lookup + registration_regional against the synthetic atlas fixtures.

Expected numbers come from the fixture geometry (see conftest):
  16 voxels region A (left), 16 region B (right), 8 one step above region A (fillable),
  1 corner 258 um out (beyond a 200 um cap).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_lookup import ImageAnalyser, SUMMARY_FILENAME  # noqa: E402
from registration_regional import HemisphereAnalysis  # noqa: E402
from conftest import REGION_A, REGION_B  # noqa: E402


@pytest.fixture
def run_pipeline(channel_dir, registration_dir, structures_csv):
    """Run whole-brain lookup + hemisphere split the way main.py does; returns the analyser."""
    original_cwd = os.getcwd()
    os.chdir(channel_dir)  # ImageAnalyser reads relative filenames, as it does in the pipeline
    try:
        def _run(cap=200.0):
            analyser = ImageAnalyser(str(channel_dir), max_fill_distance_um=cap, voxel_size_um=20.0)
            analyser.run_analysis(
                "thresholded_image.tif",
                os.path.join(registration_dir, "registered_atlas.tiff"),
                structures_csv,
                os.path.join(registration_dir, "volumes.csv"),
            )
            HemisphereAnalysis(str(channel_dir), registration_dir=str(registration_dir)
                               ).split(analyser, structures_csv)
            return analyser
        yield _run
    finally:
        os.chdir(original_cwd)


def test_assignment_counts_200um_cap(run_pipeline):
    analyser = run_pipeline(cap=200.0)
    assert analyser.summary["n_signal_voxels"] == 41
    assert analyser.summary["n_in_atlas"] == 32
    assert analyser.summary["n_filled"] == 8
    assert analyser.summary["n_excluded_beyond_cap"] == 1
    counts = analyser.df.set_index("Region_ID")["Signal_Pixel_Count"]
    assert counts[REGION_A] == 24  # 16 direct + 8 filled
    assert counts[REGION_B] == 16


def test_uncapped_fills_the_far_voxel(run_pipeline):
    analyser = run_pipeline(cap=float("inf"))
    assert analyser.summary["n_filled"] == 9
    assert analyser.summary["n_excluded_beyond_cap"] == 0
    counts = analyser.df.set_index("Region_ID")["Signal_Pixel_Count"]
    assert counts[REGION_A] == 24
    assert counts[REGION_B] == 17  # far voxel's nearest label is in region B


def test_cap_smaller_than_step_excludes_everything_outside(run_pipeline):
    analyser = run_pipeline(cap=10.0)  # < one 20 um voxel step
    assert analyser.summary["n_in_atlas"] == 32
    assert analyser.summary["n_filled"] == 0
    assert analyser.summary["n_excluded_beyond_cap"] == 9


def test_summary_written_last_with_cap_recorded(run_pipeline, channel_dir):
    run_pipeline(cap=200.0)
    summary = pd.read_csv(channel_dir / SUMMARY_FILENAME)
    whole = summary[summary.compartment == "whole"].iloc[0]
    assert np.isclose(whole.max_fill_distance_um, 200.0)
    assert set(summary.compartment) == {"whole", "left", "right"}


def test_hemispheres_sum_to_whole_for_every_region(run_pipeline, channel_dir):
    run_pipeline(cap=200.0)
    whole = pd.read_csv(channel_dir / "region_counts.csv").set_index("Region_ID")["Signal_Pixel_Count"]
    left = (pd.read_csv(channel_dir / "left_hemisphere" / "region_counts.csv")
            .set_index("Region_ID")["Signal_Pixel_Count"])
    right = (pd.read_csv(channel_dir / "right_hemisphere" / "region_counts.csv")
             .set_index("Region_ID")["Signal_Pixel_Count"])
    combined = left.add(right, fill_value=0).sort_index()
    pd.testing.assert_series_equal(combined, whole.sort_index().reindex(combined.index)
                                   .astype(combined.dtype))
    # fixture-specific: region A entirely left, region B entirely right (absent, not zero,
    # from the other hemisphere's file)
    assert left[REGION_A] == 24 and REGION_A not in right.index
    assert right[REGION_B] == 16 and REGION_B not in left.index


@pytest.mark.xfail(strict=True, reason="decode_region_counts merges volumes.csv whose 'id' and "
                     "'structure_id_path' columns collide with structures.csv; pandas renames "
                     "them id_x/id_y so the whole-brain decoded CSV has no plain 'id' column")
def test_decoding_joins_shipped_structures_csv(run_pipeline, channel_dir, structures_csv):
    run_pipeline(cap=200.0)
    decoded = pd.read_csv(channel_dir / "decoded_region_counts.csv")
    counted = decoded[decoded.Signal_Pixel_Count.notna()]
    assert set(counted["id"]) == {REGION_A, REGION_B}
    structures = pd.read_csv(structures_csv)
    names = structures.set_index("id")["name"]
    assert set(counted["name"]) == {names[REGION_A], names[REGION_B]}
    # volumes came through from the registration's volumes.csv
    assert set(counted["volume_mm3"]) == {8.0, 4.0}


def test_shape_mismatch_raises(channel_dir, registration_dir):
    """Signal and atlas must agree on shape or the lookup fails loudly (regression guard)."""
    import tifffile
    tifffile.imwrite(channel_dir / "bad_shape.tif", np.zeros((8, 8, 8), dtype=np.uint8))
    analyser = ImageAnalyser(str(channel_dir), max_fill_distance_um=200.0)
    with pytest.raises(ValueError, match="same shape"):
        analyser.load_images("bad_shape.tif",
                             os.path.join(registration_dir, "registered_atlas.tiff"))
