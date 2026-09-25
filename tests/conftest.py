"""Shared fixtures: a tiny synthetic 'brain' exercising the lookup + hemisphere path.

All 20 um isotropic. Atlas IDs and names come from the repo's real structures.csv
(id 9 = 'Primary somatosensory area, trunk, layer 6a', id 653 = 'Abducens nucleus')
so region decoding is genuinely exercised against the shipped lookup table.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Two real atlas IDs standing in for "region A" (cortex-like block) and "region B".
REGION_A = 9
REGION_B = 653

SHAPE = (32, 32, 24)  # z, y, x (axis order the pipeline uses)
VOXEL_UM = 20.0

# x >= this value is the right hemisphere (brainreg labels: 1 = left, 2 = right)
HEM_SPLIT_X = 16


def _atlas_labels():
    """Two labelled cuboids inside an unlabelled volume, hemispheres split on x."""
    atlas = np.zeros(SHAPE, dtype=np.uint32)
    hemis = np.zeros(SHAPE, dtype=np.uint32)
    # Region A block: z 6..13, y 8..23, x 8..23 (inclusive)
    atlas[6:14, 8:24, 8:24] = REGION_A
    # Region B block: z 16..21, y 8..23, x 8..23
    atlas[16:22, 8:24, 8:24] = REGION_B
    hemis[atlas > 0] = 1
    hemis[(atlas > 0) & (np.arange(SHAPE[2])[None, None, :] >= HEM_SPLIT_X)] = 2
    return atlas, hemis


@pytest.fixture
def atlas_and_hemispheres():
    return _atlas_labels()


@pytest.fixture
def signal_mask():
    """Segmented 'tracer' with all four assignment fates represented.

    - 16 voxels inside region A, all x < split -> left hemisphere
    - 16 voxels inside region B, all x >= split -> right hemisphere
    - 8 voxels one z-step above the region A block (x < split, so the nearest boundary voxel
      is left too) -> filled at the 200 um cap, counted as region A / left
    - 1 corner voxel ~258 um (12.9 voxels) from the nearest label -> excluded at 200 um,
      filled (into region B) with an infinite cap

    Every voxel's hemisphere is unambiguous, so left + right must equal the whole brain.
    """
    sig = np.zeros(SHAPE, dtype=np.uint8)
    sig[8:10, 12:14, 12:16] = 255   # 2*2*4 = 16 in region A (x 12..15 -> left)
    sig[18:20, 10:12, 18:22] = 255  # 2*2*4 = 16 in region B (x 18..21 -> right)
    sig[5:6, 14:18, 14:16] = 255    # 1*4*2 = 8 outside, 20 um above region A, left of split
    sig[30, 30, 2] = 255            # 1 beyond a 200 um cap, nearest label is region B
    return sig


@pytest.fixture
def volumes_df():
    """Registration volumes.csv rows, named to match structures.csv exactly."""
    structures = pd.read_csv(os.path.join(REPO_ROOT, "structures.csv"))
    names = structures.set_index("id")["name"]
    rows = []
    for region_id, vol in ((REGION_A, 8.0), (REGION_B, 4.0)):
        rows.append({"id": region_id, "structure_name": names[region_id],
                     "structure_id_path": "/997/8/", "hemisphere": "both",
                     "volume_mm3": vol, "left_volume_mm3": vol / 2, "right_volume_mm3": vol / 2})
    return pd.DataFrame(rows)


@pytest.fixture
def structures_csv():
    """The repo's real structures.csv — decoding must join against it, not a stub."""
    return os.path.join(REPO_ROOT, "structures.csv")


@pytest.fixture
def registration_dir(tmp_path, atlas_and_hemispheres, volumes_df):
    atlas, hemis = atlas_and_hemispheres
    reg_dir = tmp_path / "anTEST" / "registration_dir"
    reg_dir.mkdir(parents=True)
    tifffile.imwrite(reg_dir / "registered_atlas.tiff", atlas)
    tifffile.imwrite(reg_dir / "registered_hemispheres.tiff", hemis)
    volumes_df.to_csv(reg_dir / "volumes.csv", index=False)
    return reg_dir


@pytest.fixture
def channel_dir(tmp_path, signal_mask):
    chan = tmp_path / "anTEST" / "FITC"
    chan.mkdir(parents=True)
    tifffile.imwrite(chan / "thresholded_image.tif", signal_mask)
    return chan
