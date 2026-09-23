r"""
Everything dataset-specific lives here. To run this pipeline on another cohort, edit this file and
nothing else.

The values below are the ones used for the NS24122 cohort, so the pipeline runs as-is on that data
and can be checked against the published numbers before being pointed somewhere new.
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# --- where the data is -------------------------------------------------------------------------
# DATA_DIR holds one folder per animal. Each animal folder must contain:
#   downsampled/<REFERENCE_IMAGE>     the image the mask is built from
#   downsampled/<channel>.tif         one per entry in CHANNELS, same grid as the reference
DATA_DIR = r'E:\tracer_uptake\wt_mice\new_analysis_0926'
DATA_MAP = r'E:\tracer_uptake\wt_mice\data_map.csv'

# Optional second modality for the cross-validation in 04. Set to None to skip that section.
# Needs columns: animal id, tracer, tissue ('dorsal_brain'/'ventral_brain'), and the two
# radiant-efficiency columns.
IVIS_CSV = r'E:\tracer_uptake\wt_mice\ivis_raw_data\ex_vivo\brain_data.csv'
IVIS_ANIMAL_COLUMN = 'blinded_number'   # the animal id column in IVIS_CSV

ANIMAL_COLUMN = 'blinded_number'        # column in DATA_MAP holding the animal id
TREATMENT_COLUMN = 'treatment'          # column in DATA_MAP holding the group

# --- images ------------------------------------------------------------------------------------
REFERENCE_IMAGE = 'registration.tif'    # anatomy reference; ideally NOT a tracer channel (see README)
CHANNELS = {'FITC': 'fitc', 'TxR': 'txr'}       # display name -> folder/file stem under downsampled/
VOXEL_UM = 20.0                                  # isotropic voxel size of the downsampled images

# --- groups ------------------------------------------------------------------------------------
GROUP_ORDER = ['Vehicle', 'NS24122_10mg', 'NS24122_30mg']
REFERENCE_GROUP = 'Vehicle'                      # every contrast is against this group
EXCLUDE_ANIMALS = ['an12', 'an14', 'an35', 'an25', 'an26']    # dropped everywhere
GROUP_COLORS = {'Vehicle': '#7f7f7f', 'NS24122_10mg': '#4C72B0', 'NS24122_30mg': '#C1666B'}
SHORT_LABELS = {'NS24122_10mg': '10mg', 'NS24122_30mg': '30mg'}   # optional, for compact axis labels

# --- analysis parameters -------------------------------------------------------------------------
K_MAD = 5.0                   # tracer threshold = background + K_MAD robust SDs, per animal per channel
MASK_CLOSING_VOX = 2          # 40 um: closes thresholding pits without bridging real gaps
ENVELOPE_CLOSING_VOX = 5      # 100 um: seals ventricles and clefts so they do not count as "surface"
SUBSAMPLE = 4                 # stride when estimating thresholds from the histogram, for speed
SLAB = 40                     # planes held in memory at once in 03
DEPTH_LO, DEPTH_HI, DEPTH_STEP = -0.4, 3.0, 0.02      # mm; negative = outside the brain surface
SURFACE_MM = 0.5              # boundary between the surface and deep compartments
EXAMPLE_ANIMAL = 'an36'       # the animal 01 draws its illustration from

# --- outputs ---------------------------------------------------------------------------------------
# Override with the SURFACE_TRACER_RESULTS environment variable to write somewhere else, or to point
# the analysis at results computed earlier.
RESULTS_DIR = os.environ.get('SURFACE_TRACER_RESULTS', os.path.join(HERE, 'results'))
MASK_DIR = os.path.join(RESULTS_DIR, 'masks')
DEPTH_DIR = os.path.join(RESULTS_DIR, 'depth')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
OUT_DIR = os.path.join(RESULTS_DIR, 'outputs')
QC_DIR = os.path.join(RESULTS_DIR, 'qc')

# --- derived (do not edit) ---------------------------------------------------------------------
VOXEL_MM = VOXEL_UM / 1000
VOXEL_MM3 = VOXEL_MM ** 3
CONTRASTS = [(group, REFERENCE_GROUP) for group in GROUP_ORDER if group != REFERENCE_GROUP]


def short(group):
    """Compact label for a group name."""
    return SHORT_LABELS.get(group, group)


def sort_key(name):
    """Sort animal ids the way a person would: an2 before an10."""
    digits = re.findall(r'\d+', name)
    return (int(digits[0]) if digits else 0, name)


def load_data_map():
    """The data map, with the animal id column renamed to `animal_number` and set as the index."""
    table = pd.read_csv(DATA_MAP, encoding='utf-8-sig').rename(
        columns={ANIMAL_COLUMN: 'animal_number', TREATMENT_COLUMN: 'treatment'})
    return table.set_index('animal_number')


def animals():
    """Every animal folder that has the images this pipeline needs, minus the exclusions."""
    found = []
    for name in os.listdir(DATA_DIR):
        reference = os.path.join(DATA_DIR, name, 'downsampled', REFERENCE_IMAGE)
        if os.path.isfile(reference) and name not in EXCLUDE_ANIMALS:
            found.append(name)
    return sorted(found, key=sort_key)
