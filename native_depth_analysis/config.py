r"""
Everything dataset-specific lives in DATASET below. Point that at a directory of animals and a data
map, and nothing else in the pipeline needs to change.

    <data_dir>/<animal>/downsampled/<REFERENCE_IMAGE>       the image the mask is built from
    <data_dir>/<animal>/downsampled/<channel>.tif           one per tracer, same grid as the reference

Everything outside DATASET is a constant, because it has been the same for every dataset this
pipeline has been run on: 20 um isotropic voxels, brainreg run with --orientation asr, a
dorsal/ventral split, and the Perens LSFM mouse atlas (also 20 um). The assumptions that carries are
listed in README.md under "What this assumes"; a dataset that breaks one needs a new constant, not a
new special case.

One dataset at a time: to run another, edit DATASET, or keep a copy of this file per dataset. Results
are namespaced by the dataset's name under results/<name>/, because animal ids repeat across datasets
(both series run an17) and a shared results directory would half-overwrite. Set NATIVE_DEPTH_RESULTS
to one exact directory to override, which is how you point step 04 at results computed earlier.
"""
import hashlib
import json
import os
import re
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RESULTS_ROOT = os.path.join(HERE, 'results')
RESULTS_ENV = 'NATIVE_DEPTH_RESULTS'

# --- constants: the same for every dataset ------------------------------------------------------
REFERENCE_IMAGE = 'registration.tif'    # anatomy reference; ideally NOT a copy of a tracer channel
CHANNELS = {'FITC': 'fitc', 'TxR': 'txr'}    # display name -> file stem under downsampled/
SIDES = ['dorsal', 'ventral']
VOXEL_UM = 20.0                              # isotropic voxel size of the downsampled images

ANIMAL_COLUMN = 'blinded_number'         # column in the data map holding the animal id
TREATMENT_COLUMN = 'treatment'           # column in the data map holding the group

K_MAD = 5.0                   # tracer threshold = background + K_MAD robust SDs, per animal per channel
TISSUE_K = 2.0                # 05: tissue = registration image above background by TISSUE_K robust SDs
MASK_CLOSING_VOX = 2          # 40 um: closes thresholding pits without bridging real gaps
ENVELOPE_CLOSING_VOX = 5      # 100 um: seals ventricles and clefts so they do not count as "surface"
SUBSAMPLE = 4                 # stride when estimating thresholds from the histogram, for speed
SLAB = 40                     # planes per histogram pass in 03; bounds the per-pass temporaries only
DEPTH_LO, DEPTH_HI, DEPTH_STEP = -0.4, 3.0, 0.02      # mm; negative = outside the brain surface
SURFACE_MM = 0.5              # boundary between the surface and deep compartments; BANDS follow it

# The optional second modality (IVIS ex vivo, or whatever replaces it). The column names live here
# because "IVIS-equivalent" is not the same layout everywhere; the CSV itself is in DATASET.
SECOND_MODALITY_COLUMNS = {'animal': 'blinded_number', 'tracer': 'tracer', 'tissue': 'tissue',
                           'total': 'total_radiant_efficiency', 'average': 'avg_radiant_efficiency'}
SECOND_MODALITY_TISSUE_TO_SIDE = {'dorsal_brain': 'dorsal', 'ventral_brain': 'ventral'}
SECOND_MODALITY_LABEL = 'IVIS'       # what the figures call it; override per dataset in DATASET
# native measure -> second-modality column. The two columns differ only by ROI area, so the matched
# light-sheet quantity differs too: an integral for the total, a per-voxel average for the average.
SECOND_MODALITY_PAIRS = (('surface_0_500um', 'total'), ('surface_per_voxel', 'average'))

# The atlas, and the display grid the atlas-space figures are drawn on. ATLAS_BIN=2 takes the
# 20 um atlas to the 40 um grid the figures use.
ATLAS = 'perens_lsfm_mouse_20um'
ATLAS_VOXEL_UM = 20.0
ATLAS_BIN = 2
# Step 05's one external input: a brainglobe install of ATLAS, which lives wherever it was installed.
# The _v1.2 suffix is part of the installed directory name. Override per dataset in DATASET.
ATLAS_ANNOTATION_DEFAULT = r'C:\Users\skgtpm1\.brainglobe\perens_lsfm_mouse_20um_v1.2\annotation.tiff'

# --- the one dict: point this at a dataset ------------------------------------------------------
# name                  namespaces the results directory, and is part of every file's provenance
# data_dir              one folder per animal, each with downsampled/<channel>.tif as above
# data_map              CSV with a column of animal ids (ANIMAL_COLUMN) and one of groups
# second_modality_csv   the per-animal table to cross-validate against, or None
# groups                the treatment groups, in figure order, as name (must match the data map
#                       exactly), label (compact axis label) and color; the first is not special,
#                       reference_group names whichever one every contrast is tested against
# exclude_animals       animals dropped everywhere; 03 and 04 also drop any left over from an
#                       earlier run, so an exclusion added late still takes effect
# example_animal        the animal the QC figures illustrate
# mask_channel_is_tracer
#                       the tracer name when downsampled/REFERENCE_IMAGE is a copy of that tracer's
#                       image, which means the mask follows the tracer's surface rim and "depth 0"
#                       is not independent of the signal (see README); '' if it is its own channel
# atlas_annotation      only for steps 05 and 06; defaults to ATLAS_ANNOTATION_DEFAULT
DATASET = {
    'name': 'NS24122',
    'data_dir': r'E:\tracer_uptake\wt_mice\new_analysis_0926',
    'data_map': r'E:\tracer_uptake\wt_mice\data_map.csv',
    'second_modality_csv': r'E:\tracer_uptake\wt_mice\ivis_raw_data\ex_vivo\brain_data.csv',
    'groups': [
        {'name': 'Vehicle', 'label': 'Vehicle', 'color': '#7f7f7f'},
        {'name': 'NS24122_10mg', 'label': '10mg', 'color': '#4C72B0'},
        {'name': 'NS24122_30mg', 'label': '30mg', 'color': '#C1666B'},
    ],
    'reference_group': 'Vehicle',
    'exclude_animals': ['an12', 'an14', 'an35', 'an25', 'an26'],
    'example_animal': 'an36',
    'mask_channel_is_tracer': 'FITC',
    'atlas_annotation': None,
}

# The anaesthetic series, kept here as the worked example of the other shape - three groups whose
# names are the anaesthetic conditions, with isoflurane as the control, and the tracers and
# processing otherwise identical. Swap it into DATASET above to run it, and confirm at first run
# that the data map's treatment strings match these names character for character; validate() stops
# with a list if they do not.
ANAESTHETIC_DATASET = {
    'name': 'anaesthetic',
    'data_dir': r'D:\anaesthetic_experiments\cleared_brains_new_analysis',
    'data_map': r'D:\anaesthetic_experiments\cleared_brains_new_analysis\data_map.csv',  # TODO(confirm)
    'second_modality_csv': None,                  # TODO(confirm): every series has one
    'groups': [
        {'name': 'Isoflurane', 'label': 'Isoflurane', 'color': '#7f7f7f'},
        {'name': 'Medetomidine', 'label': 'Medetomidine', 'color': '#4C72B0'},
        {'name': 'K/X', 'label': 'K/X', 'color': '#C1666B'},
    ],
    'reference_group': 'Isoflurane',
    'exclude_animals': [],                        # TODO(confirm)
    'example_animal': 'an17',
    'mask_channel_is_tracer': 'FITC',
    'atlas_annotation': None,
}

# --- constants derived from the constants above -------------------------------------------------
VOXEL_MM = VOXEL_UM / 1000
VOXEL_MM3 = VOXEL_MM ** 3
ATLAS_VOXEL_MM = ATLAS_VOXEL_UM / 1000
ATLAS_SPACING_MM = ATLAS_VOXEL_MM * ATLAS_BIN      # how far apart the display planes are


def band_name(prefix, value_mm):
    """Bands are named after their boundary, so SURFACE_MM is the only place the number lives."""
    return f'{prefix}_{int(round(value_mm * 1000))}um'


SHALLOW_BAND = 'surface_0_200um'                           # the shell the README warns about
SURFACE_BAND = band_name('surface_0', SURFACE_MM)          # surface_0_500um -> (0, SURFACE_MM)
DEEP_BAND = band_name('deep', SURFACE_MM) + '_plus'        # deep_500um_plus -> (SURFACE_MM, inf)
OUTSIDE_BAND = 'outside_surface'                           # (-inf, 0): measured by 03, excluded by 04
INSIDE_BAND = 'inside_total'                               # (0, inf)
BANDS = {SHALLOW_BAND: (0.0, 0.2), SURFACE_BAND: (0.0, SURFACE_MM),
         DEEP_BAND: (SURFACE_MM, float('inf')), INSIDE_BAND: (0.0, float('inf'))}

# OUTSIDE_BAND is deliberately not in BANDS above. It is where the pial rim and the basal cisterns
# sit - up to ~17% of the thresholded signal in these datasets - so 03 measures it and 03's QC prints
# it, but no band in 04 covers it and it never reaches the statistics. Putting it in BANDS would
# change what is tested and how many comparisons there are: a scientific decision, not formatting.

# --- names resolved by resolve(); do not edit ---------------------------------------------------
DATASET_NAME = ''
DATA_DIR = DATA_MAP = SECOND_MODALITY_CSV = SECOND_MODALITY_NAME = ''
ATLAS_ANNOTATION = ''
ATLAS_SPACE_DIR = THRESHOLD_SUMMARY = ''
MASK_CHANNEL_IS_TRACER = ''
EXAMPLE_ANIMAL = ''
GROUPS = []
GROUP_ORDER = []
GROUP_COLORS = {}
SHORT_LABELS = {}
REFERENCE_GROUP = ''
CONTRASTS = []
EXCLUDE_ANIMALS = []
RESULTS_DIR = FIG_DIR = MASK_DIR = DEPTH_DIR = OUT_DIR = QC_DIR = ''

_data_map_cache = None


def resolve():
    """
    Resolve every name that depends on DATASET. Runs once at import, and is called again by the tests
    after they patch DATASET.

    This is why no script may snapshot these names at import time: a module-level alias or a default
    argument in a function would keep whatever DATASET said first, and a test could not change it.
    Read `config.<name>` inside the function that uses it.
    """
    global DATASET_NAME, GROUPS, DATA_DIR, DATA_MAP, SECOND_MODALITY_CSV, SECOND_MODALITY_NAME
    global ATLAS_ANNOTATION, MASK_CHANNEL_IS_TRACER, EXAMPLE_ANIMAL
    global GROUP_ORDER, GROUP_COLORS, SHORT_LABELS, REFERENCE_GROUP, CONTRASTS, EXCLUDE_ANIMALS
    global RESULTS_DIR, FIG_DIR, MASK_DIR, DEPTH_DIR, OUT_DIR, QC_DIR
    global ATLAS_SPACE_DIR, THRESHOLD_SUMMARY, _data_map_cache

    spec = DATASET
    DATASET_NAME = spec['name']
    GROUPS = [dict(g) for g in spec['groups']]
    GROUP_ORDER = [g['name'] for g in GROUPS]
    GROUP_COLORS = {g['name']: g['color'] for g in GROUPS}
    SHORT_LABELS = {g['name']: g['label'] for g in GROUPS if g['label'] != g['name']}
    REFERENCE_GROUP = spec['reference_group']
    CONTRASTS = [(g, REFERENCE_GROUP) for g in GROUP_ORDER if g != REFERENCE_GROUP]
    EXCLUDE_ANIMALS = list(spec['exclude_animals'])
    DATA_DIR = spec['data_dir']
    DATA_MAP = spec['data_map']
    SECOND_MODALITY_CSV = spec.get('second_modality_csv') or ''
    SECOND_MODALITY_NAME = spec.get('second_modality_label', SECOND_MODALITY_LABEL)
    ATLAS_ANNOTATION = spec.get('atlas_annotation') or ATLAS_ANNOTATION_DEFAULT
    MASK_CHANNEL_IS_TRACER = spec.get('mask_channel_is_tracer') or ''
    EXAMPLE_ANIMAL = spec['example_animal']
    _data_map_cache = None

    # an explicit override points at one exact directory (results computed earlier); otherwise the
    # dataset gets its own namespace, because animal ids repeat between datasets
    RESULTS_DIR = os.environ.get(RESULTS_ENV) or os.path.join(RESULTS_ROOT, DATASET_NAME)
    MASK_DIR = os.path.join(RESULTS_DIR, 'masks')
    DEPTH_DIR = os.path.join(RESULTS_DIR, 'depth')
    FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
    OUT_DIR = os.path.join(RESULTS_DIR, 'outputs')
    QC_DIR = os.path.join(RESULTS_DIR, 'qc')
    # 05 writes these and 06 reads them; dataset can override them to reuse volumes computed elsewhere
    ATLAS_SPACE_DIR = spec.get('atlas_space_dir') or os.path.join(RESULTS_DIR, 'atlas_space')
    THRESHOLD_SUMMARY = spec.get('threshold_summary') or os.path.join(ATLAS_SPACE_DIR,
                                                                     'threshold_summary.csv')

    # Created up front so every step can write its first file immediately. Scripts also makedirs
    # before use, but 01 writes its QC csv before the only makedirs call it has (inside figure()).
    for directory in (RESULTS_DIR, MASK_DIR, DEPTH_DIR, FIG_DIR, OUT_DIR, QC_DIR, ATLAS_SPACE_DIR):
        os.makedirs(directory, exist_ok=True)
    return DATASET_NAME


def short(group):
    """Compact label for a group name."""
    return SHORT_LABELS.get(group, group)


def sort_key(name):
    """Sort animal ids the way a person would: an2 before an10."""
    digits = re.findall(r'\d+', name)
    return (int(digits[0]) if digits else 0, name)


def load_data_map():
    """The data map, with the animal id column renamed to `animal_number` and set as the index."""
    global _data_map_cache
    if _data_map_cache is None:
        table = pd.read_csv(DATA_MAP, encoding='utf-8-sig').rename(
            columns={ANIMAL_COLUMN: 'animal_number', TREATMENT_COLUMN: 'treatment'})
        _data_map_cache = table.set_index('animal_number')
    return _data_map_cache


def treatment_of(animal):
    """The group string the data map gives an animal."""
    return str(load_data_map().loc[animal, 'treatment'])


def animals(with_images=True):
    """Every animal in the data map, minus the exclusions, in id order."""
    found = [a for a in load_data_map().index if a not in EXCLUDE_ANIMALS]
    if with_images:
        found = [a for a in found if os.path.isfile(image_path(a, REFERENCE_IMAGE))]
    return sorted(found, key=sort_key)


def group_members(treatment, strict=True):
    """
    The animals of one group, in id order.

    An empty group reaches the statistics as NaN bars, a figure and exit code 0, so by default this
    refuses rather than returning []. validate() passes strict=False so it can report every problem
    at once.
    """
    members = [a for a in animals() if treatment_of(a) == treatment]
    if strict and not members:
        raise SystemExit(f'group {treatment!r} has no animals with images under {DATA_DIR} - check '
                         f'the group names against the {TREATMENT_COLUMN} column of {DATA_MAP}')
    return members


def groups():
    """Every group's members, in the configured order."""
    return {treatment: group_members(treatment) for treatment in GROUP_ORDER}


def image_path(animal, filename):
    """One file under an animal's downsampled/ directory."""
    return os.path.join(DATA_DIR, animal, 'downsampled', filename)


def channel_files():
    """Channel display name -> filename, including the reference image."""
    return {'reference': REFERENCE_IMAGE,
            **{name: f'{stem}.tif' for name, stem in CHANNELS.items()}}


def provenance():
    """
    A fingerprint of everything that determines the numbers, written into every mask and depth
    profile so a later step can tell whether the file in front of it was produced by the current
    configuration. Same inputs -> same string.

    Paths are deliberately *not* in it: moving a data directory does not change a measurement, and
    the dataset name plus the run manifest already record where the files came from. The exclusion
    list is not in it either, and deliberately: exclusions decide *which* animals are analysed, not
    how one is measured, so changing them must not invalidate profiles that are still correct - 03
    and 04 drop excluded animals at read time instead. Adding either would force a 90-minute re-run
    for a change that alters no number.
    """
    material = json.dumps({'dataset': DATASET_NAME, 'channels': CHANNELS, 'sides': SIDES,
                           'voxel_um': VOXEL_UM, 'mask_closing_vox': MASK_CLOSING_VOX,
                           'envelope_closing_vox': ENVELOPE_CLOSING_VOX, 'k_mad': K_MAD,
                           'depth_lo': DEPTH_LO, 'depth_hi': DEPTH_HI, 'depth_step': DEPTH_STEP,
                           'surface_mm': SURFACE_MM}, sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def write_manifest(stage, extra=None):
    """
    Record this run next to the results it produced: which dataset, which parameters, which animals.

    Call it after validate(), so the dataset is known to be loadable. One file per dataset, keyed by
    stage, so re-running a step replaces that step's record and leaves the others alone.
    """
    path = os.path.join(RESULTS_DIR, 'run_manifest.json')
    manifest = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            manifest = {}
    manifest[stage] = {
        'dataset': DATASET_NAME,
        'provenance': provenance(),
        'when': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': describe(),
        'animals': {treatment: group_members(treatment, strict=False) for treatment in GROUP_ORDER},
        **(extra or {}),
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def describe():
    """The resolved dataset, for the console and the run manifest. Reads no data, so it is safe to
    print before validate() has established that the paths even exist."""
    return '\n'.join([
        f'dataset         {DATASET_NAME}',
        f'data            {DATA_DIR}',
        f'data map        {DATA_MAP}',
        f'second modality {SECOND_MODALITY_CSV or "(none configured)"}',
        f'atlas           {ATLAS} at {ATLAS_VOXEL_UM:g} um, binned x{ATLAS_BIN} for display',
        f'voxel           {VOXEL_UM:g} um isotropic',
        'groups          ' + ', '.join(f'{g["name"]} [{g["label"]}]' for g in GROUPS),
        f'reference       {REFERENCE_GROUP}',
        f'excluded        {", ".join(EXCLUDE_ANIMALS) if EXCLUDE_ANIMALS else "(none)"}',
        f'surface band    {SURFACE_BAND} (0-{SURFACE_MM:g} mm), deep {DEEP_BAND}',
        'mask channel    ' + REFERENCE_IMAGE
        + (f' is a copy of the {MASK_CHANNEL_IS_TRACER} channel' if MASK_CHANNEL_IS_TRACER else ''),
        f'results         {RESULTS_DIR}',
    ])


def validate():
    """
    Check everything the pipeline is about to assume and stop with a list of problems, rather than a
    traceback halfway through a long run or - worse - a figure that silently drops a group.

    Returns lines ready to print: 'note: ...' for the census it establishes, 'warning: ...' for
    things that are legal but worth reading.
    """
    problems, warnings = [], []
    if not RESULTS_DIR:
        raise SystemExit('config.resolve() has not run, so no dataset is loaded')
    if SURFACE_MM <= 0 or SURFACE_MM >= DEPTH_HI:
        problems.append(f'SURFACE_MM = {SURFACE_MM} must lie inside (0, {DEPTH_HI})')
    if len(set(GROUP_ORDER)) != len(GROUP_ORDER):
        problems.append(f'duplicate group names in DATASET: {GROUP_ORDER}')
    if REFERENCE_GROUP not in GROUP_ORDER:
        problems.append(f'reference_group {REFERENCE_GROUP!r} is not one of {GROUP_ORDER}')
    if len(set(SHORT_LABELS.values())) != len(SHORT_LABELS):
        warnings.append(f'two groups share a short label: {SHORT_LABELS} - figures will be ambiguous')
    if not CHANNELS:
        problems.append('CHANNELS is empty, so there is nothing to measure')
    if not os.path.isdir(DATA_DIR):
        problems.append(f'data_dir does not exist: {DATA_DIR}')
    if not os.path.isfile(DATA_MAP):
        problems.append(f'data map does not exist: {DATA_MAP}')
    if SECOND_MODALITY_CSV and not os.path.isfile(SECOND_MODALITY_CSV):
        warnings.append('second_modality_csv does not exist, step 04 will skip the cross-validation: '
                        + SECOND_MODALITY_CSV)
    # The atlas annotation is a real external input (a brainglobe install). The other two are written
    # by step 05, so their absence is normal until it has run - it is 06 that needs telling.
    if ATLAS_ANNOTATION and not os.path.isfile(ATLAS_ANNOTATION):
        warnings.append(f'atlas_annotation does not exist, step 05 cannot run: {ATLAS_ANNOTATION}')
    if not os.path.isfile(THRESHOLD_SUMMARY) or not os.path.isdir(ATLAS_SPACE_DIR):
        warnings.append('no atlas-space volumes for this dataset yet, step 06 cannot run until 05 '
                        f'has: {ATLAS_SPACE_DIR}')
    if problems:
        raise SystemExit('config problems:\n  - ' + '\n  - '.join(problems))

    # the data map's own column names, before the rename above made the check vacuous
    raw_columns = pd.read_csv(DATA_MAP, encoding='utf-8-sig', nrows=0).columns
    for role, column in (('animal id', ANIMAL_COLUMN), ('treatment', TREATMENT_COLUMN)):
        if column not in raw_columns:
            problems.append(f'data map has no {role} column {column!r}; it has: {list(raw_columns)}')
    if problems:
        raise SystemExit('config problems:\n  - ' + '\n  - '.join(problems))

    on_disk = sorted([name for name in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, name))], key=sort_key)
    in_map = [str(a) for a in load_data_map().index]
    missing_from_map = [a for a in on_disk if a not in in_map and a not in EXCLUDE_ANIMALS]
    missing_on_disk = [a for a in in_map if a not in on_disk and a not in EXCLUDE_ANIMALS]
    if missing_from_map:
        problems.append(f'folders present but not in the data map, so they would be ignored: '
                        f'{missing_from_map}')
    if missing_on_disk:
        warnings.append(f'data map rows with no folder: {missing_on_disk}')

    for animal in animals():
        for name, filename in channel_files().items():
            if not os.path.isfile(image_path(animal, filename)):
                problems.append(f'{animal}: missing downsampled/{filename} ({name})')
        treatment = treatment_of(animal)
        if treatment not in GROUP_ORDER:
            problems.append(f'{animal}: treatment {treatment!r} is not one of the groups in DATASET '
                            f'{GROUP_ORDER}, so it would vanish from every figure')
    for treatment in GROUP_ORDER:
        if not group_members(treatment, strict=False):
            problems.append(f'group {treatment!r} has no animals (after exclusions)')

    if problems:
        raise SystemExit('config problems:\n  - ' + '\n  - '.join(problems))

    # the census, so a run states up front which animals it is about to analyse
    notes = [f'note: {treatment} n={len(group_members(treatment, strict=False))}'
             for treatment in GROUP_ORDER]
    notes.append(f'note: {len(animals())} animals in total, {len(EXCLUDE_ANIMALS)} excluded')
    return notes + [f'warning: {w}' for w in warnings]


resolve()