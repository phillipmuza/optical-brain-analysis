r"""
Everything dataset-specific lives here. Pointing the pipeline at another cohort means adding or
editing one entry in COHORTS and nothing else.

A cohort is a directory of animal directories plus a data map:

    <data_dir>/<animal>/downsampled/<REFERENCE_IMAGE>       the image the mask is built from
    <data_dir>/<animal>/downsampled/<channel>.tif           one per tracer, same grid as the reference

Everything outside COHORTS is a constant, because it is the same for every cohort this pipeline has
been run on: 20 um isotropic voxels, brainreg run with --orientation asr, a dorsal/ventral split,
and the Perens LSFM mouse atlas (also 20 um). The assumptions that carries are listed in README.md
under "What this assumes"; a cohort that breaks one needs a new constant, not a new special case.

Select a cohort with --cohort on any of the scripts, or NATIVE_DEPTH_COHORT in the environment.
Without either it uses DEFAULT_COHORT.

Results are namespaced per cohort, results/<cohort>/... . Animal ids are NOT unique across cohorts -
the NS24122 and anaesthetic series both run an1, an2, an17 - so two cohorts must never share a
results directory. (Set NATIVE_DEPTH_RESULTS to an exact path to override, which is how you point
step 04 at results computed earlier.)
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

# --- constants: the same for every cohort ------------------------------------------------------
REFERENCE_IMAGE = 'registration.tif'    # anatomy reference; ideally NOT a copy of a tracer channel
CHANNELS = {'FITC': 'fitc', 'TxR': 'txr'}    # display name -> file stem under downsampled/
SIDES = ['dorsal', 'ventral']
VOXEL_UM = 20.0                              # isotropic voxel size of the downsampled images

ANIMAL_COLUMN = 'blinded_number'         # column in the data map holding the animal id
TREATMENT_COLUMN = 'treatment'           # column in the data map holding the group

K_MAD = 5.0                   # tracer threshold = background + K_MAD robust SDs, per animal per channel
MASK_CLOSING_VOX = 2          # 40 um: closes thresholding pits without bridging real gaps
ENVELOPE_CLOSING_VOX = 5      # 100 um: seals ventricles and clefts so they do not count as "surface"
SUBSAMPLE = 4                 # stride when estimating thresholds from the histogram, for speed
SLAB = 40                     # planes per histogram pass in 03; bounds the per-pass temporaries only
DEPTH_LO, DEPTH_HI, DEPTH_STEP = -0.4, 3.0, 0.02      # mm; negative = outside the brain surface
SURFACE_MM = 0.5              # boundary between the surface and deep compartments; BANDS follow it

# The optional second modality (IVIS ex vivo, or whatever replaces it). The column names live here
# because "IVIS-equivalent" is not the same layout everywhere; the CSV itself is set per cohort.
SECOND_MODALITY_COLUMNS = {'animal': 'blinded_number', 'tracer': 'tracer', 'tissue': 'tissue',
                           'total': 'total_radiant_efficiency', 'average': 'avg_radiant_efficiency'}
SECOND_MODALITY_TISSUE_TO_SIDE = {'dorsal_brain': 'dorsal', 'ventral_brain': 'ventral'}
SECOND_MODALITY_LABEL = 'IVIS'       # default name for the modality in labels; override per cohort
TISSUE_K = 2.0                # step 05: tissue = registration image > background - TISSUE_K robust SDs
# native measure -> second-modality column. The two columns differ only by ROI area, so the matched
# light-sheet quantity differs too: an integral for the total, a per-voxel average for the average.
SECOND_MODALITY_PAIRS = (('surface_0_500um', 'total'), ('surface_per_voxel', 'average'))

# The atlas, and the display grid the atlas-space figures are drawn on. ATLAS_BIN=2 takes the
# 20 um atlas to the 40 um grid documented for the resampled volumes - check it against
# atlas_space_images.BIN if those volumes are ever regenerated.
ATLAS = 'perens_lsfm_mouse_20um'
ATLAS_VOXEL_UM = 20.0
ATLAS_BIN = 2
# The one atlas input this pipeline cannot produce for itself: a brainglobe install of ATLAS, which
# lives wherever it was installed. The _v1.2 suffix is part of the installed directory name. A cohort
# with a different atlas overrides it in its own entry.
ATLAS_ANNOTATION_DEFAULT = r'C:\Users\skgtpm1\.brainglobe\perens_lsfm_mouse_20um_v1.2\annotation.tiff'

# --- cohorts: the only part that changes between datasets --------------------------------------
# mask_channel_is_tracer: set to the tracer name when downsampled/REFERENCE_IMAGE is a copy of that
# tracer channel's image. The mask then follows the tracer's surface rim, so "depth 0" is not
# independent of the signal and the outermost shell has to be read with suspicion (see README).
COHORTS = {
    'NS24122': {
        'data_dir': r'E:\tracer_uptake\wt_mice\new_analysis_0926',
        'data_map': r'E:\tracer_uptake\wt_mice\data_map.csv',
        'second_modality_csv': r'E:\tracer_uptake\wt_mice\ivis_raw_data\ex_vivo\brain_data.csv',
        # Step 05 and 06 only. ATLAS_SPACE_DIR and THRESHOLD_SUMMARY are written by 05 into the cohort's
        # own results directory, so they are left unset here - set them only to reuse volumes computed
        # elsewhere (the original intensity arm kept them in NS24122_intensity/). ATLAS_ANNOTATION
        # comes from ATLAS_ANNOTATION_DEFAULT.
        'mask_channel_is_tracer': 'FITC',
        'groups': [
            {'name': 'Vehicle', 'label': 'Vehicle', 'color': '#7f7f7f'},
            {'name': 'NS24122_10mg', 'label': '10mg', 'color': '#4C72B0'},
            {'name': 'NS24122_30mg', 'label': '30mg', 'color': '#C1666B'},
        ],
        'reference_group': 'Vehicle',
        'exclude_animals': ['an12', 'an14', 'an35', 'an25', 'an26'],
        'example_animal': 'an36',
    },

    # TODO(confirm) - the values marked below are the ones not yet seen; the rest come from the
    # cohort directory tree. The group names are the anaesthetic conditions as they appear in the
    # data map's treatment column, with isoflurane as the control. validate() fails loudly on a
    # wrong path or a group name that does not match the data map, so a first run reports these
    # rather than mis-analysing quietly.
    'anaesthetic': {
        'data_dir': r'D:\anaesthetic_experiments\cleared_brains_new_analysis',
        'data_map': r'D:\anaesthetic_experiments\cleared_brains_new_analysis\data_map.csv',  # TODO(confirm)
        'second_modality_csv': None,                  # TODO(confirm): all cohorts have one
        # Step 05 and 06 only, and unset: 05 has not been run for this cohort, so it has no atlas-space
        # volumes yet. 05 writes them into this cohort's results directory; leave these None unless
        # you are reusing volumes computed elsewhere.
        'mask_channel_is_tracer': 'FITC',
        'groups': [
            {'name': 'Isoflurane', 'label': 'Isoflurane', 'color': '#7f7f7f'},
            {'name': 'Medetomidine', 'label': 'Medetomidine', 'color': '#4C72B0'},
            {'name': 'K/X', 'label': 'K/X', 'color': '#C1666B'},
        ],
        'reference_group': 'Isoflurane',
        'exclude_animals': [],                        # TODO(confirm)
        'example_animal': 'an17',
    },
}

DEFAULT_COHORT = 'NS24122'
COHORT_ENV = 'NATIVE_DEPTH_COHORT'
RESULTS_ENV = 'NATIVE_DEPTH_RESULTS'

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
# sit - up to ~17% of the thresholded signal in these cohorts - so 03 measures it and 03's QC prints
# it, but no band in 04 covers it and it never reaches the statistics. Putting it in BANDS would
# change what is tested and how many comparisons there are: a scientific decision, not formatting.

# --- names resolved by select(); do not edit ----------------------------------------------------
COHORT = ''
GROUPS = []                   # the cohort's groups, in order, as {name, label, color}
DATA_DIR = DATA_MAP = SECOND_MODALITY_CSV = SECOND_MODALITY_NAME = ''
ATLAS_ANNOTATION = ''
ATLAS_SPACE_DIR = THRESHOLD_SUMMARY = ''
MASK_CHANNEL_IS_TRACER = ''
EXAMPLE_ANIMAL = ''
GROUP_ORDER = []
GROUP_COLORS = {}
SHORT_LABELS = {}
REFERENCE_GROUP = ''
CONTRASTS = []
EXCLUDE_ANIMALS = []
RESULTS_DIR = FIG_DIR = MASK_DIR = DEPTH_DIR = OUT_DIR = QC_DIR = ''

_data_map_cache = None


def select(cohort=None):
    """
    Point the module at a cohort and resolve every name that depends on it.

    Called by each script's main() before it touches any data, so that `--cohort` and the
    NATIVE_DEPTH_COHORT environment variable behave identically. Scripts therefore must not
    snapshot these names at import time - a module-level alias or a default argument would keep
    whichever cohort was selected first.
    """
    global COHORT, GROUPS, DATA_DIR, DATA_MAP, SECOND_MODALITY_CSV, MASK_CHANNEL_IS_TRACER
    global ATLAS_ANNOTATION, ATLAS_SPACE_DIR, THRESHOLD_SUMMARY, EXAMPLE_ANIMAL
    global SECOND_MODALITY_NAME
    global GROUP_ORDER, GROUP_COLORS, SHORT_LABELS, REFERENCE_GROUP, CONTRASTS, EXCLUDE_ANIMALS
    global RESULTS_DIR, FIG_DIR, MASK_DIR, DEPTH_DIR, OUT_DIR, QC_DIR, _data_map_cache

    name = cohort or os.environ.get(COHORT_ENV) or DEFAULT_COHORT
    if name not in COHORTS:
        raise SystemExit(f'unknown cohort {name!r}; known cohorts: {", ".join(sorted(COHORTS))}')

    spec = COHORTS[name]
    COHORT = name
    GROUPS = [dict(g) for g in spec['groups']]
    GROUP_ORDER = [g['name'] for g in GROUPS]
    GROUP_COLORS = {g['name']: g['color'] for g in GROUPS}
    SHORT_LABELS = {g['name']: g['label'] for g in GROUPS if g['label'] != g['name']}
    REFERENCE_GROUP = spec['reference_group']
    CONTRASTS = [(g, REFERENCE_GROUP) for g in GROUP_ORDER if g != REFERENCE_GROUP]
    EXCLUDE_ANIMALS = list(spec['exclude_animals'])
    DATA_DIR = spec['data_dir']
    DATA_MAP = spec['data_map']
    SECOND_MODALITY_CSV = spec['second_modality_csv'] or ''
    SECOND_MODALITY_NAME = spec.get('second_modality_label', SECOND_MODALITY_LABEL)
    ATLAS_ANNOTATION = spec.get('atlas_annotation') or ATLAS_ANNOTATION_DEFAULT
    MASK_CHANNEL_IS_TRACER = spec['mask_channel_is_tracer'] or ''
    EXAMPLE_ANIMAL = spec['example_animal']
    _data_map_cache = None

    # an explicit override points at one exact directory (results computed earlier); otherwise the
    # cohort gets its own namespace, because animal ids repeat across cohorts
    RESULTS_DIR = os.environ.get(RESULTS_ENV) or os.path.join(RESULTS_ROOT, name)
    MASK_DIR = os.path.join(RESULTS_DIR, 'masks')
    DEPTH_DIR = os.path.join(RESULTS_DIR, 'depth')
    FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
    OUT_DIR = os.path.join(RESULTS_DIR, 'outputs')
    QC_DIR = os.path.join(RESULTS_DIR, 'qc')
    # Step 05's outputs, and step 06's inputs. Namespaced per cohort for the same reason everything
    # else is: animal ids repeat across cohorts, and the code this was ported from wrote
    # atlas_space/<animal>.npz for whichever cohort its hardcoded data path pointed at. A cohort
    # spec can still override these to reuse volumes computed elsewhere.
    ATLAS_SPACE_DIR = spec.get('atlas_space_dir') or os.path.join(RESULTS_DIR, 'atlas_space')
    THRESHOLD_SUMMARY = spec.get('threshold_summary') or os.path.join(ATLAS_SPACE_DIR, 'threshold_summary.csv')

    # Created up front so every step can write its first file immediately. Scripts also makedirs
    # before use, but 01 writes its QC csv before the only makedirs call it has (inside figure()).
    for directory in (RESULTS_DIR, MASK_DIR, DEPTH_DIR, FIG_DIR, OUT_DIR, QC_DIR, ATLAS_SPACE_DIR):
        os.makedirs(directory, exist_ok=True)
    return name


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


def provenance():
    """
    A fingerprint of everything that determines the numbers, written into every mask and depth
    profile so a later step can tell whether the file in front of it was produced by the current
    configuration. Same inputs -> same string.

    Paths are deliberately *not* in it: moving a data directory does not change a measurement, and
    the cohort name plus the run manifest already record where the files came from.
    """
    material = json.dumps({'cohort': COHORT, 'channels': CHANNELS, 'sides': SIDES, 'voxel_um': VOXEL_UM,
                           'mask_closing_vox': MASK_CLOSING_VOX, 'envelope_closing_vox': ENVELOPE_CLOSING_VOX,
                           'k_mad': K_MAD, 'depth_lo': DEPTH_LO, 'depth_hi': DEPTH_HI,
                           'depth_step': DEPTH_STEP, 'surface_mm': SURFACE_MM},
                          sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def write_manifest(stage, extra=None):
    """
    Record this run next to the results it produced: which cohort, which parameters, which animals.

    Call it after validate(), so the cohort is known to be loadable. One file per cohort, keyed by
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
        'cohort': COHORT,
        'provenance': provenance(),
        'when': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': describe(),
        'animals': {treatment: group_members(treatment, strict=False) for treatment in GROUP_ORDER},
        **(extra or {}),
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def channel_files():
    """Channel display name -> filename, including the reference image."""
    return {'reference': REFERENCE_IMAGE,
            **{name: f'{stem}.tif' for name, stem in CHANNELS.items()}}


def describe():
    """The resolved cohort, for the console and the run manifest. Reads no data, so it is safe to
    print before validate() has established that the paths even exist."""
    return '\n'.join([
        f'cohort          {COHORT}',
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
    if not RESULTS_DIR:
        raise SystemExit('config.select() has not been called, so no cohort is loaded')

    problems, warnings = [], []
    if SURFACE_MM <= 0 or SURFACE_MM >= DEPTH_HI:
        problems.append(f'SURFACE_MM = {SURFACE_MM} must lie inside (0, {DEPTH_HI})')
    if len(set(GROUP_ORDER)) != len(GROUP_ORDER):
        problems.append(f'duplicate group names in COHORTS[{COHORT!r}]: {GROUP_ORDER}')
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
        warnings.append(f'no atlas-space volumes for this cohort yet, step 06 cannot run until 05 has: '
                        f'{ATLAS_SPACE_DIR}')
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
            problems.append(f'{animal}: treatment {treatment!r} is not one of the groups in '
                            f'COHORTS[{COHORT!r}] {GROUP_ORDER}, so it would vanish from every figure')
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