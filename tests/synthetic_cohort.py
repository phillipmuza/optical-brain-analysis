"""
A synthetic cohort, small enough to run in seconds, used by the native-depth tests.

The point of one cohort per dataset is that a new dataset is a config entry and nothing else. This
builds exactly that: a data directory, a data map, three groups, two tracers, and a config COHORTS
entry pointing at them - all fabricated, all with hand-known geometry.

The image is shaped like the real ones: dark mounting medium, dimmer tissue, and a bright rim of
tracer at the brain surface, with a little signal beyond the rim so the deep compartment is not
identically zero. The rim's brightness rises with the animal index, which is enough for the
statistics to have something to find without pretending to be biology.
"""
import numpy as np
import pandas as pd
import tifffile

SHAPE = (64, 64, 64)          # 64 voxels at 20 um = 1.28 mm
VOXEL_UM = 20.0
BRAIN_RADIUS_VOX = 28.0       # the "brain" is everything closer than this to the centre
RIM_MM = 0.06                 # the bright shell, from the surface inwards (3 voxels)

MEDIUM = 0                    # mounting medium
TISSUE = 3000                 # dim, fairly uniform tissue autofluorescence
TISSUE_NOISE = 200            # tight: real tissue is uniform, and a wide spread inflates the MAD
RIM_BASE = 2000               # how much brighter the tracer rim is than tissue
RIM_GAIN = 300                # rim brightness per animal index: the group effect the pipeline looks for
RIM_TRACER = 50               # rim brightness per tracer
DEEP_FRACTION = 0.03          # share of the interior carrying tracer past the threshold
DEEP_GAIN = 1000              # and how bright those voxels are, per animal index


def sphere(shape=SHAPE):
    """Distance from the centre of the volume, in voxels."""
    grid = np.indices(shape, dtype=np.float32)
    centre = np.array(shape, dtype=np.float32)[:, None, None, None] / 2 - 0.5
    return np.sqrt(((grid - centre) ** 2).sum(axis=0))


def brains(shape=SHAPE):
    """(brain, rim) masks for the synthetic volume."""
    distance = sphere(shape)
    brain = distance < BRAIN_RADIUS_VOX
    rim = brain & (distance > BRAIN_RADIUS_VOX - RIM_MM / (VOXEL_UM / 1000))
    return brain, rim


def synthetic_image(animal_index, tracer_index, seed=0):
    """
    One fake light-sheet stack.

    Tissue is uniform, the rim is clearly brighter, and a small fraction of the interior is bright
    too. That last part matters: real tracer penetrates along perivascular spaces, so the deep
    compartment is a measurement rather than a column of zeros - and a measure with no variance at
    all is not something the statistics can be run on.
    """
    rng = np.random.default_rng(seed + 1000 * animal_index + tracer_index)
    brain, rim = brains()
    interior = brain & ~rim

    image = np.full(SHAPE, MEDIUM, dtype=np.uint16)
    image[brain] = TISSUE + rng.integers(0, TISSUE_NOISE, size=int(brain.sum())).astype(np.uint16)
    # the rim is not perfectly uniform: a constant value inside the mask would give a robust SD of
    # zero, a threshold equal to the background, and then no voxel above it at all
    image[rim] = (TISSUE + RIM_BASE + RIM_GAIN * animal_index + RIM_TRACER * tracer_index
                  + rng.integers(0, TISSUE_NOISE, size=int(rim.sum())).astype(np.uint16))
    bright = rng.random(int(interior.sum())) < DEEP_FRACTION
    interior_values = image[interior].astype(np.int32)
    interior_values[bright] += rng.integers(DEEP_GAIN, DEEP_GAIN * (2 + animal_index % 3),
                                            size=int(bright.sum()))
    image[interior] = interior_values.astype(np.uint16)
    return image


DEFAULT_GROUPS = (("Vehicle", 2), ("Medetomidine", 2), ("K/X", 2))


def build_cohort(root, groups=DEFAULT_GROUPS, seed=0):
    """
    Write a cohort under `root` and return (data_dir, data_map_path, animal_ids).

    Every animal gets downsampled/{registration.tif,fitc.tif,txr.tif}; registration.tif is a copy of
    fitc.tif, exactly as in the real cohorts, which is what config's mask_channel_is_tracer records.
    """
    data_dir = root / 'cleared_brains'
    animals, rows = [], []
    index = 0
    for group, count in groups:
        for _ in range(count):
            index += 1
            animal = f'an{index}'
            animal_dir = data_dir / animal / 'downsampled'
            animal_dir.mkdir(parents=True, exist_ok=True)
            fitc = synthetic_image(index, 0, seed=seed)
            txr = synthetic_image(index, 1, seed=seed)
            tifffile.imwrite(animal_dir / 'fitc.tif', fitc)
            tifffile.imwrite(animal_dir / 'txr.tif', txr)
            # a copy on disk, not a symlink: the pipeline reads it as its own file
            tifffile.imwrite(animal_dir / 'registration.tif', fitc)
            animals.append(animal)
            rows.append({'blinded_number': animal, 'treatment': group})
    data_map = root / 'data_map.csv'
    pd.DataFrame(rows).to_csv(data_map, index=False)
    return data_dir, data_map, animals


def cohort_entry(data_dir, data_map, groups=DEFAULT_GROUPS, name='synthetic'):
    """The config.COHORTS value for a cohort built by build_cohort()."""
    palette = ['#7f7f7f', '#4C72B0', '#C1666B', '#8ED081']
    return {
        'data_dir': str(data_dir),
        'data_map': str(data_map),
        'second_modality_csv': None,
        'atlas_annotation': None,
        'atlas_space_dir': None,
        'threshold_summary': None,
        'mask_channel_is_tracer': 'FITC',
        'groups': [{'name': group, 'label': group.replace('Drug_', ''), 'color': palette[k % len(palette)]}
                   for k, (group, _) in enumerate(groups)],
        'reference_group': groups[0][0],
        'exclude_animals': [],
        'example_animal': 'an1',
    }