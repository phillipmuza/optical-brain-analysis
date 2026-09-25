r"""
The inputs stage 06 needs, produced from this cohort's own registration: raw tracer intensity on the
atlas grid, and one detection threshold per animal x tracer.

This is a port of the two things that used to live outside this repository - the resampling in
NS24122_intensity/atlas_space_images.py and the per-animal threshold in
NS24122_intensity/fixed_threshold_intensity.py - reduced to exactly what the atlas-space stage
consumes, so that stage no longer depends on another folder's constants or its cache.

Why resample at all: the native-space analysis (03, 04) measures each animal against its own surface,
which is right for *how much* tracer there is. To see *where* it is, animals have to share a grid, so
each sample voxel is dropped into the atlas voxel its brainreg deformation field points at and each
atlas voxel takes the mean raw intensity of the sample voxels that landed there. Nothing is
normalised or thresholded in that product: the saved volumes are raw intensities, so a figure can
window all animals together (absolute, valid only because acquisition settings were identical across
the cohort) or window each animal against its own background.

Voxels are kept wherever there is tissue, NOT only inside the registered atlas, so tracer on the pial
surface and in the basal cisterns - the compartment the atlas does not cover - stays visible. Sample
voxels whose atlas coordinate falls outside the atlas volume are dropped rather than clipped.

Saved per animal to results/<cohort>/atlas_space/<animal>.npz, on the atlas grid binned by
config.ATLAS_BIN (40 um at BIN=2):
  <tracer>          mean raw intensity per atlas voxel, uint16, 0 where no sample voxel landed
  n                 how many sample voxels landed in each atlas voxel, uint8 (clipped), for coverage
  bg_<tracer>       that animal's raw background, for the per-animal scaling
  cohort, provenance  which configuration produced the file
and once per cohort, next to them, threshold_summary.csv with one row per animal x tracer: the
threshold 06 applies, plus the background and robust SD it came from.

Two deliberate differences from the code this was ported from:
  * everything is written under the cohort's own results directory, because animal ids repeat across
    cohorts (both series run an17) and the original wrote atlas_space/<animal>.npz for whichever
    cohort its hardcoded PARENT pointed at;
  * only DETECT == 'raw' is carried over. The 'unsharp' variant thresholds the high-pass image that
    the older per-slice pipeline wrote into <animal>/<tracer>/unsharp_image.tif, which this pipeline
    does not produce, so it would be a dependency on the very thing the native arm exists to avoid.

Run:  python 05_atlas_space.py --cohort NS24122 --qc an4     # one animal, prints checks, writes a figure
      python 05_atlas_space.py --cohort NS24122              # all animals -> results/<cohort>/atlas_space
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tifffile

import config

HERE = os.path.dirname(os.path.abspath(__file__))

REGISTERED_ATLAS = 'registered_atlas.tiff'
DEFORMATION_FIELD = 'deformation_field_{axis}.tiff'


def binned_shape(shape, bin_size=None):
    """The atlas grid that 06 draws on: the full-resolution atlas shape, binned by ATLAS_BIN."""
    bin_size = bin_size or config.ATLAS_BIN
    return tuple(int(np.ceil(s / bin_size)) for s in shape)


def robust_background(values):
    """Median and robust SD (1.4826 * MAD) of a set of intensities."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, 1.4826 * mad


def registration_dir(animal):
    return os.path.join(config.DATA_DIR, animal, 'registration_dir')


def tissue_mask(animal, brain):
    """
    Voxels of the registered atlas that actually hold tissue.

    The registered atlas overhangs the sample at its edges, and those voxels are far darker than
    tissue. Left in, each one subtracts roughly a background's worth of intensity from its region,
    and how much overhang there is varies between animals with how snugly the registration fits -
    which is itself group-related in these cohorts. Tissue is defined on the registration image with
    a low threshold (TISSUE_K robust SDs *below* background), so it marks tissue rather than tracer.

    Returns:
    tuple: (tissue mask over the whole volume, fraction of atlas voxels that are tissue)
    """
    reference = tifffile.imread(config.image_path(animal, config.REFERENCE_IMAGE))
    median, sd = robust_background(reference[brain].astype(np.float64))
    tissue = reference > (median - config.TISSUE_K * sd)
    return tissue, float(tissue[brain].mean())


def read_channel(animal, tracer):
    """The raw 20 um tracer image, the one the native arm measures too."""
    folder = config.CHANNELS[tracer]
    return tifffile.imread(config.image_path(animal, f'{folder}.tif'))


def channel_threshold(image, measured, k=None):
    """
    One detection threshold for the whole brain of one animal x channel.

    T = background + k robust SDs, both from the atlas voxels that hold tissue. That population is
    deliberately the same one the intensity arm used, so a threshold in threshold_summary.csv means
    what it meant there.

    Returns:
    tuple: (threshold, background, robust SD).
    """
    k = config.K_MAD if k is None else k
    median, sd = robust_background(image[measured].astype(np.float64))
    return median + k * sd, median, sd


def resample_animal(animal, atlas_shape, bin_size=None):
    """
    Mean raw intensity per (binned) atlas voxel for both channels of one animal.

    Returns:
    dict: tracer names -> float32 mean intensity, 'n' (sample voxels per atlas voxel),
          'bg_<tracer>', plus 'tissue_fraction', 'threshold_<tracer>' and the QC numbers main() prints.
    """
    bin_size = bin_size or config.ATLAS_BIN
    reg_dir = registration_dir(animal)
    labels = tifffile.imread(os.path.join(reg_dir, REGISTERED_ATLAS))
    inside = labels > 0
    tissue, tissue_fraction = tissue_mask(animal, inside)
    del labels

    # every tissue voxel, whether or not the registered atlas covers it
    flat_sample = np.flatnonzero(tissue.ravel())
    shape = binned_shape(atlas_shape, bin_size)

    # atlas coordinate of each of those voxels, one deformation field at a time: each is a float32
    # volume the size of the image, so three at once is a lot of memory for no reason
    index = np.zeros(len(flat_sample), dtype=np.int64)
    keep = np.ones(len(flat_sample), dtype=bool)
    stride = [int(np.prod(shape[i + 1:])) for i in range(3)]
    for axis in range(3):
        path = os.path.join(reg_dir, DEFORMATION_FIELD.format(axis=axis))
        field = tifffile.imread(path).ravel()
        coord = np.rint(field[flat_sample] / config.ATLAS_VOXEL_MM / bin_size).astype(np.int64)
        del field
        keep &= (coord >= 0) & (coord < shape[axis])
        index += np.clip(coord, 0, shape[axis] - 1) * stride[axis]

    index = index[keep]
    n_bins = int(np.prod(shape))
    counts = np.bincount(index, minlength=n_bins)
    result = {'n': counts.reshape(shape).astype(np.int32), 'tissue_fraction': np.float64(tissue_fraction),
              'sample_voxels': np.int64(len(flat_sample)), 'kept_voxels': np.int64(int(keep.sum()))}
    for tracer in config.CHANNELS:
        raw = read_channel(animal, tracer)
        values = raw.ravel()[flat_sample][keep].astype(np.float64)
        bg, _ = robust_background(raw[inside & tissue].astype(np.float64))
        total = np.bincount(index, weights=values, minlength=n_bins)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(counts > 0, total / np.maximum(counts, 1), 0.0)
        result[tracer] = mean.reshape(shape).astype(np.float32)
        result[f'bg_{tracer}'] = np.float32(bg)
        threshold, _, _ = channel_threshold(raw, inside & tissue)
        result[f'threshold_{tracer}'] = np.float64(threshold)
        del raw, values, total, mean
    del tissue
    return result


def save(animal, data):
    """One npz per animal. 06 reads exactly these keys, and the provenance guard reads the last two."""
    os.makedirs(config.ATLAS_SPACE_DIR, exist_ok=True)
    payload = {'n': np.clip(data['n'], 0, 255).astype(np.uint8),
               'cohort': config.COHORT, 'provenance': config.provenance()}
    for tracer in config.CHANNELS:
        payload[tracer] = np.clip(data[tracer], 0, 65535).astype(np.uint16)
        payload[f'bg_{tracer}'] = data[f'bg_{tracer}']
    np.savez_compressed(os.path.join(config.ATLAS_SPACE_DIR, f'{animal}.npz'), **payload)


def write_thresholds(animal, data, rows):
    """The per animal x tracer threshold 06 applies, in the shape it expects."""
    for tracer in config.CHANNELS:
        rows.append({'animal_number': animal, 'animal': animal, 'treatment': config.treatment_of(animal),
                     'tracer': tracer, 'threshold': float(data[f'threshold_{tracer}']),
                     'bg_raw': float(data[f'bg_{tracer}']), 'k_mad': config.K_MAD,
                     'detect_on': 'raw', 'tissue_fraction_of_atlas': float(data['tissue_fraction'])})


def qc(animal, bin_size=None):
    """Resample one animal and check it looks like a brain in the right place."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bin_size = bin_size or config.ATLAS_BIN
    annotation = tifffile.imread(config.ATLAS_ANNOTATION)
    t0 = time.time()
    data = resample_animal(animal, annotation.shape, bin_size)
    print(f'{animal}: resampled in {time.time() - t0:.0f} s; grid {data["n"].shape}')

    atlas_small = annotation[::bin_size, ::bin_size, ::bin_size] > 0
    covered = data['n'] > 0
    print(f'  atlas voxels with at least one sample voxel: {100 * covered[atlas_small].mean():.1f}%')
    print(f'  tissue voxels inside the atlas: {100 * data["tissue_fraction"]:.1f}%')
    if (atlas_small & covered).any():
        print(f'  sample voxels per atlas voxel inside the brain: '
              f'median {np.median(data["n"][atlas_small & covered]):.1f}')
    print(f'  signal-bearing voxels outside the atlas outline: {covered[~atlas_small].sum():,} '
          '(pial surface and cisterns, which the atlas does not cover)')
    for tracer in config.CHANNELS:
        v = data[tracer]
        print(f'  {tracer}: background {data[f"bg_{tracer}"]:.0f}, threshold '
              f'{data[f"threshold_{tracer}"]:.0f}; mean inside brain '
              f'{v[atlas_small & covered].mean():.0f}' if (atlas_small & covered).any() else '')

    planes = (np.linspace(0.15, 0.85, 6) * atlas_small.shape[0]).astype(int)
    fig, axes = plt.subplots(len(config.CHANNELS), len(planes), figsize=(17, 3.2 * len(config.CHANNELS)),
                             squeeze=False)
    for row, tracer in enumerate(config.CHANNELS):
        hi = np.percentile(data[tracer][covered], 99.5) if covered.any() else 1.0
        for col, plane in enumerate(planes):
            ax = axes[row, col]
            ax.imshow(data[tracer][plane], cmap='magma', vmin=0, vmax=hi, interpolation='nearest')
            ax.contour(atlas_small[plane], levels=[0.5], colors='#66CCFF', linewidths=0.6)
            if row == 0:
                ax.set_title(f'{plane * config.ATLAS_SPACING_MM:.1f} mm', fontsize=9)
            if col == 0:
                ax.set_ylabel(tracer, fontsize=10)
            ax.set_axis_off()
    fig.suptitle(f'{animal}: raw intensity resampled into atlas space (blue = atlas brain outline)',
                 x=0.01, ha='left', fontsize=12)
    plt.tight_layout()
    os.makedirs(config.QC_DIR, exist_ok=True)
    out = os.path.join(config.QC_DIR, f'{animal}_atlas_space.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qc', metavar='ANIMAL', help='resample one animal and write a check figure')
    parser.add_argument('--bin', type=int, help=f'atlas voxels per output voxel (default {config.ATLAS_BIN})')
    parser.add_argument('--cohort', help='cohort from config.COHORTS (default: $NATIVE_DEPTH_COHORT)')
    args = parser.parse_args()
    config.select(args.cohort)
    for line in config.validate():
        print(line)
    print(config.describe(), '\n', flush=True)

    if not os.path.isfile(config.ATLAS_ANNOTATION):
        raise SystemExit(f'step 05 needs the atlas annotation and cannot find it: '
                         f'{config.ATLAS_ANNOTATION}\nSet ATLAS_ANNOTATION in config.py to the '
                         f'annotation.tiff of the brainglobe install of {config.ATLAS}.')

    if args.qc:
        qc(args.qc, args.bin)
        return

    annotation_shape = tifffile.imread(config.ATLAS_ANNOTATION).shape
    # every animal whose brainreg output is present; the native masks are not needed here, so this
    # does not require 02 to have run
    animals = [a for a in config.animals()
               if os.path.isdir(registration_dir(a))
               and os.path.isfile(os.path.join(registration_dir(a), REGISTERED_ATLAS))]
    missing = [a for a in config.animals() if a not in animals]
    if missing:
        print(f'note: no registration_dir/{REGISTERED_ATLAS} for {missing}, skipping them\n')
    if not animals:
        raise SystemExit(f'no animal under {config.DATA_DIR} has a '
                         f'registration_dir/{REGISTERED_ATLAS}; run brainreg for this cohort first')

    os.makedirs(config.ATLAS_SPACE_DIR, exist_ok=True)
    rows, start = [], time.time()
    for animal in animals:
        t0 = time.time()
        data = resample_animal(animal, annotation_shape, args.bin)
        save(animal, data)
        write_thresholds(animal, data, rows)
        print(f'{animal}: {time.time() - t0:.0f} s, {data["kept_voxels"]:,} voxels placed, '
              f'coverage {100 * (data["n"] > 0).mean():.1f}% of the grid', flush=True)
    summary = os.path.join(config.ATLAS_SPACE_DIR, 'threshold_summary.csv')
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f'\nwrote {len(animals)} volumes to {config.ATLAS_SPACE_DIR} in '
          f'{(time.time() - start) / 60:.1f} min')
    print(f'thresholds: {summary}')
    print(f'manifest: {config.write_manifest("05_atlas_space", {"animals": len(animals), "bin": args.bin or config.ATLAS_BIN})}')
    print('\nnow run 06_atlas_maps.py for this cohort')


if __name__ == '__main__':
    main()