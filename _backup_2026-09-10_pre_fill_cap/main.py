# Title: Main Analysis Script to run Downsampling, Brain Registration and per-channel Tracer Analysis
# Author: Phillip Muza
# Date: 23.08.24

import os
import shutil
from tracer_segmentation import TracerSignalAnalyser
from brain_registration import BrainRegistration
from signal_lookup import ImageAnalyser
from registration_regional import HemisphereAnalysis
from downsample_images import downsample_file, read_stack
from orientation import (check_orientation_status, apply_orientation_transform,
                         write_transform_check_grid, transformed_voxel_size,
                         TRANSFORMED_DIRNAME, ORIENTATION_DIRNAME)
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse

DOWNSAMPLED_DIRNAME = "downsampled"
REGISTRATION_DIRNAME = "registration_dir"


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_signal_images(files, signal_img_names):
    """Return every requested signal image present in a directory, matched case-insensitively.

    Returns a list of the actual on-disk filenames, in the order they were requested.
    """
    lookup = {f.lower(): f for f in files}
    return [lookup[name.lower()] for name in signal_img_names if name.lower() in lookup]


def downsample_channels(animal_dir, signal_imgs, source_voxel_size, target_voxel_size,
                        raw_source_dir=None):
    """Downsample each raw signal image into animal_dir/downsampled/, keeping its original name.

    raw_source_dir overrides where the *input* to downsampling is read from (e.g. an animal's
    transformed/ folder, once orientation correction has run); output always lands under
    animal_dir regardless.

    Returns {original_filename: path_to_downsampled_image}. Images already present are reused.
    """
    logger = logging.getLogger(__name__)
    raw_source_dir = raw_source_dir or animal_dir
    downsampled_dir = os.path.join(animal_dir, DOWNSAMPLED_DIRNAME)
    os.makedirs(downsampled_dir, exist_ok=True)

    downsampled = {}
    for img_name in signal_imgs:
        target_path = os.path.join(downsampled_dir, img_name)
        if os.path.exists(target_path):
            logger.info(f"Downsampled image already exists, reusing: {target_path}")
        else:
            start_time = time.time()
            downsample_file(os.path.join(raw_source_dir, img_name), target_path,
                            source_voxel_size, target_voxel_size)
            logger.info(f"Downsampled {img_name} in {time.time() - start_time:.2f} seconds")
        downsampled[img_name] = target_path

    return downsampled


def prepare_registration_image(animal_dir, downsampled, registration_img, registration_source,
                               source_voxel_size, target_voxel_size, raw_source_dir=None):
    """Resolve the single registration image shared by every signal channel.

    Resolution order:
      1. downsampled/<registration_img> already exists  -> use it
      2. <raw_source_dir>/<registration_img> exists (raw) -> downsample it
      3. registration_source names a signal channel     -> copy its downsampled version
    """
    logger = logging.getLogger(__name__)
    raw_source_dir = raw_source_dir or animal_dir
    downsampled_dir = os.path.join(animal_dir, DOWNSAMPLED_DIRNAME)
    registration_path = os.path.join(downsampled_dir, registration_img)

    if os.path.exists(registration_path):
        logger.info(f"Using existing registration image: {registration_path}")
        return registration_path

    raw_registration = os.path.join(raw_source_dir, registration_img)
    if os.path.exists(raw_registration):
        logger.info(f"Downsampling supplied registration image: {raw_registration}")
        downsample_file(raw_registration, registration_path, source_voxel_size, target_voxel_size)
        return registration_path

    # Fall back to copying one of the downsampled signal channels
    source_match = next((name for name in downsampled if name.lower() == registration_source.lower()), None)
    if source_match is None:
        raise FileNotFoundError(
            f"Cannot build a registration image in {animal_dir}: no {registration_img} was found and "
            f"the registration source {registration_source} is not among the signal images "
            f"({', '.join(downsampled) or 'none'})."
        )

    logger.info(f"Copying {downsampled[source_match]} -> {registration_path}")
    shutil.copy2(downsampled[source_match], registration_path)
    return registration_path


def run_registration(animal_dir, registration_path, voxel_size, orientation):
    """Run brainreg once per animal into animal_dir/registration_dir/, shared by all channels."""
    logger = logging.getLogger(__name__)
    registration_dir = os.path.join(animal_dir, REGISTRATION_DIRNAME)

    if os.path.exists(os.path.join(registration_dir, "registered_atlas.tiff")):
        logger.info(f"Registration already complete, skipping: {registration_dir}")
        return registration_dir

    logger.info("Starting Brain Registration")
    start_time = time.time()
    registrator = BrainRegistration(input_file=registration_path,
                                    output_dir=registration_dir,
                                    voxel_size=voxel_size,
                                    orientation=orientation,
                                    preprocess=True)
    if not registrator.run_registration():
        raise RuntimeError(f"Brain registration failed for {animal_dir}")
    logger.info(f"Brain Registration completed in {time.time() - start_time:.2f} seconds")
    return registration_dir


def process_channel(animal_dir, channel_name, signal_path, registration_dir,
                    path_to_structures_file):
    """Segment and quantify one signal channel against the shared registration.

    Hemisphere analysis always runs: the left/right split is cheap next to segmentation and
    registration, and the extra columns are easy to drop downstream but expensive to go back
    for.
    """
    logger = logging.getLogger(__name__)

    channel_dir = os.path.join(animal_dir, os.path.splitext(channel_name)[0])
    os.makedirs(channel_dir, exist_ok=True)

    if os.path.exists(os.path.join(channel_dir, "decoded_region_counts.csv")):
        # Whole-brain results are already here, but the hemispheres may not be (channels
        # processed before this became the default), so fall through to that step.
        logger.info(f"Whole-brain results already present, skipping to hemispheres: {channel_dir}")
        process_hemispheres(channel_dir, registration_dir, path_to_structures_file)
        return

    logger.info(f"Processing channel {channel_name} in {channel_dir}")

    registered_atlas = os.path.join(registration_dir, "registered_atlas.tiff")
    volumes_csv = os.path.join(registration_dir, "volumes.csv")

    # Tracer segmentation writes brain_mask.tif / unsharp_image.tif / thresholded_image.tif
    # into the working directory, so run it from inside the channel directory.
    logger.info("Starting Tracer Segmentation")
    start_time = time.time()
    os.chdir(channel_dir)
    image_stack = read_stack(signal_path)
    tracer_analyser = TracerSignalAnalyser(image_stack)
    tracer_analyser.process()
    tracer_analyser.save_results()
    logger.info(f"Tracer Segmentation completed in {time.time() - start_time:.2f} seconds")

    logger.info("Starting Signal Lookup")
    start_time = time.time()
    analyser = ImageAnalyser(channel_dir)
    analyser.run_analysis(
        "thresholded_image.tif",
        registered_atlas,
        path_to_structures_file,
        volumes_csv
    )
    logger.info(f"Signal Lookup completed in {time.time() - start_time:.2f} seconds")

    process_hemispheres(channel_dir, registration_dir, path_to_structures_file)


def process_hemispheres(channel_dir, registration_dir, path_to_structures_file):
    """Split one channel's results into left and right hemispheres."""
    logger = logging.getLogger(__name__)

    if all(os.path.exists(os.path.join(channel_dir, f"{h}_hemisphere", "decoded_region_counts.csv"))
           for h in ("left", "right")):
        logger.info(f"Hemisphere results already present, skipping: {channel_dir}")
        return

    logger.info(f"Starting Hemisphere Analysis for {channel_dir}")
    start_time = time.time()

    required_files = [
        os.path.join(registration_dir, "registered_atlas.tiff"),
        os.path.join(registration_dir, "registered_hemispheres.tiff"),
        os.path.join(registration_dir, "volumes.csv"),
        os.path.join(channel_dir, "thresholded_image.tif"),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error("Cannot perform hemisphere analysis. Missing required files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        return

    try:
        hemisphere_analyser = HemisphereAnalysis(channel_dir, registration_dir=registration_dir)
        hemisphere_analyser.process_all_hemispheres()
        logger.info(f"Hemisphere Analysis completed in {time.time() - start_time:.2f} seconds")

        for hemisphere in tqdm(["left", "right"], desc="Processing Hemispheres"):
            hemisphere_path = os.path.join(channel_dir, f"{hemisphere}_hemisphere")

            logger.info(f"Starting Signal Lookup for {hemisphere} hemisphere in {hemisphere_path}")
            start_time = time.time()

            analyser = ImageAnalyser(hemisphere_path)
            analyser.run_analysis(
                "signal_mask.tiff",
                "registration_mask.tiff",
                path_to_structures_file,
                "volumes.csv"
            )

            logger.info(f"Signal Lookup for {hemisphere} hemisphere completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during hemisphere analysis: {str(e)}")
        logger.error("Skipping hemisphere analysis for this channel.")


def discover_animal_dirs(parent_dir, signal_img_names):
    """Walk parent_dir once and return [(animal_dir, signal_imgs_found), ...].

    Never descends into directories the pipeline itself creates, including the
    orientation-check outputs.
    """
    animal_dirs = []
    for root, dirs, files in os.walk(parent_dir):
        dirs[:] = [d for d in dirs if d not in
                   (DOWNSAMPLED_DIRNAME, REGISTRATION_DIRNAME, TRANSFORMED_DIRNAME, ORIENTATION_DIRNAME)
                   and not any(d.lower() == os.path.splitext(s)[0].lower() for s in signal_img_names)]

        signal_imgs = find_signal_images(files, signal_img_names)
        if signal_imgs:
            animal_dirs.append((root, signal_imgs))

    return animal_dirs


def process_directory(animal_dir, signal_imgs, registration_img, registration_source,
                      source_voxel_size, target_voxel_size, orientation,
                      path_to_structures_file, raw_source_dir=None):
    """Run the full pipeline for one animal directory holding one or more raw signal images.

    raw_source_dir overrides where raw signal images are read from for downsampling (e.g. an
    animal's transformed/ folder after orientation correction); everything else about the
    animal's output layout is unaffected.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting analysis in directory: {animal_dir}")

    # Step 0: downsample every channel to atlas resolution
    downsampled = downsample_channels(animal_dir, signal_imgs, source_voxel_size, target_voxel_size,
                                      raw_source_dir=raw_source_dir)

    # Step 1: resolve the shared registration image
    registration_path = prepare_registration_image(
        animal_dir, downsampled, registration_img, registration_source,
        source_voxel_size, target_voxel_size, raw_source_dir=raw_source_dir
    )

    # Step 2: register once for all channels
    voxel_size = " ".join(str(v) for v in target_voxel_size)
    registration_dir = run_registration(animal_dir, registration_path, voxel_size, orientation)

    # Step 3: analyse each channel against that single registration
    for channel_name in tqdm(signal_imgs, desc="Processing Channels"):
        try:
            process_channel(animal_dir, channel_name, downsampled[channel_name], registration_dir,
                            path_to_structures_file)
        except Exception as e:
            logger.error(f"Error processing channel {channel_name} in {animal_dir}: {e}")

    logger.info("Analysis complete. Results are saved in the respective channel directories.")


def resolve_orientation(parent_dir, animal_dirs, registration_source):
    """Gate the whole run on every animal having a scored acquisition direction.

    Regenerates contact sheets for anything unscored and returns None if the run should stop
    there for a QC pass. Otherwise transforms every animal not already transformed, writes one
    cross-animal verification grid, and returns {animal_dir: transformed_dir_or_None} (None for
    animals whose images were already coronal-correct going in, before this run started).
    """
    logger = logging.getLogger(__name__)
    ready, unscored, csv_path = check_orientation_status(parent_dir, animal_dirs, registration_source)

    if unscored:
        logger.warning(
            f"\n{len(unscored)} of {len(animal_dirs)} animal(s) still need orientation scoring "
            f"before this run can proceed:\n"
            f"  1. Open the contact sheets in {os.path.join(parent_dir, ORIENTATION_DIRNAME)}\n"
            f"  2. Fill in the 'direction' column in {csv_path} "
            f"with dorsal_to_ventral or ventral_to_dorsal\n"
            f"  3. Re-run this exact command\n"
            f"Nothing has been processed this run so the whole batch gets one QC pass together."
        )
        return None

    transformed_by_animal = {}
    for animal_dir, signal_imgs in animal_dirs:
        direction = ready.get(animal_dir)
        if direction is None:
            continue  # already transformed in a previous run
        transformed_by_animal[animal_dir] = apply_orientation_transform(animal_dir, signal_imgs, direction)

    if transformed_by_animal:
        write_transform_check_grid(parent_dir, transformed_by_animal, registration_source)
        logger.info(
            f"Transformed {len(transformed_by_animal)} animal(s). Check "
            f"{os.path.join(parent_dir, ORIENTATION_DIRNAME, 'transform_check.png')} before trusting "
            f"the results -- a wrong direction call shows up as a row running backwards against the others."
        )

    return {animal_dir: os.path.join(animal_dir, TRANSFORMED_DIRNAME) for animal_dir, _ in animal_dirs}


def main(parent_dir, signal_img_names, registration_img, registration_source,
         source_voxel_size, target_voxel_size, orientation, source_orientation='transverse'):
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the path to structures file relative to the script location
    path_to_structures_file = os.path.join(script_dir, "structures.csv")

    parent_dir = os.path.abspath(parent_dir)
    animal_dirs = discover_animal_dirs(parent_dir, signal_img_names)
    if not animal_dirs:
        logger.error(f"No animal directories found under {parent_dir} "
                     f"(looked for {', '.join(signal_img_names)})")
        return

    logger.info(f"Found {len(animal_dirs)} animal director{'y' if len(animal_dirs) == 1 else 'ies'}")

    # Raw LSM stacks are transverse and need normalising to coronal before anything downstream
    # can use them; --source_orientation coronal skips this for data that already is.
    raw_source_by_animal = {}
    downsample_source_voxel_size = source_voxel_size
    if source_orientation == 'transverse':
        raw_source_by_animal = resolve_orientation(parent_dir, animal_dirs, registration_source)
        if raw_source_by_animal is None:
            return  # unscored animals remain; nothing processed this run
        # transformed_view's transpose(1, 0, 2) swaps the z and y axes -- downsampling must use
        # the reordered voxel size for images read from transformed/, or the result comes out
        # anisotropic (this bit us on the first real test: stretched/squashed instead of isotropic)
        downsample_source_voxel_size = transformed_voxel_size(source_voxel_size)

    original_cwd = os.getcwd()
    for animal_dir, signal_imgs in animal_dirs:
        logger.info(f"Found {len(signal_imgs)} signal image(s) in {animal_dir}: {', '.join(signal_imgs)}")
        try:
            process_directory(animal_dir, signal_imgs, registration_img, registration_source,
                              downsample_source_voxel_size, target_voxel_size, orientation,
                              path_to_structures_file,
                              raw_source_dir=raw_source_by_animal.get(animal_dir))
        except Exception as e:
            logger.error(f"Error processing directory {animal_dir}: {e}")
        finally:
            # Channel processing chdirs into per-channel folders; restore before the next animal
            os.chdir(original_cwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process brain images, quantifying whole brain and left/right hemispheres')
    parser.add_argument('--parent_dir', type=str, required=True, help='Path to the parent directory containing the images')
    parser.add_argument('--signal_imgs', type=str, nargs='+', default=['FITC.tif', 'TxR.tif'],
                        help='Names of signal image files to look for (default: FITC.tif TxR.tif)')
    parser.add_argument('--registration_img', type=str, default='registration.tif',
                        help='Filename of the shared registration image inside the downsampled folder')
    parser.add_argument('--registration_source', type=str, default='FITC.tif',
                        help='Signal image copied to build the registration image when none is supplied')
    parser.add_argument('--source_voxel_size', type=float, nargs=3, default=[20, 6.55, 6.55],
                        metavar=('Z', 'X', 'Y'),
                        help='Voxel size of the raw images in microns (default: 20 6.55 6.55)')
    parser.add_argument('--target_voxel_size', type=float, nargs=3, default=[20, 20, 20],
                        metavar=('Z', 'X', 'Y'),
                        help='Voxel size to downsample to, matching the atlas (default: 20 20 20)')
    parser.add_argument('--orientation', type=str, default='asr',
                        help='Orientation of the images passed to brainreg (default: asr)')
    parser.add_argument('--source_orientation', type=str, choices=['transverse', 'coronal'], default='transverse',
                        help="'transverse' (default) treats raw images as acquired transverse and "
                             "runs orientation scoring + the transverse->coronal transform as step 0, "
                             "gating the run until every animal is scored. 'coronal' skips this "
                             "entirely for data that is already correctly oriented.")

    args = parser.parse_args()

    main(args.parent_dir, args.signal_imgs, args.registration_img, args.registration_source,
         np.array(args.source_voxel_size), np.array(args.target_voxel_size), args.orientation,
         args.source_orientation)
