# Title: Image Transformations
# Author: Phillip Muza
# Date: 09/12/24

import os
import csv
import argparse
from skimage.io import imread, imsave
import numpy as np
import tifffile
import logging
from multiprocessing import Pool, cpu_count
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Acquisition direction -> transform. Both transforms transpose (1, 0, 2), taking a transverse
# stack to a coronal one; they differ only in the flip that follows.
DIRECTION_TO_TRANSFORM = {
    "ventral_to_dorsal": 1,   # transpose + rot90(2): reverses axes 0 and 1
    "dorsal_to_ventral": 2,   # transpose + [::-1]:   reverses axis 0 only
}

# TIFF switches to BigTIFF beyond 4 GB; leave headroom for headers
BIGTIFF_THRESHOLD_BYTES = 3.5 * 1024 ** 3


def read_stack(image_path):
    """Read a 3D stack, tolerating TIFFs whose pages are not grouped into one series.

    Some stacks are written with every page as its own series. skimage.io.imread (and
    tifffile.asarray) then return only the first plane, silently yielding a 2D image where a
    stack was expected. Where that happens the pages are assembled explicitly.
    """
    with tifffile.TiffFile(image_path) as tif:
        n_pages = len(tif.pages)

        arr = tif.asarray() if len(tif.series) == 1 else None

        if arr is None or arr.ndim < 3:
            if n_pages < 2:
                raise ValueError(f"{image_path} has {n_pages} page(s); not a stack")
            logging.info(f"  {os.path.basename(image_path)}: pages not grouped into a series "
                         f"({len(tif.series)} series), assembling {n_pages} pages directly")
            first = tif.pages[0].asarray()
            arr = np.empty((n_pages,) + first.shape, dtype=first.dtype)
            arr[0] = first
            for i in range(1, n_pages):
                arr[i] = tif.pages[i].asarray()

    if arr.ndim != 3:
        raise ValueError(f"{image_path} loaded as {arr.ndim}D with shape {arr.shape}; expected 3D")
    return arr


def transformed_view(img, transform_type):
    """Return the transformed stack as a numpy view (no copy).

    transpose(1, 0, 2) reorders a transverse stack (z, y, x) into a coronal one, moving the
    in-plane anterior-posterior axis to the front and the acquisition axis to position 1.
    Note this also reorders the voxel sizes: (20, 6.55, 6.55) becomes (6.55, 20, 6.55).
    """
    transposed = img.transpose(1, 0, 2)
    if transform_type == 1:
        return np.rot90(transposed, 2)      # == transposed[::-1, ::-1, :]
    elif transform_type == 2:
        return transposed[::-1, :, :]
    raise ValueError(f"Unknown transform type: {transform_type}")


def save_stack_streaming(view, output_path):
    """Write a transformed view one plane at a time.

    The views produced by transformed_view are non-contiguous, so handing one to imsave
    materialises a second full copy of the stack. For multi-GB light-sheet data that doubles
    peak memory, so planes are written individually instead.
    """
    n_bytes = int(np.prod(view.shape)) * view.dtype.itemsize
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with tifffile.TiffWriter(output_path, bigtiff=n_bytes > BIGTIFF_THRESHOLD_BYTES) as writer:
        for i in range(view.shape[0]):
            writer.write(np.ascontiguousarray(view[i]), contiguous=True)

class ImageTransformer:
    def __init__(self, input_dir=None, output_dir=None, debug=False, transform_type=1, batch_size=4, mirror_structure=False, recursive_debug=False):
        self.input_dir = input_dir
        self.output_dir = output_dir or os.path.join(input_dir, "transformations") if input_dir else "debug_transformations"
        self.debug = debug
        self.transform_type = transform_type
        self.batch_size = batch_size
        self.mirror_structure = mirror_structure
        self.recursive_debug = recursive_debug
        logging.info(f"ImageTransformer initialized with input_dir={input_dir}, output_dir={output_dir}, debug={debug}, transform_type={transform_type}, mirror_structure={mirror_structure}, recursive_debug={recursive_debug}")

    def transform_image_1(self, image_path): # With transpose and reverse - Images in the transverse plane going from ventral to dorsal
        logging.info(f"Starting transformation 1 for {image_path} with transpose and reverse")
        # Save the current working directory
        original_dir = os.getcwd()
        
        try:
            img = imread(image_path)
            image_name = os.path.basename(image_path)

            # Apply the transformations
            img_rotated = transformed_view(img, 1)

            if self.mirror_structure and self.input_dir and self.output_dir:
                # Create mirrored directory structure
                rel_path = os.path.relpath(os.path.dirname(image_path), self.input_dir)
                output_dir = os.path.join(self.output_dir, rel_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, image_name)
                imsave(output_path, img_rotated)
            else:
                # Original behavior - save in the same directory
                image_dir = os.path.dirname(image_path)
                # Change to the image directory
                os.chdir(image_dir)
                imsave(f"{image_name}", img_rotated)
                
            logging.info(f"Transformation 1 complete and saved for {image_path}")
        except Exception as e:
            logging.error(f"Error transforming {image_path}: {e}")
        finally:
            # Restore the original working directory
            os.chdir(original_dir)

    def transform_image_2(self, image_path): # With transpose and reverse - Images in the transverse plane going from dorsal to ventral
        logging.info(f"Starting transformation 2 for {image_path} with transpose and reverse")
        # Save the current working directory
        original_dir = os.getcwd()
        
        try:
            img = imread(image_path)
            image_name = os.path.basename(image_path)

            # Apply the transformations
            img_reversed = transformed_view(img, 2)

            if self.mirror_structure and self.input_dir and self.output_dir:
                # Create mirrored directory structure
                rel_path = os.path.relpath(os.path.dirname(image_path), self.input_dir)
                output_dir = os.path.join(self.output_dir, rel_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, image_name)
                imsave(output_path, img_reversed)
            else:
                # Original behavior - save in the same directory
                image_dir = os.path.dirname(image_path)
                # Change to the image directory
                os.chdir(image_dir)
                imsave(f"{image_name}", img_reversed)
                
            logging.info(f"Transformation 2 complete and saved for {image_path}")
        except Exception as e:
            logging.error(f"Error transforming {image_path}: {e}")
        finally:
            # Restore the original working directory
            os.chdir(original_dir)

    def debug_transform(self, image_path, output_dir):
        """Process an image with both transformation methods and save to debug directory"""
        logging.info(f"Starting debug transformation for {image_path}")
        
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Load the image
            img = imread(image_path)
            
            # Get image identifiers for naming
            path_parts = os.path.normpath(image_path).split(os.sep)
            basename = os.path.basename(image_path)
            
            # Create unique identifier using [-1,-2] parts of path plus basename
            if len(path_parts) >= 2:
                identifier = f"{path_parts[-3]}_{path_parts[-2]}"
            else:
                identifier = basename
            
            # Get middle slice for debugging
            middle_slice_idx = img.shape[0] // 2
            middle_slice = img[middle_slice_idx]
                
            # Copy original middle slice
            original_output = os.path.join(output_dir, f"{identifier}_original.png")
            imsave(original_output, middle_slice)
            
            # Transform 1 - ventral to dorsal
            img_rotated = transformed_view(img, 1)
            middle_slice_t1 = img_rotated[middle_slice_idx]
            transform1_output = os.path.join(output_dir, f"{identifier}_transform1.png")
            imsave(transform1_output, middle_slice_t1)

            # Transform 2 - dorsal to ventral
            img_reversed = transformed_view(img, 2)
            middle_slice_t2 = img_reversed[middle_slice_idx]
            transform2_output = os.path.join(output_dir, f"{identifier}_transform2.png")
            imsave(transform2_output, middle_slice_t2)
            
            logging.info(f"Debug transformations complete for {image_path}")
            return True
        except Exception as e:
            logging.error(f"Error in debug transformation for {image_path}: {e}")
            return False

    def process_files_in_batches(self, file_list, output_dir, batch_size=None):
        """Process files in batches using multiprocessing"""
        batch_size = batch_size or self.batch_size
        num_processes = min(batch_size, cpu_count())
        logging.info(f"Processing {len(file_list)} files in batches of {batch_size} using {num_processes} processes")
        
        # If in debug mode, only process one file
        if self.debug and file_list:
            file_list = [file_list[0]]
            logging.info(f"Debug mode: Processing only one representative file: {file_list[0]}")
        
        for i in range(0, len(file_list), batch_size):
            batch = file_list[i:i + batch_size]
            
            if self.debug:
                process_args = [(file, output_dir) for file in batch]
                with Pool(processes=num_processes) as pool:
                    pool.starmap(self.debug_transform, process_args)
            else:
                # Choose which transform method to use based on transform_type
                transform_func = self.transform_image_1 if self.transform_type == 1 else self.transform_image_2
                with Pool(processes=num_processes) as pool:
                    pool.map(transform_func, batch)
            
            # Small delay between batches
            time.sleep(1)

    def get_basename(self, filepath):
        """Extract the base name without extension and any suffixes like '_downsampled'"""
        filename = os.path.basename(filepath)
        # Remove extension
        basename = os.path.splitext(filename)[0]
        # Remove common suffixes
        basename = re.sub(r'_downsampled$|_fitc$|_txr$', '', basename)
        return basename

    def run_recursive_debug(self):
        """Run debug mode recursively, selecting one representative image per basename"""
        if not self.input_dir:
            logging.error("No input directory specified for recursive debug")
            return
            
        logging.info(f"Running recursive debug on directory: {self.input_dir}")
        
        # Dictionary to track basenames and their corresponding files
        basename_files = {}
        
        # Walk through directory structure
        for root, dirs, files in os.walk(self.input_dir):
            tif_files = [os.path.join(root, f) for f in files if f.endswith((".tif", ".tiff"))]
            
            if not tif_files:
                continue
                
            # Group files by basename and directory
            for file_path in tif_files:
                # Use directory as part of the key to ensure we process files in each directory
                dir_key = os.path.dirname(file_path)
                basename = self.get_basename(file_path)
                key = f"{dir_key}|{basename}"
                
                if key not in basename_files:
                    basename_files[key] = []
                basename_files[key].append(file_path)
        
        # Select representative files (prefer fitc_downsampled.tiff)
        selected_files = []
        for key, files in basename_files.items():
            # Try to find fitc_downsampled.tiff first
            fitc_downsampled = [f for f in files if "fitc_downsampled" in f.lower()]
            if fitc_downsampled:
                selected_files.append(fitc_downsampled[0])
            else:
                # Otherwise take the first file
                selected_files.append(files[0])
        
        logging.info(f"Selected {len(selected_files)} representative files for debug processing")
        
        # Process each selected file
        for file_path in selected_files:
            # Create debug folder in the same directory as the file
            file_dir = os.path.dirname(file_path)
            debug_dir = os.path.join(file_dir, "debug")
            
            # Debug transform the file
            self.debug_transform(file_path, debug_dir)
            
        logging.info("Recursive debug processing complete")

    def run_from_csv(self, csv_path, base_dir=None, output_dir=None, images=None, in_place=False):
        """Transform each folder using the direction scored in orientations.csv.

        Direction varies between animals, so a single --transform_type cannot describe a whole
        dataset. The CSV written by check_orientation.py records one direction per folder, and
        this maps each to its transform.

        Folder paths in the CSV are relative to the directory that check_orientation.py was
        pointed at, which is the parent of the orientation_check folder holding the CSV.
        """
        csv_path = os.path.abspath(csv_path)
        base_dir = base_dir or os.path.dirname(os.path.dirname(csv_path))
        output_dir = output_dir or os.path.join(base_dir, "transformed")

        with open(csv_path, newline='') as fh:
            rows = list(csv.DictReader(fh))

        scored = [r for r in rows if (r.get('direction') or '').strip()]
        unscored = [r for r in rows if not (r.get('direction') or '').strip()]
        for row in unscored:
            logging.warning(f"No direction recorded for {row['folder']}, skipping")

        invalid = [r for r in scored if r['direction'].strip().lower() not in DIRECTION_TO_TRANSFORM]
        if invalid:
            for row in invalid:
                logging.error(f"Unrecognised direction {row['direction']!r} for {row['folder']}; "
                              f"expected one of {', '.join(DIRECTION_TO_TRANSFORM)}")
            return

        if in_place:
            logging.warning("Running in place: original images will be OVERWRITTEN with their "
                            "transformed versions")
        else:
            logging.info(f"Writing transformed images to {output_dir} (originals untouched)")

        for row in scored:
            folder = row['folder'].strip()
            direction = row['direction'].strip().lower()
            transform_type = DIRECTION_TO_TRANSFORM[direction]
            folder_path = os.path.join(base_dir, folder)

            if not os.path.isdir(folder_path):
                logging.error(f"Folder not found, skipping: {folder_path}")
                continue

            # Every channel in a folder shares the folder's acquisition direction
            entries = sorted(f for f in os.listdir(folder_path)
                             if f.lower().endswith(('.tif', '.tiff'))
                             and os.path.isfile(os.path.join(folder_path, f)))
            if images:
                wanted = {i.lower() for i in images}
                entries = [f for f in entries if f.lower() in wanted]

            if not entries:
                logging.warning(f"No matching images in {folder_path}, skipping")
                continue

            logging.info(f"{folder}: {direction} -> transform {transform_type} "
                         f"({len(entries)} image(s): {', '.join(entries)})")

            for image_name in entries:
                image_path = os.path.join(folder_path, image_name)
                destination = (image_path if in_place
                               else os.path.join(output_dir, folder, image_name))

                if not in_place and os.path.exists(destination):
                    logging.info(f"  {image_name}: already transformed, skipping")
                    continue

                try:
                    start = time.time()
                    img = read_stack(image_path)
                    view = transformed_view(img, transform_type)
                    logging.info(f"  {image_name}: {img.shape} -> {view.shape}")

                    # Write to a temporary name and rename, so an interrupted write cannot leave
                    # a truncated file that the skip-if-exists check would treat as complete
                    partial = destination + ".partial"
                    try:
                        save_stack_streaming(view, partial)
                        os.replace(partial, destination)
                    finally:
                        if os.path.exists(partial):
                            os.remove(partial)

                    del img, view
                    logging.info(f"  {image_name}: written to {destination} "
                                 f"in {time.time() - start:.1f}s")
                except Exception as e:
                    logging.error(f"  {image_name}: failed ({e})")

        logging.info("CSV-driven transformation complete")

    def run(self):
        """Main method to run the image transformation process"""
        if not self.input_dir:
            logging.error("No input directory specified")
            return
        
        # If recursive debug mode is enabled, use that instead of normal processing
        if self.recursive_debug:
            self.run_recursive_debug()
            return
            
        logging.info(f"Processing directory: {self.input_dir}")
        tif_files = []
        
        # Collect all tif files first
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith((".tif", ".tiff")):
                    tif_files.append(os.path.join(root, file))
        
        logging.info(f"Found {len(tif_files)} .tif files to process")
        
        # Process files in batches
        self.process_files_in_batches(tif_files, self.output_dir)
        logging.info("Processing complete")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Transform images with various methods')
    parser.add_argument('--input_dir', '-i', type=str, help='Input directory containing images to transform')
    parser.add_argument('--orientations_csv', '-c', type=str,
                        help='orientations.csv from check_orientation.py; applies the transform '
                             'matching each folder\'s scored direction, overriding --transform_type')
    parser.add_argument('--base_dir', type=str,
                        help='Root the CSV folder column is relative to '
                             '(default: the parent of the folder containing the CSV)')
    parser.add_argument('--images', type=str, nargs='+',
                        help='Restrict CSV mode to these filenames (default: every .tif in the folder)')
    parser.add_argument('--in_place', action='store_true',
                        help='CSV mode: overwrite the originals instead of writing to --output_dir')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory for transformed images (default: input_dir/transformations)')
    parser.add_argument('--debug', '-d', action='store_true', help='Run in debug mode to save all transformation types')
    parser.add_argument('--transform_type', '-t', type=int, choices=[1, 2], default=1, 
                        help='Transformation type to use when not in debug mode: 1=ventral to dorsal, 2=dorsal to ventral')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size for processing (default: 4)')
    parser.add_argument('--mirror_structure', '-m', action='store_true', help='Mirror the input directory structure in the output directory')
    parser.add_argument('--recursive_debug', '-r', action='store_true', help='Run debug mode recursively, selecting one representative image per basename')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if not args.orientations_csv and not args.input_dir:
        raise SystemExit("Provide --orientations_csv or --input_dir")

    transformer = ImageTransformer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        debug=args.debug,
        transform_type=args.transform_type,
        batch_size=args.batch_size,
        mirror_structure=args.mirror_structure,
        recursive_debug=args.recursive_debug
    )

    if args.orientations_csv:
        transformer.run_from_csv(args.orientations_csv, base_dir=args.base_dir,
                                 output_dir=args.output_dir, images=args.images,
                                 in_place=args.in_place)
    else:
        transformer.run()
