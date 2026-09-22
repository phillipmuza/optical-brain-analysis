#Title: Downsampling images from MesoSPIM
#Author: Phillip Muza
#Date: 23.07.2024

import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize
from tqdm import tqdm
import os
import logging
import tifffile
from multiprocessing import Pool, cpu_count
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_stack(image_path):
    """Read a 3D stack, tolerating TIFFs whose pages are not grouped into one series.

    Some stacks are written with every page as its own series. skimage.io.imread then returns
    only the first plane, silently yielding a 2D image where a stack was expected. Where that
    happens the pages are assembled explicitly.
    """
    with tifffile.TiffFile(image_path) as tif:
        n_pages = len(tif.pages)

        arr = tif.asarray() if len(tif.series) == 1 else None

        if arr is None or arr.ndim < 3:
            if n_pages < 2:
                raise ValueError(f"{image_path} has {n_pages} page(s); not a stack")
            logging.info(f"{image_path}: pages not grouped into a series ({len(tif.series)} "
                         f"series), assembling {n_pages} pages directly")
            first = tif.pages[0].asarray()
            arr = np.empty((n_pages,) + first.shape, dtype=first.dtype)
            arr[0] = first
            for i in range(1, n_pages):
                arr[i] = tif.pages[i].asarray()

    if arr.ndim != 3:
        raise ValueError(f"{image_path} loaded as {arr.ndim}D with shape {arr.shape}; expected 3D")
    return arr

def downsample_file(input_file, output_file, original_resolution, target_resolution):
    """Downsample a single 3D TIFF and write it to an explicit output path."""
    original_resolution = np.asarray(original_resolution, dtype=float)
    target_resolution = np.asarray(target_resolution, dtype=float)

    logging.info(f"Starting downsampling for {input_file}")
    logging.info(f"Original resolution: {original_resolution}")
    logging.info(f"Target resolution: {target_resolution}")
    data = read_stack(input_file)
    logging.info(f"Original shape: {data.shape}")

    # Calculate the new shape with a safety check
    new_shape = tuple(np.maximum(
        np.round((np.array(data.shape) * original_resolution) / target_resolution).astype(int),
        [1, 1, 1]  # Ensure minimum size of 1 in each dimension
    ))
    logging.info(f"New shape calculated: {new_shape}")

    # Add a safety check for scaling factors
    scale_factors = np.array(new_shape) / np.array(data.shape)
    if np.any(scale_factors == 0) or np.any(np.isinf(scale_factors)):
        raise ValueError(f"Invalid scaling factors calculated: {scale_factors}. Check your resolution values.")

    # Create an empty array to store the downsampled image
    downsampled_data = np.zeros(new_shape, dtype=data.dtype)

    # Define the chunk size (adjust these values based on your memory constraints)
    chunk_size = (50, data.shape[1], data.shape[2])  # Process 50 z-slices at a time

    # Calculate the total number of chunks
    total_chunks = int(np.ceil(data.shape[0] / chunk_size[0]))

    # Process the image in 3D chunks with tqdm progress bar
    for z_start in tqdm(range(0, data.shape[0], chunk_size[0]), total=total_chunks, desc="Processing chunks"):
        z_end = min(z_start + chunk_size[0], data.shape[0])
        
        # Extract the chunk
        chunk = data[z_start:z_end]
        
        # Calculate the corresponding slice range in the output
        output_z_start = int(np.round(z_start * scale_factors[0]))
        output_z_end = min(int(np.round(z_end * scale_factors[0])), new_shape[0])
        
        # Resize directly to the exact size needed
        chunk_new_shape = (output_z_end - output_z_start, new_shape[1], new_shape[2])
        
        # Skip processing if the chunk would be too small
        if chunk_new_shape[0] == 0:
            continue
            
        # Resize the 3D chunk
        try:
            downsampled_chunk = resize(chunk, 
                                     chunk_new_shape, 
                                     anti_aliasing=True, 
                                     preserve_range=True, 
                                     order=1)  # Changed to order=1 for better stability
        except Exception as e:
            logging.error(f"Error processing chunk {z_start}-{z_end}. Shape: {chunk.shape} -> {chunk_new_shape}")
            raise e
        
        # Ensure the downsampled chunk fits into the output array
        downsampled_chunk = downsampled_chunk[:output_z_end - output_z_start]
        
        # Place the downsampled chunk into the output array
        downsampled_data[output_z_start:output_z_end] = downsampled_chunk

    # Ensure the output data type matches the input
    downsampled_data = downsampled_data.astype(data.dtype)

    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imsave(output_file, downsampled_data)
    logging.info(f"Downsampled image saved to {output_file}")
    return output_file


def downsample_image(input_file, original_resolution, target_resolution, output_dir):
    """Downsample into output_dir, appending a _downsampled suffix to the filename."""
    name, ext = os.path.splitext(os.path.basename(input_file))
    output_file = os.path.join(output_dir, f"{name}_downsampled{ext}")
    return downsample_file(input_file, output_file, original_resolution, target_resolution)

def process_tif_files(parent_directory, original_resolution, target_resolution):
    for root, dirs, files in os.walk(parent_directory):
        # Create or check downsampled_imgs directory
        downsampled_dir = os.path.join(root, "downsampled_imgs")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(downsampled_dir):
            os.makedirs(downsampled_dir)
            logging.info(f"Created directory: {downsampled_dir}")
        
        # Check if the downsampled directory is empty
        if os.listdir(downsampled_dir):
            logging.info(f"Skipping {root} - downsampled_imgs directory already contains files")
            continue
        
        # Collect TIFF files in the current directory
        tiff_files = []
        for file in files:
            if file.endswith(('.tif', '.tiff')):
                tiff_files.append(os.path.join(root, file))
        
        if not tiff_files:
            logging.info(f"No TIFF files found in {root}")
            continue
            
        logging.info(f"Processing {len(tiff_files)} files in {root}")
        
        # Process in batches of 4 files
        batch_size = 4
        # Use min(batch_size, cpu_count()) to never create more processes than files
        num_processes = min(batch_size, cpu_count(), len(tiff_files))
        
        try:
            for i in range(0, len(tiff_files), batch_size):
                batch = tiff_files[i:i + batch_size]
                process_args = [(file, original_resolution, target_resolution, downsampled_dir) for file in batch]
                
                with Pool(processes=num_processes) as pool:
                    pool.starmap(downsample_image, process_args)
                
                time.sleep(1)
            
            logging.info(f"Successfully processed all files in {root}")
        except Exception as e:
            logging.error(f"Error processing directory {root}: {str(e)}")

# Example usage:
# process_tif_files(r'path/to/your/parent/directory')

if __name__ == "__main__":
    process_tif_files(r'Z:\alexis_msc\p301s_baseline_LSFM', np.array([20, 6.55, 6.55]), np.array([20, 20, 20]))
    logging.info("Processing complete.")
