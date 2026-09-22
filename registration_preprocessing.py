# Title: Preprocessing 3D brain image to improve registration by reducing bright surface signals
# Author: Phillip Muza
# Date: 01.05.2025

import numpy as np
from scipy import ndimage
from skimage import io, exposure, filters
from mask_generation import MaskGenerator

def preprocess_for_registration(source_img, use_existing_mask=False, mask_path=None, mask_output_dir="."):
    """
    Preprocess 3D brain image to improve registration by reducing bright surface signals.

    Parameters:
    source_img (numpy.ndarray): 3D source image to preprocess
    use_existing_mask (bool): Whether to use an existing mask file
    mask_path (str): Path to existing mask file (only used if use_existing_mask=True)
    mask_output_dir (str): Where a newly generated brain_mask.tif is written

    Returns:
    numpy.ndarray: Preprocessed image with surface signals reduced
    """
    # Create brain mask
    if use_existing_mask and mask_path:
        print(f"Using existing brain mask from {mask_path}")
        brain_mask = io.imread(mask_path).astype(bool)
        if brain_mask.shape != source_img.shape:
            raise ValueError("Mask dimensions do not match image dimensions")
    else:
        print("Generating brain mask using MaskGenerator")
        mask_maker = MaskGenerator(output_dir=mask_output_dir, image_stack_to_process=source_img)
        mask_maker.preprocess()
        brain_mask = mask_maker.generate_iterative_mask().astype(bool)

    # Calculate distance from brain surface
    distance_map = ndimage.distance_transform_edt(brain_mask)
    distance_map = distance_map / np.max(distance_map)  # Normalize to [0,1]
    
    # Initialize output with original values
    processed_img = source_img.copy().astype(float)
    
    # Define regions
    outer_region = (distance_map < 0.15) & brain_mask
    transition_zone = (distance_map >= 0.15) & (distance_map < 0.30) & brain_mask
    
    # Process outer region
    if np.sum(outer_region) > 0:
        outer_voxels = source_img[outer_region]
        p2 = np.percentile(outer_voxels, 2)
        p98 = np.percentile(outer_voxels, 98)
        local_mean = np.mean(outer_voxels)
        
        # Non-linear transform to compress bright values
        # This compresses very bright values more than moderate values
        adjusted_voxels = np.arctan(outer_voxels / local_mean) * local_mean * 2 / np.pi
        
        # Calculate percentiles of transformed values for rescaling
        transformed_p2 = np.percentile(adjusted_voxels, 2)
        transformed_p98 = np.percentile(adjusted_voxels, 98)
        
        # Rescale the transformed values to maintain good contrast
        adjusted_voxels = exposure.rescale_intensity(
            adjusted_voxels, 
            in_range=(transformed_p2, transformed_p98),  
            out_range=(p2, local_mean * 1.4)
        )
        
        processed_img[outer_region] = adjusted_voxels
    
    # Smooth transition between processed outer region and original inner regions
    if np.sum(transition_zone) > 0:
        weights = (0.30 - distance_map) / 0.15  # Linear interpolation
        weights = np.clip(weights, 0.0, 1.0)
        
        transition_voxels = source_img[transition_zone]
        p1 = np.percentile(transition_voxels, 1)  
        p98 = np.percentile(transition_voxels, 98)
        local_mean = np.mean(transition_voxels)
        
        cap_val = p98 * 0.90
        adjusted_transition = np.minimum(transition_voxels, cap_val)
        adjusted_transition = exposure.rescale_intensity(
            adjusted_transition, 
            in_range=(p1, cap_val),  
            out_range=(p1, local_mean * 1.3)  
        )
        
        for i, idx in enumerate(zip(*np.where(transition_zone))):
            weight = weights[idx]
            processed_img[idx] = weight * adjusted_transition[i] + (1.0 - weight) * source_img[idx]
    
    # Final smoothing to reduce artifacts
    processed_img = ndimage.gaussian_filter(processed_img, sigma=0.3)
    
    return processed_img

# Example usage:
# source_img = io.imread('path/to/your/source/image.tif')
# processed_img = preprocess_for_registration(source_img)
# io.imsave('processed.tif', processed_img.astype(np.uint16))
#
# To use an existing mask:
# processed_img = preprocess_for_registration(source_img, use_existing_mask=True, mask_path='brain_mask.tif')