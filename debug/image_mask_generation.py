import os
import numpy as np
from skimage.io import imread, imsave
from skimage.filters import gaussian
from skimage.morphology import closing, disk, convex_hull_image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import argparse


class MaskGenerator:
    def __init__(self, output_dir="generated_masks", image_stack_to_process=None):
        """Initialize the mask generator"""
        self.image_path = None
        self.output_dir = output_dir
        self.image_stack = None
        self.direct_input_stack = image_stack_to_process
        self.preprocessed_image = None # Grayscale, after Gaussian blur
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def load_image(self, image_path):
        """Load a 3D image stack (if not provided directly)"""
        if self.direct_input_stack is not None:
            print("Image stack already provided directly, load_image call ignored.")
            return
        print(f"Loading image from {image_path}")
        self.image_path = image_path
        self.image_stack = imread(image_path)
        print(f"Image loaded with shape {self.image_stack.shape}")
        
    def preprocess(self): 
        """Apply Gaussian blurring as preprocessing."""
        image_to_process = None
        if self.direct_input_stack is not None:
            image_to_process = self.direct_input_stack
            print("Processing direct input stack.")
        elif self.image_stack is not None:
            image_to_process = self.image_stack
            print("Processing loaded image stack.")
        else:
            raise ValueError("No image data available for MaskGenerator preprocessing. Load an image or provide an image_stack.")
            
        if not isinstance(image_to_process, np.ndarray) or image_to_process.ndim != 3:
            raise ValueError("Input to MaskGenerator preprocess must be a 3D numpy array")
        
        # Apply Gaussian blur
        self.preprocessed_image = np.maximum(gaussian(image_to_process, sigma=2, preserve_range=True, truncate=4.0), 0) 
        return self.preprocessed_image
    
    def _perform_slice_wise_binary_operations(self, single_slice_binary_mask):
        """Helper function to apply standard 2D binary operations on a single slice."""
        # These operations expect a binary image
        cleaned_mask = closing(single_slice_binary_mask, disk(3))
        hull_mask = convex_hull_image(cleaned_mask) # Can warn if cleaned_mask is all zero
        filled_mask = binary_fill_holes(hull_mask)
        return filled_mask.astype(np.uint8) * 255 # Return as uint8 with 0 and 255

    def generate_final_mask(self):
        if self.preprocessed_image is None:
            raise ValueError("Please preprocess the image first.")

        # 1. Determine Thresholding Strategy using the gaussian blurred image directly
        flat_preprocessed_image = self.preprocessed_image.flatten()
        hist, bin_edges = np.histogram(flat_preprocessed_image, bins=256)
        peak_idx = np.argmax(hist[:50]) # Consider first 50 bins for background peak
        background_thresh_grayscale = bin_edges[peak_idx + 1]
        margin = 1.001 
        final_threshold_value = background_thresh_grayscale * margin
        print(f"Computing masks using threshold value: {final_threshold_value:.2f}")

        # 2. Binarize and Apply 2D operations slice by slice
        num_slices = self.preprocessed_image.shape[2]
        # Initialize with zeros (binary mask)
        binary_mask_stack = np.zeros_like(self.preprocessed_image, dtype=np.uint8) 

        # Track empty slices for summary reporting
        empty_slices_count = 0
        first_empty_slice = None
        last_empty_slice = None

        print("Processing slices...")
        for i in tqdm(range(num_slices), desc="Creating mask", ncols=80):
            slice_data = self.preprocessed_image[:, :, i]
            
            # Binarize the slice
            slice_binary_mask = slice_data > final_threshold_value
            
            if np.sum(slice_binary_mask) == 0: # If slice is empty after thresholding
                binary_mask_stack[:, :, i] = 0
                empty_slices_count += 1
                if first_empty_slice is None:
                    first_empty_slice = i
                last_empty_slice = i
                continue # Skip binary operations if mask is empty

            # Apply 2D binary operations
            try:
                processed_slice_binary = self._perform_slice_wise_binary_operations(slice_binary_mask)
                binary_mask_stack[:, :, i] = (processed_slice_binary > 0).astype(np.uint8)  # Ensure 0 or 1
            except Exception as e:
                # Fallback: use the simply thresholded mask, converted to uint8
                binary_mask_stack[:, :, i] = slice_binary_mask.astype(np.uint8)  # Boolean to 0 or 1

        # Report summary of empty slices
        if empty_slices_count > 0:
            print(f"Found {empty_slices_count} empty slices out of {num_slices} total slices.")
            print(f"Empty slice range: {first_empty_slice}-{last_empty_slice} (non-continuous)")

        # Save the final mask
        mask_filename = os.path.join(self.output_dir, "brain_mask.tif")
        imsave(mask_filename, binary_mask_stack.astype(np.uint8), check_contrast=False) # Ensure uint8 for saving
        print(f"Saved brain mask to {mask_filename}")
        return binary_mask_stack

def main():
    parser = argparse.ArgumentParser(description="Generate 3D image masks.")
    parser.add_argument("--input_image", required=True, help="Path to the input 3D image file.")
    parser.add_argument("--output_dir", default=None, help="Directory to save generated masks.")
    args = parser.parse_args()

    actual_output_dir = args.output_dir
    if actual_output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(args.input_image))
        actual_output_dir = os.path.join(base_dir, "generated_masks_refined") # New default subfolder
        if not base_dir: # if input is just 'file.tif'
             actual_output_dir = "generated_masks_refined"


    generator = MaskGenerator(output_dir=actual_output_dir)
    generator.load_image(args.input_image)
    generator.preprocess() 
    final_mask = generator.generate_final_mask() 
    
    print(f"Mask generation complete. Output saved in {actual_output_dir}")

if __name__ == "__main__":
    main() 