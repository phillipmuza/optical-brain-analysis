import os
import numpy as np
from skimage.io import imread, imsave
from skimage.filters import median, threshold_triangle, gaussian, threshold_otsu 
from skimage.morphology import closing, disk, convex_hull_image, ball
from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening, median_filter
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
        self.original_preprocessed = None  # Keep original for iterative masking
        
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
        self.original_preprocessed = self.preprocessed_image.copy()  # Store original
        return self.preprocessed_image
    
    def _perform_slice_wise_binary_operations(self, single_slice_binary_mask):
        """Helper function to apply standard 2D binary operations on a single slice."""
        # These operations expect a binary image
        cleaned_mask = closing(single_slice_binary_mask, disk(3))
        hull_mask = convex_hull_image(cleaned_mask) # Can warn if cleaned_mask is all zero
        filled_mask = binary_fill_holes(hull_mask)
        return filled_mask.astype(np.uint8) * 255 # Return as uint8 with 0 and 255

    def _calculate_threshold(self, image_data, exclude_zeros=False):
        """Calculate threshold value, optionally excluding zero values."""
        if exclude_zeros:
            # Only consider non-zero pixels for threshold calculation
            non_zero_pixels = image_data[image_data > 0]
            if len(non_zero_pixels) == 0:
                return 0
            flat_image = non_zero_pixels.flatten()
        else:
            flat_image = image_data.flatten()
        
        hist, bin_edges = np.histogram(flat_image, bins=256)
        
        if exclude_zeros:
            # Find peak in the non-zero range
            peak_idx = np.argmax(hist)
        else:
            # Original method: look in first 50 bins
            peak_idx = np.argmax(hist[:50])
        
        background_thresh_grayscale = bin_edges[peak_idx + 1]
        margin = 1.001 
        return background_thresh_grayscale * margin

    def _compare_masks(self, mask1, mask2, threshold=0.01):
        """Compare two masks and return similarity measure."""
        if mask1.shape != mask2.shape:
            return 0
        
        # Calculate Dice coefficient
        intersection = np.sum(mask1 * mask2)
        total = np.sum(mask1) + np.sum(mask2)
        
        if total == 0:
            return 1.0  # Both masks are empty
        
        dice = (2.0 * intersection) / total
        return dice

    def generate_iterative_mask(self, max_iterations=2, convergence_threshold=0.99):
        """Generate mask using iterative refinement."""
        if self.preprocessed_image is None:
            raise ValueError("Please preprocess the image first.")

        current_image = self.preprocessed_image.copy()
        previous_mask = None
        
        print(f"Starting iterative mask generation (max {max_iterations} iterations)")
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Calculate threshold (exclude zeros after first iteration)
            exclude_zeros = iteration > 0
            final_threshold_value = self._calculate_threshold(current_image, exclude_zeros)
            print(f"Threshold value: {final_threshold_value:.2f}")

            # Generate mask for current iteration
            num_slices = current_image.shape[2]
            binary_mask_stack = np.zeros_like(current_image, dtype=np.uint8)
            
            for i in tqdm(range(num_slices), desc=f"Iteration {iteration + 1}", ncols=80):
                slice_data = current_image[:, :, i]
                slice_binary_mask = slice_data > final_threshold_value
                
                if np.sum(slice_binary_mask) == 0:
                    continue
                
                try:
                    processed_slice_binary = self._perform_slice_wise_binary_operations(slice_binary_mask)
                    binary_mask_stack[:, :, i] = (processed_slice_binary > 0).astype(np.uint8)
                except Exception as e:
                    binary_mask_stack[:, :, i] = slice_binary_mask.astype(np.uint8)
            
            # Check convergence
            if previous_mask is not None:
                similarity = self._compare_masks(binary_mask_stack, previous_mask)
                print(f"Mask similarity with previous iteration: {similarity:.4f}")
                
                if similarity >= convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break
            
            # Prepare for next iteration: apply mask to original preprocessed image
            if iteration < max_iterations - 1:  # Don't modify for last iteration
                current_image = self.original_preprocessed * (binary_mask_stack > 0)
                previous_mask = binary_mask_stack.copy()
        
        # Save final mask
        final_filename = os.path.join(self.output_dir, "brain_mask.tif")
        imsave(final_filename, binary_mask_stack.astype(np.uint8), check_contrast=False)
        print(f"\nFinal mask saved to {final_filename}")
        
        return binary_mask_stack

    def generate_final_mask(self):
        """Original single-pass mask generation method."""
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
    parser.add_argument("--iterative", action="store_true", help="Use iterative mask refinement")
    parser.add_argument("--max_iterations", type=int, default=2, help="Maximum number of iterations")
    args = parser.parse_args()

    actual_output_dir = args.output_dir
    if actual_output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(args.input_image))
        actual_output_dir = os.path.join(base_dir, "generated_masks_refined")
        if not base_dir:
             actual_output_dir = "generated_masks_refined"

    generator = MaskGenerator(output_dir=actual_output_dir)
    generator.load_image(args.input_image)
    generator.preprocess() 
    
    if args.iterative:
        final_mask = generator.generate_iterative_mask(max_iterations=args.max_iterations)
        print("Iterative mask generation complete.")
    else:
        final_mask = generator.generate_final_mask()
        print("Single-pass mask generation complete.")
    
    print(f"Output saved in {actual_output_dir}")

if __name__ == "__main__":
    main()