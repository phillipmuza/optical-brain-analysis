# Title: Debug and improve brain masking methods
# Author: Phillip Muza
# Date: 24.08.24

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import median, gaussian, threshold_li, threshold_triangle, threshold_otsu, threshold_yen
from skimage.morphology import closing, disk, binary_dilation, convex_hull_image, ball
from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening, median_filter
from tqdm import tqdm

class MaskDebugger:
    def __init__(self, image_path=None, output_dir="debug_masks"):
        """Initialize the mask debugger with an image"""
        self.image_path = image_path
        self.output_dir = output_dir
        self.image_stack = None
        self.median_filtered = None
        self.mask_results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load image if path is provided
        if image_path:
            self.load_image(image_path)
            
    def load_image(self, image_path):
        """Load a 3D image stack"""
        print(f"Loading image from {image_path}")
        self.image_path = image_path
        self.image_stack = imread(image_path)
        print(f"Image loaded with shape {self.image_stack.shape}")
        
    def preprocess(self, kernel_size=3):
        """Apply median filtering as preprocessing"""
        print(f"Applying 3D median filter with kernel size {kernel_size}...")
        footprint = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
        self.median_filtered = np.maximum(median(self.image_stack, footprint=footprint), 0)
        print(f"Image range after filtering: {np.min(self.median_filtered)} to {np.max(self.median_filtered)}")
        
        # Save a sample slice for reference
        middle_slice = self.median_filtered.shape[2] // 2
        plt.figure(figsize=(10, 8))
        plt.imshow(self.median_filtered[:, :, middle_slice], cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.title('Median Filtered Slice')
        plt.savefig(os.path.join(self.output_dir, "median_filtered_slice.png"))
        plt.close()
        
        return self.median_filtered
    
    def analyze_histogram(self, bins=256, percentile_cutoff=99.5):
        """Analyze and plot the histogram of the filtered image"""
        if self.median_filtered is None:
            raise ValueError("Please preprocess the image first")
        
        # Flatten the image for histogram analysis
        flat_image = self.median_filtered.flatten()
        
        # Calculate histogram
        hist, bin_edges = np.histogram(flat_image, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate cumulative histogram
        cumulative = np.cumsum(hist) / np.sum(hist)
        
        # Find percentile cutoff for better visualization
        intensity_cutoff = np.percentile(flat_image, percentile_cutoff)
        
        # Plot histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Regular histogram
        ax1.plot(bin_centers, hist, 'b-')
        ax1.set_title('Intensity Histogram')
        ax1.set_xlabel('Intensity Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Zoom in on the relevant range
        ax1.set_xlim(0, intensity_cutoff)
        
        # Highlight first major peak
        peak_idx = np.argmax(hist[:50])
        background_thresh = bin_edges[peak_idx + 1]
        ax1.axvline(x=background_thresh, color='r', linestyle='--', 
                   label=f'Background threshold: {background_thresh:.2f}')
        ax1.legend()
        
        # Cumulative histogram
        ax2.plot(bin_centers, cumulative, 'g-')
        ax2.set_title('Cumulative Histogram')
        ax2.set_xlabel('Intensity Value')
        ax2.set_ylabel('Cumulative Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, intensity_cutoff)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "histogram_analysis.png"))
        plt.close()
        
        print(f"Background threshold from histogram peak: {background_thresh:.2f}")
        return background_thresh
    
    def create_histogram_mask(self, slice_idx=None, margin=1.075, threshold_method='triangle'):
        """Create a mask based on histogram analysis followed by local thresholding"""
        if self.median_filtered is None:
            raise ValueError("Please preprocess the image first")
            
        # If no slice index provided, use the middle slice
        if slice_idx is None:
            slice_idx = self.median_filtered.shape[2] // 2
            
        # Analyze histogram to find background threshold
        flat_image = self.median_filtered.flatten()
        hist, bin_edges = np.histogram(flat_image, bins=256)
        peak_idx = np.argmax(hist[:50])  # Look only in first 50 bins for background
        background_thresh = bin_edges[peak_idx + 1]
        
        # Get the slice data
        slice_data = self.median_filtered[:, :, slice_idx]
        
        # First threshold: remove background using histogram-based value
        initial_mask = slice_data > (background_thresh * margin)
        
        # Second threshold: refine with specified method on the remaining signal
        foreground_pixels = slice_data[initial_mask]
        
        if len(foreground_pixels) > 0:
            if threshold_method == 'triangle':
                refined_thresh = threshold_triangle(foreground_pixels)
            elif threshold_method == 'li':
                refined_thresh = threshold_li(foreground_pixels)
            elif threshold_method == 'otsu':
                refined_thresh = threshold_otsu(foreground_pixels)
            elif threshold_method == 'yen':
                refined_thresh = threshold_yen(foreground_pixels)
            else:
                refined_thresh = threshold_triangle(foreground_pixels)
                
            refined_mask = slice_data > refined_thresh
        else:
            refined_mask = initial_mask
        
        # Clean up mask with closing and hole filling
        cleaned_mask = closing(refined_mask, disk(3))
        hull_mask = convex_hull_image(cleaned_mask)
        filled_mask = binary_fill_holes(hull_mask)
        
        # Store the results
        mask_name = f"histogram_{threshold_method}"
        self.mask_results[mask_name] = {
            'initial_mask': initial_mask.copy(),
            'refined_mask': refined_mask.copy(),
            'cleaned_mask': cleaned_mask.copy(),
            'hull_mask': hull_mask.copy(),
            'final_mask': filled_mask.copy(),
            'background_thresh': background_thresh,
            'refined_thresh': refined_thresh if 'refined_thresh' in locals() else None,
            'margin': margin,
            'method': threshold_method
        }
        
        return filled_mask
    
    def create_simple_threshold_mask(self, slice_idx=None, threshold_method='otsu', cleanup=True):
        """Create a mask using a simple thresholding approach"""
        if self.median_filtered is None:
            raise ValueError("Please preprocess the image first")
            
        # If no slice index provided, use the middle slice
        if slice_idx is None:
            slice_idx = self.median_filtered.shape[2] // 2
            
        # Get the slice data
        slice_data = self.median_filtered[:, :, slice_idx]
        
        # Apply threshold
        if threshold_method == 'otsu':
            thresh = threshold_otsu(slice_data)
        elif threshold_method == 'li':
            thresh = threshold_li(slice_data)
        elif threshold_method == 'triangle':
            thresh = threshold_triangle(slice_data)
        elif threshold_method == 'yen':
            thresh = threshold_yen(slice_data)
        else:
            thresh = threshold_otsu(slice_data)
            
        # Create initial mask
        mask = slice_data > thresh
        
        # Apply cleanup if requested
        if cleanup:
            mask = closing(mask, disk(3))
            mask = convex_hull_image(mask)
            mask = binary_fill_holes(mask)
            
        # Store the results
        mask_name = f"simple_{threshold_method}"
        self.mask_results[mask_name] = {
            'initial_mask': mask.copy() if not cleanup else None,
            'final_mask': mask.copy(),
            'threshold': thresh,
            'method': threshold_method
        }
        
        return mask
    
    def try_all_methods(self, slice_idx=None):
        """Try all masking methods and compare results"""
        # Try histogram-based methods with different threshold techniques
        for method in ['triangle', 'li', 'otsu', 'yen']:
            self.create_histogram_mask(slice_idx=slice_idx, threshold_method=method)
            
        # Try simple threshold methods
        for method in ['otsu', 'li', 'triangle', 'yen']:
            self.create_simple_threshold_mask(slice_idx=slice_idx, threshold_method=method)
            
        # Compare the results
        self.compare_masks(slice_idx=slice_idx)
        
    def compare_masks(self, slice_idx=None):
        """Compare the different masking methods"""
        if not self.mask_results:
            raise ValueError("No mask results available for comparison")
            
        # If no slice index provided, use the middle slice
        if slice_idx is None:
            slice_idx = self.median_filtered.shape[2] // 2
            
        slice_data = self.median_filtered[:, :, slice_idx]
        
        # Create a figure with subplots
        num_methods = len(self.mask_results)
        rows = (num_methods + 2) // 3  # +2 to include original image
        fig, axes = plt.subplots(rows, 3, figsize=(20, 6*rows))
        axes = axes.flatten()
        
        # Plot the original image first
        axes[0].imshow(slice_data, cmap='viridis')
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Plot each mask result
        for i, (name, result) in enumerate(self.mask_results.items(), start=1):
            if i >= len(axes):
                break
                
            axes[i].imshow(result['final_mask'], cmap='gray')
            
            # Prepare title with threshold info if available
            title = name
            if 'threshold' in result:
                title += f"\nThresh: {result['threshold']:.2f}"
            elif 'background_thresh' in result and 'refined_thresh' in result and result['refined_thresh'] is not None:
                title += f"\nBkg: {result['background_thresh']:.2f}, Ref: {result['refined_thresh']:.2f}"
                
            axes[i].set_title(title, fontsize=12)
            axes[i].axis('off')
            
        # Hide any unused subplots
        for i in range(1 + len(self.mask_results), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mask_comparison.png"))
        plt.close()
        
    def generate_debug_images(self, method_name, slice_idx=None):
        """Generate debug images showing intermediate steps for a specific method"""
        if method_name not in self.mask_results:
            raise ValueError(f"Method {method_name} not found in results")
            
        # If no slice index provided, use the middle slice
        if slice_idx is None:
            slice_idx = self.median_filtered.shape[2] // 2
            
        result = self.mask_results[method_name]
        slice_data = self.median_filtered[:, :, slice_idx]
        
        # Determine how many steps we have
        steps = []
        for key in ['initial_mask', 'refined_mask', 'cleaned_mask', 'hull_mask', 'final_mask']:
            if key in result and result[key] is not None:
                steps.append((key, result[key]))
                
        num_steps = len(steps) + 1  # +1 for original image
        
        # Create figure
        fig, axes = plt.subplots(1, num_steps, figsize=(5*num_steps, 5))
        if num_steps == 1:
            axes = [axes]
            
        # Plot original image
        axes[0].imshow(slice_data, cmap='viridis')
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Plot each processing step
        for i, (name, mask) in enumerate(steps, start=1):
            axes[i].imshow(mask, cmap='gray')
            title = name.replace('_', ' ').title()
            axes[i].set_title(title, fontsize=12)
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"debug_{method_name}.png"))
        plt.close()
        
    def apply_3d_smoothing(self, binary_mask, method='median', kernel_size=3, preserve_edges=True):
        """
        Apply 3D smoothing to reduce slice-by-slice variation in the mask.
        
        Parameters:
        binary_mask (numpy.ndarray): 3D binary mask
        method (str): Smoothing method ('median', 'morphological', or 'both')
        kernel_size (int): Size of the smoothing kernel
        preserve_edges (bool): Whether to preserve edge slices by copying from neighbors
        
        Returns:
        numpy.ndarray: Smoothed 3D mask
        """
        print(f"Applying 3D smoothing using {method} method...")
        smoothed_mask = binary_mask.copy()
        
        if method == 'median' or method == 'both':
            # 3D median filtering to remove small variations
            smoothed_mask = median_filter(smoothed_mask, size=kernel_size)
            
        if method == 'morphological' or method == 'both':
            # 3D morphological operations
            # Create a ball-shaped structuring element for 3D operations
            structure = ball(kernel_size // 2)
            
            # Close small holes and connect nearby structures
            smoothed_mask = binary_closing(smoothed_mask, structure=structure)
            
            # Optional: Remove small isolated regions
            smoothed_mask = binary_opening(smoothed_mask, structure=ball(1))
            
        # Ensure the mask is binary
        smoothed_mask = smoothed_mask > 0
        
        # Preserve edge slices by copying from neighbors
        if preserve_edges and smoothed_mask.shape[2] > 2:
            # First slice: copy from second slice if first is empty or significantly smaller
            if np.sum(smoothed_mask[:,:,0]) < 0.5 * np.sum(smoothed_mask[:,:,1]):
                print("Preserving first slice by copying from second slice")
                smoothed_mask[:,:,0] = smoothed_mask[:,:,1]
                
            # Last slice: copy from second-to-last slice if last is empty or significantly smaller
            if np.sum(smoothed_mask[:,:,-1]) < 0.5 * np.sum(smoothed_mask[:,:,-2]):
                print("Preserving last slice by copying from second-to-last slice")
                smoothed_mask[:,:,-1] = smoothed_mask[:,:,-2]
        
        return smoothed_mask
    
    def compare_smoothing(self, mask, slice_range=None):
        """
        Compare original mask with different smoothing methods
        
        Parameters:
        mask (numpy.ndarray): The original 3D mask
        slice_range (tuple): Range of slices to display (start, end)
        """
        if slice_range is None:
            # Default to middle slice +/- 5 slices
            middle = mask.shape[2] // 2
            slice_range = (middle - 5, middle + 6)  # Show 11 slices
        
        # Generate smoothed versions
        median_smoothed = self.apply_3d_smoothing(mask, method='median')
        morph_smoothed = self.apply_3d_smoothing(mask, method='morphological')
        both_smoothed = self.apply_3d_smoothing(mask, method='both')
        
        # Create comparison images
        fig, axes = plt.subplots(4, len(range(*slice_range)), figsize=(20, 16))
        
        for i, slice_idx in enumerate(range(*slice_range)):
            if slice_idx >= mask.shape[2]:
                continue
                
            # Original mask
            axes[0, i].imshow(mask[:, :, slice_idx], cmap='gray')
            axes[0, i].set_title(f"Original - Slice {slice_idx}")
            axes[0, i].axis('off')
            
            # Median smoothed
            axes[1, i].imshow(median_smoothed[:, :, slice_idx], cmap='gray')
            axes[1, i].set_title(f"Median - Slice {slice_idx}")
            axes[1, i].axis('off')
            
            # Morphological smoothed
            axes[2, i].imshow(morph_smoothed[:, :, slice_idx], cmap='gray')
            axes[2, i].set_title(f"Morphological - Slice {slice_idx}")
            axes[2, i].axis('off')
            
            # Both methods
            axes[3, i].imshow(both_smoothed[:, :, slice_idx], cmap='gray')
            axes[3, i].set_title(f"Both - Slice {slice_idx}")
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "smoothing_comparison.png"))
        plt.close()
        
        # Save smoothed masks
        imsave(os.path.join(self.output_dir, "median_smoothed_mask.tif"), median_smoothed.astype(np.uint8))
        imsave(os.path.join(self.output_dir, "morph_smoothed_mask.tif"), morph_smoothed.astype(np.uint8))
        imsave(os.path.join(self.output_dir, "both_smoothed_mask.tif"), both_smoothed.astype(np.uint8))
        
        return {
            'median': median_smoothed,
            'morphological': morph_smoothed,
            'both': both_smoothed
        }
        
    def apply_best_method_to_stack(self, method_name, margin=1.075, apply_smoothing=False, smoothing_method='both'):
        """Apply the best method to the entire stack"""
        if self.median_filtered is None:
            raise ValueError("Please preprocess the image first")
            
        print(f"Applying {method_name} to entire image stack...")
        img_mask = np.zeros_like(self.median_filtered, dtype="uint8")
        
        # First analyze the histogram of the entire volume to find background threshold
        flat_image = self.median_filtered.flatten()
        hist, bin_edges = np.histogram(flat_image, bins=256)
        peak_idx = np.argmax(hist[:50])
        background_thresh = bin_edges[peak_idx + 1]
        
        print(f"Background threshold from histogram: {background_thresh}")
        
        threshold_method = method_name.split('_')[1] if '_' in method_name else 'triangle'
        
        for i in tqdm(range(self.median_filtered.shape[2]), desc="Processing slices"):
            slice_data = self.median_filtered[:, :, i]
            
            if method_name.startswith('histogram'):
                # First threshold: remove background using histogram-based value
                initial_mask = slice_data > (background_thresh * margin)
                
                # Second threshold: refine with chosen method on the remaining signal
                foreground_pixels = slice_data[initial_mask]
                
                if len(foreground_pixels) > 0:
                    if threshold_method == 'triangle':
                        refined_thresh = threshold_triangle(foreground_pixels)
                    elif threshold_method == 'li':
                        refined_thresh = threshold_li(foreground_pixels)
                    elif threshold_method == 'otsu':
                        refined_thresh = threshold_otsu(foreground_pixels)
                    elif threshold_method == 'yen':
                        refined_thresh = threshold_yen(foreground_pixels)
                    else:
                        refined_thresh = threshold_triangle(foreground_pixels)
                        
                    refined_mask = slice_data > refined_thresh
                else:
                    refined_mask = initial_mask
                
                # Clean up mask
                cleaned_mask = closing(refined_mask, disk(3))
                hull_mask = convex_hull_image(cleaned_mask)
                filled_mask = binary_fill_holes(hull_mask)
                
            elif method_name.startswith('simple'):
                # Apply simple threshold
                if threshold_method == 'otsu':
                    thresh = threshold_otsu(slice_data)
                elif threshold_method == 'li':
                    thresh = threshold_li(slice_data)
                elif threshold_method == 'triangle':
                    thresh = threshold_triangle(slice_data)
                elif threshold_method == 'yen':
                    thresh = threshold_yen(slice_data)
                else:
                    thresh = threshold_otsu(slice_data)
                    
                # Create mask
                mask = slice_data > thresh
                filled_mask = closing(mask, disk(3))
                filled_mask = convex_hull_image(filled_mask)
                filled_mask = binary_fill_holes(filled_mask)
                
            img_mask[:, :, i] = filled_mask.astype(int)
        
        # Apply 3D smoothing if requested
        if apply_smoothing:
            print("Applying 3D smoothing to reduce slice-by-slice variation...")
            img_mask = self.apply_3d_smoothing(img_mask, method=smoothing_method)
        
        # Save the final mask
        mask_suffix = "_smoothed" if apply_smoothing else ""
        mask_filename = os.path.join(self.output_dir, f"{method_name}{mask_suffix}_mask.tif")
        imsave(mask_filename, img_mask.astype(np.uint8))
        print(f"Saved full stack mask to {mask_filename}")
        
        return img_mask
        
    def export_mask_settings(self, method_name, margin=1.075, apply_smoothing=True, smoothing_method='both'):
        """
        Export detailed settings used for mask generation to help identify differences between implementations.
        
        Parameters:
        method_name (str): The method used for masking (e.g., 'histogram_triangle')
        margin (float): Margin multiplier used for background threshold
        apply_smoothing (bool): Whether smoothing was applied
        smoothing_method (str): Which smoothing method was used
        """
        if self.median_filtered is None:
            raise ValueError("No mask has been generated yet. Call apply_best_method_to_stack first.")
            
        settings = {
            "method_name": method_name,
            "margin": margin,
            "apply_smoothing": apply_smoothing,
            "smoothing_method": smoothing_method if apply_smoothing else None,
            "histogram_settings": {
                "bins": 256,
                "background_threshold_range": 50,  # First 50 bins used for finding peak
            },
            "median_filter": {
                "kernel_size": 3,
                "footprint": "3D cube (kernel_size × kernel_size × kernel_size)",
            },
            "threshold_settings": {
                "method": method_name.split('_')[1] if '_' in method_name else 'triangle',
                "min_foreground_pixels": 0,  # In debug_image_mask we use > 0 check
                "foreground_check": "if len(foreground_pixels) > 0",
            },
            "cleaning_settings": {
                "closing_disk_size": 3,
                "convex_hull": True,
                "binary_fill_holes": True,
            },
            "smoothing_settings": {
                "kernel_size": 3,
                "median_filter_mode": "reflect",  # Default for median_filter
                "morphological_structure": f"ball({3//2})",
                "preserve_edges": True,
                "edge_copy_threshold": 0.5,  # Threshold for determining significantly smaller slices
            },
            "dtypes": {
                "median_filtered": str(self.median_filtered.dtype),
                "img_mask": "uint8",
                "smoothed_mask": "uint8",
            },
        }
        
        # Save settings to JSON file
        import json
        with open(os.path.join(self.output_dir, "mask_settings.json"), 'w') as f:
            json.dump(settings, f, indent=2)
            
        # Create a human-readable text version
        with open(os.path.join(self.output_dir, "mask_settings.txt"), 'w') as f:
            f.write("=== MASK GENERATION SETTINGS ===\n\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Background threshold margin: {margin}\n")
            f.write(f"Apply smoothing: {apply_smoothing}\n")
            if apply_smoothing:
                f.write(f"Smoothing method: {smoothing_method}\n")
            f.write("\n=== PREPROCESSING ===\n")
            f.write(f"Median filter kernel size: 3 (3D, {3}×{3}×{3})\n")
            f.write("\n=== THRESHOLDING ===\n")
            f.write(f"Background threshold: Using first 50 bins of histogram\n")
            f.write(f"Secondary threshold method: {method_name.split('_')[1] if '_' in method_name else 'triangle'}\n")
            f.write(f"Min foreground pixels check: > 0 (debug_image_mask)\n")
            f.write("\n=== SMOOTHING ===\n") 
            f.write(f"Median filter mode: reflect (default)\n")
            f.write(f"Edge preservation: {True}\n")
            
            # Add numpy and skimage version info
            import numpy as np
            import skimage
            f.write("\n=== LIBRARY VERSIONS ===\n")
            f.write(f"NumPy version: {np.__version__}\n")
            f.write(f"scikit-image version: {skimage.__version__}\n")
            
        print(f"Mask settings exported to {self.output_dir}/mask_settings.txt and .json")
        return settings
        
# Example usage
if __name__ == "__main__":
    # Configure input path here - change this to your actual file path
    input_path = r"F:\tracer_uptake\p301s_mice\whole_brain_analysis_2\an2\fitc\fitc.tif"  
    
    # Initialize the debugger
    debugger = MaskDebugger(input_path)
    
    # Preprocess the image
    debugger.preprocess(kernel_size=3)
    
    # Analyze histogram
    debugger.analyze_histogram()
    
    # Try and compare all methods
    debugger.try_all_methods()
    
    # Generate debug images for the histogram-triangle method
    debugger.generate_debug_images('histogram_triangle')
    
    # Apply best method to entire stack with and without smoothing
    mask = debugger.apply_best_method_to_stack('histogram_triangle', apply_smoothing=False)
    smoothed_mask = debugger.apply_best_method_to_stack('histogram_triangle', apply_smoothing=True, smoothing_method='both')
    
    # Compare different smoothing methods
    debugger.compare_smoothing(mask)
    
    # Export detailed settings for comparison
    debugger.export_mask_settings('histogram_triangle', margin=1.075, apply_smoothing=True, smoothing_method='both')
    
    print(f"Debug results saved to {debugger.output_dir} directory")
    print("Review the mask_comparison.png to select the best method") 