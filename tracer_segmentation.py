# Title: Segment tracer from brains, and quantify tracer coverage
# Author: Phillip Muza
# Date:  21.08.24

import numpy as np, pandas as pd
from skimage.filters import median, gaussian, threshold_yen, threshold_otsu
from skimage.io import imsave
from tqdm import tqdm
from mask_generation import MaskGenerator

class TracerSignalAnalyser:
    def __init__(self, image_stack):
        """
        Initialize the TracerSignalAnalyzer with the given image stack.

        Parameters:
        image_stack (numpy.ndarray): 3D image stack containing the fluorescent signal.
        """
        if not isinstance(image_stack, np.ndarray) or image_stack.ndim != 3:
            raise ValueError("Image stack must be a 3D numpy array")
        self.image_stack = image_stack
        self.median_filtered = None
        self.img_mask = None
        self.gaussian_blurred = None
        self.sharp_img = None
        self.results = None

    def median_filtering(self, kernel_size=3):
        """
        Apply a 3D median filter to the image stack.
        
        Parameters:
        kernel_size (int): Size of the median filter kernel. 
        """
        if kernel_size <= 0:
            print("Median filtering skipped as kernel_size <= 0.")
            self.median_filtered = self.image_stack.copy() # Use the original stack if no filtering
            return

        footprint = np.ones((kernel_size, kernel_size, kernel_size), dtype=bool)
        # Apply median filtering on the original self.image_stack
        print("Applying median filtering...")
        self.median_filtered = np.maximum(median(self.image_stack, footprint=footprint), 0)
        
    def gaussian_blur(self, sigma=(10, 10, 10)):
        """
        Apply Gaussian blur to the (masked) median-filtered image stack.
        """
        if self.median_filtered is None:
            raise ValueError("Median filtering must be applied before Gaussian blur.")

        print("Applying Gaussian blur...")
        # Ensure median_filtered is float for gaussian filter
        source_for_blur = self.median_filtered.astype(np.float32) 
        blurred = gaussian(source_for_blur, sigma=sigma, preserve_range=True)
        # Convert back to uint16 or original dtype as appropriate. Assuming uint16 for now.
        self.gaussian_blurred = blurred.astype(np.uint16) 

    def unsharp_masking(self, use_gaussian=True):
        """
        Perform unsharp masking.
        """
        if self.median_filtered is None:
            raise ValueError("Median filtering must be performed first.")
        if use_gaussian and self.gaussian_blurred is None:
            raise ValueError("Gaussian blur must be performed if use_gaussian is True for unsharp masking.")

        print("Performing unsharp masking...")
        if use_gaussian:
            # Ensure dtypes are compatible for subtraction if they aren't already
            median_filtered_for_unsharp = self.median_filtered.astype(np.int32) # Use a type that can handle potential negative intermediate
            gaussian_blurred_for_unsharp = self.gaussian_blurred.astype(np.int32)
            
            self.sharp_img = np.maximum(median_filtered_for_unsharp - gaussian_blurred_for_unsharp, 0).astype(self.image_stack.dtype) # cast back to original dtype
            
        else:
            self.sharp_img = self.median_filtered.copy() # Use the median_filtered image

    def measure_area(self, method='hybrid', weight=0.4):
        """
        Measure the percentage area of the tracer signal in each slice.
        
        Parameters:
        method (str): Thresholding method ('hybrid', 'yen', 'otsu')
                 weight (float): Weight for Yen in hybrid method (0.4 balances noise robustness with signal sensitivity)
        """
        if self.sharp_img is None:
            raise ValueError("Unsharp masking must be performed before measuring area.")

        print("Measuring the percentage area of the tracer signal...")
        data = {'slice': [], 'percentage_area': []}
        thresholded_image = np.zeros_like(self.sharp_img, dtype="uint8")

        for i in tqdm(range(self.sharp_img.shape[2]), desc="Measuring slices"):
            slice_data = self.sharp_img[:, :, i]
            
            if np.all(slice_data == 0):
                # If the entire slice is zero (e.g. fully masked out or no signal)
                percentage_area = 0.0
                binary_slice = np.zeros_like(slice_data, dtype=bool)
            else:
                if method == 'hybrid':
                    try:
                        thresh_yen = threshold_yen(slice_data)
                        thresh_otsu = threshold_otsu(slice_data)
                        # Balanced weighting: Yen for noise robustness, Otsu for signal sensitivity
                        hybrid_thresh = thresh_yen * weight + thresh_otsu * (1 - weight)
                        binary_slice = slice_data > hybrid_thresh
                    except Exception as e: # Catch potential errors if slice is weird (e.g. all same value after masking)
                        print(f"Warning: Hybrid threshold failed for slice {i} ({e}), using Yen as fallback.")
                        try: 
                            thresh_yen_fallback = threshold_yen(slice_data)
                            binary_slice = slice_data > thresh_yen_fallback
                        except: # Final fallback if Yen also fails
                            print(f"Warning: Yen threshold also failed for slice {i}, marking as no signal.")
                            binary_slice = np.zeros_like(slice_data, dtype=bool)

                elif method == 'yen':
                    try:
                        thresh = threshold_yen(slice_data)
                        binary_slice = slice_data > thresh
                    except:
                        print(f"Warning: Yen threshold failed for slice {i}, marking as no signal.")
                        binary_slice = np.zeros_like(slice_data, dtype=bool)
                elif method == 'otsu':
                    try:
                        thresh = threshold_otsu(slice_data)
                        binary_slice = slice_data > thresh
                    except:
                        print(f"Warning: Otsu threshold failed for slice {i}, marking as no signal.")
                        binary_slice = np.zeros_like(slice_data, dtype=bool)
                else:
                    raise ValueError(f"Unknown threshold method: {method}. Use 'hybrid', 'yen', or 'otsu'.")
            
            signal_area = np.sum(binary_slice)
            total_area = binary_slice.size
            percentage_area = (signal_area / total_area) * 100 if total_area > 0 else 0
            data['slice'].append(i + 1)
            data['percentage_area'].append(percentage_area)
            
            thresholded_image[:, :, i] = binary_slice.astype(np.uint8)
        
        self.results = pd.DataFrame(data)
        imsave("thresholded_image.tif", thresholded_image.astype(np.uint8) * 255, check_contrast=False) # Save as uint8 with 0 and 255

    def save_results(self, filename="tracer_percentage_area.csv"):
        """
        Save the results to a CSV file.
        """
        if self.results is None:
            print("No results to save. Run measure_area first.")
            return
        print(f"Saving results to {filename}...")
        self.results.to_csv(filename, index=False)
        

    def process(self, kernel_size=3, blur_sigma=(10, 10, 10), threshold_method='hybrid', weight_hybrid=0.4):
        """
        Run the complete processing pipeline.
        
        Parameters:
        kernel_size (int): Size of the median filter kernel for TracerSignalAnalyser's initial filtering.
        blur_sigma (tuple): Sigma values for Gaussian blur in (x,y,z).
        threshold_method (str): Method to use for final signal thresholding ('hybrid', 'yen', 'otsu').
                 weight_hybrid (float): Weight for Yen in hybrid thresholding (0.4 balances noise robustness with signal sensitivity).
        """
        print("Starting complete signal processing pipeline...")
        
        # Step 1: Apply median filtering to the original unmasked image
        self.median_filtering(kernel_size=kernel_size)
        
        # Step 2: Generate the image mask using MaskGenerator with the *original* image stack
        print("Initializing MaskGenerator to create brain mask...")
        # Assuming main.py changes cwd, output_dir='.' saves brain_mask.tif in the current base_dir
        mask_output_dir = "." 
        mask_maker = MaskGenerator(output_dir=mask_output_dir, image_stack_to_process=self.image_stack)
        mask_maker.preprocess() # MaskGenerator applies median + gaussian filtering
        self.img_mask = mask_maker.generate_iterative_mask() # Generates and saves brain_mask.tif, returns mask array

        if not isinstance(self.img_mask, np.ndarray):
            raise TypeError(f"MaskGenerator did not return a numpy array. Got: {type(self.img_mask)}")

        # Ensure mask is boolean for direct application
        self.img_mask = self.img_mask.astype(bool)

        # Step 3: Gaussian blur on unmasked data to avoid boundary artifacts
        self.gaussian_blur(sigma=blur_sigma)
        
        # Step 4: Unsharp masking on unmasked data
        self.unsharp_masking(use_gaussian=True)
        
        # Step 5: Apply mask AFTER unsharp masking to eliminate boundary artifacts
        if self.sharp_img.shape != self.img_mask.shape:
            raise ValueError(f"Shape mismatch: sharp_img is {self.sharp_img.shape} but generated mask is {self.img_mask.shape}")
        
        print("Applying brain mask after unsharp masking to avoid boundary artifacts...")
        self.sharp_img = self.sharp_img * self.img_mask.astype(self.sharp_img.dtype)
        print(f"Image range after applying brain mask: {np.min(self.sharp_img)} to {np.max(self.sharp_img)}")
        
        # Save the final masked unsharp image
        imsave("unsharp_image.tif", self.sharp_img)
        
        # Step 6: Measure area
        self.measure_area(method=threshold_method, weight=weight_hybrid)
        
        # Step 7: Save results (optional, can be called separately)
        # self.save_results()
        
        return self.results


