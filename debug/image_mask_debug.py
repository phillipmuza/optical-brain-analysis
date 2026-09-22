from skimage.io import imread, imsave
import numpy as np
from skimage.filters import gaussian
from skimage.morphology import closing, disk, convex_hull_image, ball
from scipy.ndimage import binary_fill_holes

img = imread(r"F:\tracer_uptake\p301s_mice\processed_images\an1\fitc_sliced.tif")

def preprocess_for_mask(image, sigma=2):
    """Light preprocessing for light-sheet mask generation"""
    # Skip median - use only gaussian
    preprocessed = gaussian(image, sigma=sigma, preserve_range=True, truncate=4.0)
    return preprocessed.astype(image.dtype)  # Preserve original data type

preprocessed_image = preprocess_for_mask(img)

imsave(r"F:\tracer_uptake\p301s_mice\processed_images\an1\preprocessed_image.tif", preprocessed_image)

print("=== DIAGNOSTIC INFO ===")
print(f"Image shape: {preprocessed_image.shape}")
print(f"Image dtype: {preprocessed_image.dtype}")
print(f"Image min: {np.min(preprocessed_image)}")
print(f"Image max: {np.max(preprocessed_image)}")
print(f"Image mean: {np.mean(preprocessed_image)}")
print(f"Image std: {np.std(preprocessed_image)}")

def _perform_slice_wise_binary_operations(single_slice_binary_mask):
        """Helper function to apply standard 2D binary operations on a single slice."""
        # These operations expect a binary image
        cleaned_mask = closing(single_slice_binary_mask, disk(3))
        hull_mask = convex_hull_image(cleaned_mask) # Can warn if cleaned_mask is all zero
        filled_mask = binary_fill_holes(hull_mask)
        return filled_mask.astype(np.uint8) * 255 # Return as uint8 with 0 and 255

# Thresholding strategy
flat_preprocessed_image = preprocessed_image.flatten()
hist, bin_edges = np.histogram(flat_preprocessed_image, bins=256)
peak_idx = np.argmax(hist[:50])
background_thresh_grayscale = bin_edges[peak_idx + 1]
margin = 1.001
final_threshold_value = background_thresh_grayscale * margin
print(f"Computing masks using threshold value: {final_threshold_value:.2f}")

# Binarize and Apply 2D operations slice by slice
num_slices = preprocessed_image.shape[2]
binary_mask_stack = np.zeros_like(preprocessed_image, dtype=np.uint8) 

for i in range(num_slices):
    slice_data = preprocessed_image[:, :, i]
    slice_binary_mask = slice_data > final_threshold_value

    try:
        processed_slice_binary = _perform_slice_wise_binary_operations(slice_binary_mask)
        binary_mask_stack[:, :, i] = processed_slice_binary
    except Exception as e:
        # Fallback: use the simply thresholded mask, converted to uint8
        binary_mask_stack[:, :, i] = slice_binary_mask.astype(np.uint8) * 255

imsave(r"F:\tracer_uptake\p301s_mice\processed_images\an1\binary_mask_stack.tif", binary_mask_stack)

