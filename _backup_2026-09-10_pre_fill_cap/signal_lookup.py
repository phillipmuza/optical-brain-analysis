# Title: Performing nearest-neighbour lookup between two 3D images
# Author: Phillip Muza
# Date: 23.08.24

import os
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.spatial import cKDTree

class ImageAnalyser:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.chdir(self.base_dir)
    
    # Load images
    def load_images(self, tracer_file, registration_file):
        self.tracer_signal = imread(tracer_file)
        self.registration = imread(registration_file)
        # Ensure both images have the same shape
        if self.tracer_signal.shape != self.registration.shape:
            raise ValueError(
                f"Signal and registration images must have the same shape, but "
                f"{tracer_file} is {self.tracer_signal.shape} and "
                f"{registration_file} is {self.registration.shape}. "
                "This usually means the signal and registration images were not downsampled "
                "to the same voxel size before registration."
            )

    # Process images to find corresponding pixels between tracer and registration images    
    def process_images(self):
        
        # Find non-zero pixels in the tracer signal
        non_zero_mask = self.tracer_signal > 0
        # Get coordinates of non-zero pixels in the signal mask
        self.non_zero_coords = np.argwhere(non_zero_mask)
        # Get corresponding values from the registered atlas
        self.corresponding_regions = self.registration[non_zero_mask]
        
    # Some corresponding values in the corresponding_regions are zero. We are going to use nearest-neighbour to assign these to the closest region
        # We are taking the assumption that whatever signal obtained from the tracer_signal has to be somewhere in the brain, and not in space.
    def fill_zero_regions(self):
        # Find indices where atlas values are zero
        zero_indices = np.where(self.corresponding_regions == 0)[0]
        
        if len(zero_indices) > 0:
            # Get coordinates of non-zero pixels in the atlas
            atlas_non_zero_coords = np.argwhere(self.registration > 0)
             # Build KD-tree for efficient nearest neighbor search
            tree = cKDTree(atlas_non_zero_coords)
            # For each zero-valued pixel, find the nearest non-zero pixel in the atlas
            for idx in zero_indices:
                _, nearest_idx = tree.query(self.non_zero_coords[idx])
                self.corresponding_regions[idx] = self.registration[tuple(atlas_non_zero_coords[nearest_idx])]

    # Generate a dataframe describing the amount of tracer signal in a given brain region             
    def count_regions(self):
        # Count occurrences of each unique region
        region_counts = np.bincount(self.corresponding_regions.astype(int))
        # Create a DataFrame with region IDs and their counts
        df = pd.DataFrame({
            'Region_ID': np.arange(len(region_counts)),
            'Signal_Pixel_Count': region_counts
        })
        # Remove rows where count is zero (regions not present in the signal)
        df = df[df['Signal_Pixel_Count'] > 0]
        # Sort by count in descending order
        df = df.sort_values('Signal_Pixel_Count', ascending=False)
        # Reset index
        self.df = df.reset_index(drop=True)

    # Save the dataframe  
    def save_results(self, filename='region_counts.csv'):
        self.df.to_csv(filename, index=False)

    # Region_counts is coded, here we decode the regions in region counts and 
        # combine the volumes file generated during registration   
    def merge_data(self, structures_file, volumes_file):
        # Read structures file
        structures = pd.read_csv(structures_file)
        # Devode region counts
        self.decoded_df = pd.merge(structures, self.df, left_on='id', right_on='Region_ID', how='left')
        
        # Read volumes df
        volumes = pd.read_csv(volumes_file)
        # Include volumes of regions to region counts
        self.decoded_df = pd.merge(self.decoded_df, volumes, left_on='name', right_on='structure_name', how='left')
        self.decoded_df.to_csv('decoded_region_counts.csv')

    # Run the analysis 
    def run_analysis(self, tracer_file, registration_file, structures_file, volumes_file):
        self.load_images(tracer_file, registration_file)
        self.process_images()
        self.fill_zero_regions()
        self.count_regions()
        self.save_results()
        self.merge_data(structures_file, volumes_file)

# Example usage:
# analyser = ImageAnalyser("path/to/your/parent/directory")
# analyser.run_analysis(
#     "thresholded_image.tif",
#     "registration/registered_atlas.tiff",
#     "path/to/structures.csv",
#     "registration/volumes.csv"
# )
# print(analyser.decoded_df)