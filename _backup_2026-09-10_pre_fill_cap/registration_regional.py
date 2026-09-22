# Title: Analysis of regional tracer uptake
# Author: Phillip Muza 
# Date: 01/05/2025

# Import necessary libraries
import os
import pandas as pd
from skimage import io

class HemisphereAnalysis:
    def __init__(self, parent_dir, registration_dir=None):
        """Initialise the hemisphere analysis with image paths.

        parent_dir holds the segmented signal (thresholded_image.tif) and receives the
        left_hemisphere/right_hemisphere output folders. registration_dir holds the brainreg
        output and may live outside parent_dir, so a single registration can be shared by
        several signal channels. Defaults to parent_dir/registration_dir.
        """
        self.parent_dir = parent_dir
        self.registration_dir = registration_dir or os.path.join(parent_dir, 'registration_dir')
        self.registration_img_path = os.path.join(self.registration_dir, 'registered_atlas.tiff')
        self.signal_img_path = os.path.join(parent_dir, 'thresholded_image.tif')
        self.registered_hemispheres_path = os.path.join(self.registration_dir, 'registered_hemispheres.tiff')

        # Load images
        self.registration_img = io.imread(self.registration_img_path)
        self.signal_img = io.imread(self.signal_img_path)
        self.registered_hemispheres = io.imread(self.registered_hemispheres_path)

        # Load volumes data
        self.volumes = pd.read_csv(os.path.join(self.registration_dir, "volumes.csv"))
    
    def create_hemisphere_directory(self, hemisphere):
        """Create directory for the specified hemisphere."""
        dir_name = f"{hemisphere}_hemisphere"
        os.makedirs(os.path.join(self.parent_dir, dir_name), exist_ok=True)
        return dir_name
    
    def process_hemisphere(self, hemisphere):
        """Process a specific hemisphere (left or right)."""
        # Determine hemisphere ID and volume column
        if hemisphere == "left":
            hemisphere_id = 1
            volume_column = "left_volume_mm3"
        elif hemisphere == "right":
            hemisphere_id = 2
            volume_column = "right_volume_mm3"
        else:
            raise ValueError("Hemisphere must be 'left' or 'right'")
        
        # Create directory
        dir_name = self.create_hemisphere_directory(hemisphere)
        hemisphere_dir = os.path.join(self.parent_dir, dir_name)
        
        # Create hemisphere mask
        hemisphere_mask = (self.registered_hemispheres == hemisphere_id)
        
        # Apply mask to registration and signal images
        registration_mask = self.registration_img * hemisphere_mask
        signal_mask = self.signal_img * hemisphere_mask
        
        # Save masked images
        io.imsave(os.path.join(hemisphere_dir, "registration_mask.tiff"), registration_mask)
        io.imsave(os.path.join(hemisphere_dir, "signal_mask.tiff"), signal_mask)
        
        # Subset volumes data for this hemisphere
        hemisphere_volumes = self.volumes[['structure_name', volume_column]]
        hemisphere_volumes.to_csv(os.path.join(hemisphere_dir, "volumes.csv"), index=False)
        
        return {
            'dir_name': dir_name,
            'registration_mask': registration_mask,
            'signal_mask': signal_mask,
            'volumes': hemisphere_volumes
        }
    
    def process_all_hemispheres(self):
        """Process both hemispheres."""
        left_data = self.process_hemisphere("left")
        right_data = self.process_hemisphere("right")
        return {
            'left': left_data,
            'right': right_data
        }

# Example usage:
# analyser = HemisphereAnalysis(r"F:\tracer_uptake\p301s_mice\whole_brain_analysis\trial")
# results = analyser.process_all_hemispheres()








