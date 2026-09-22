# Title: Brain registration using Brainreg (Brainglobe)
# Author: Phillip Muza
# Date:  22.08.24

import os
import logging
from plumbum import local, ProcessExecutionError, FG
from skimage import io
from registration_preprocessing import preprocess_for_registration

class BrainRegistration:
    def __init__(self, input_file, output_dir, voxel_size="20 20 20", orientation="asr", preprocess=True, atlas="perens_lsfm_mouse_20um"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        self.orientation = orientation
        self.preprocess = preprocess
        self.atlas = atlas
        self.logger = logging.getLogger(__name__)
        self.preprocessed_file = None
        
    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")
        else:
            self.logger.info(f"Output directory already exists: {self.output_dir}")
    
    def preprocess_image(self):
        """Preprocess the input image to improve registration quality"""
        if not self.preprocess:
            return self.input_file
            
        self.logger.info(f"Preprocessing image for improved registration: {self.input_file}")
        source_img = io.imread(self.input_file)

        # Look for a previously generated mask in the output dir first, then the cwd
        mask_path = next(
            (p for p in (os.path.join(self.output_dir, "brain_mask.tif"), "brain_mask.tif")
             if os.path.exists(p)),
            None
        )
        if mask_path:
            self.logger.info(f"Using existing mask file: {mask_path}")
            processed_img = preprocess_for_registration(source_img, use_existing_mask=True, mask_path=mask_path)
        else:
            self.logger.info("No existing mask file found. Generating mask during preprocessing.")
            processed_img = preprocess_for_registration(source_img, mask_output_dir=self.output_dir)

        # Save the preprocessed image in the output directory
        self.preprocessed_file = os.path.join(self.output_dir, "preprocessed.tiff")
        io.imsave(self.preprocessed_file, processed_img.astype(source_img.dtype))
        self.logger.info(f"Saved preprocessed image to: {self.preprocessed_file}")
        
        return self.preprocessed_file
        
    def generate_command(self):
        input_file = self.preprocessed_file if self.preprocess and self.preprocessed_file else self.input_file
        return f"brainreg {input_file} {self.output_dir} -v {self.voxel_size} --orientation {self.orientation} --atlas {self.atlas}"
    
    def run_registration(self):
        self.create_output_dir()
        
        # Preprocess the image if requested
        input_file = self.preprocess_image()
        
        try:
            brainreg = local['brainreg']
            # Split voxel size into separate values
            voxel_params = self.voxel_size.split()
            self.logger.info(f"Running registration with input: {input_file}")
            result = brainreg[input_file, 
                            self.output_dir, 
                            '-v', *voxel_params,  # Unpack the voxel parameters
                            '--orientation', self.orientation,
                            '--atlas', self.atlas] & FG
            self.logger.info("Brain registration completed successfully.")
            return True
        except ProcessExecutionError as e:
            self.logger.error(f"An error occurred: {e}")
            return False

# Example usage:
# registrator = BrainRegistration(input_file="path/to/your/input/file.tif", output_dir="registration")
# registrator.run_registration()
