import skimage.io as skio
import numpy as np
import os
import yaml
from pathlib import Path

def load_config(config_file):
    """
    Load folder configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_file} not found")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def slice_images(home_directory, folder_configs):
    """
    Slice 3D images in specified folders along the z-axis.
    
    Args:
        home_directory (str): Base directory path containing the folders
        folder_configs (dict): Dictionary where keys are folder names (relative to home_directory) 
                              and values are dictionaries with slicing parameters:
                              {
                                  'folder_name': {
                                      'start': int,  # starting slice index
                                      'end': int,    # ending slice index (exclusive)
                                      'step': int,   # step size (optional, default=1)
                                      'recursive': bool  # whether to search recursively (optional, default=False)
                                  }
                              }
    """
    home = Path(home_directory)
    
    if not home.exists():
        print(f"Error: Home directory {home_directory} does not exist")
        return
    
    for folder_name, slice_config in folder_configs.items():
        folder = home / folder_name
        
        if not folder.exists():
            print(f"Warning: Folder {folder} does not exist, skipping...")
            continue
            
        # Get slicing parameters
        start_slice = slice_config.get('start', 0)
        end_slice = slice_config.get('end', None)
        step = slice_config.get('step', 1)
        recursive = slice_config.get('recursive', False)
        
        # Find all .tif files in the folder (recursively if specified)
        if recursive:
            tif_files = list(folder.glob('**/*.tif')) + list(folder.glob('**/*.tiff'))
        else:
            tif_files = list(folder.glob('*.tif')) + list(folder.glob('*.tiff'))
        
        if not tif_files:
            search_type = "recursively" if recursive else "in"
            print(f"No .tif files found {search_type} {folder}")
            continue
            
        search_type = " (including subdirectories)" if recursive else ""
        print(f"Processing {len(tif_files)} files in {folder_name}{search_type}")
        
        for tif_file in tif_files:
            try:
                # Load the image using skimage
                image = skio.imread(str(tif_file))
                
                # Get image dimensions
                shape = image.shape
                
                # Check if image is 3D
                if len(shape) < 3:
                    print(f"Warning: {tif_file.name} is not a 3D image, skipping...")
                    continue
                
                # Adjust end_slice if not specified
                if end_slice is None:
                    end_slice = shape[0]
                
                # Validate slice indices
                if start_slice >= shape[0] or end_slice > shape[0] or start_slice >= end_slice:
                    print(f"Warning: Invalid slice range for {tif_file.name}, skipping...")
                    continue
                
                # Extract the slice range (z-axis is the first dimension in ZYX format)
                sliced_image = image[start_slice:end_slice:step, :, :]
                
                # Create output filename and preserve directory structure
                if recursive and tif_file.parent != folder:
                    # Maintain relative directory structure for recursive searches
                    relative_path = tif_file.relative_to(folder)
                    output_dir = folder / relative_path.parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_name = f"{tif_file.stem}_sliced{tif_file.suffix}"
                    output_path = output_dir / output_name
                else:
                    output_name = f"{tif_file.stem}_sliced{tif_file.suffix}"
                    output_path = folder / output_name
                
                # Write the sliced image using skimage
                skio.imsave(str(output_path), sliced_image)
                
                print(f"Sliced {tif_file.relative_to(folder)} -> {output_path.relative_to(folder)}")
                
            except Exception as e:
                print(f"Error processing {tif_file.name}: {str(e)}")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if config file path is provided as command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        print("Usage: python slice_images.py <path_to_config_file>")
        print("Example: python slice_images.py image_slices.yaml")
        exit(1)
    
    # Load configuration from YAML file
    config = load_config(config_file)
    
    if config is None:
        print("Failed to load configuration. Exiting.")
        exit(1)
    
    # Extract home directory and folder configs from the loaded config
    home_directory = config.get('home_directory')
    folder_configs = config.get('folder_configs', {})
    
    if not home_directory:
        print("Error: 'home_directory' not specified in config file.")
        exit(1)
        
    if not folder_configs:
        print("No folder configurations found in the config file.")
        exit(1)
    
    print(f"Using configuration file: {config_file}")
    print(f"Home directory: {home_directory}")
    print(f"Processing {len(folder_configs)} folder configurations")
    
    slice_images(home_directory, folder_configs) 