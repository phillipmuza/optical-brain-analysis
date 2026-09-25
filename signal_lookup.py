# Title: Assigning segmented tracer signal to registered atlas regions
# Author: Phillip Muza
# Date: 23.08.24

import os
import logging
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy import ndimage
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = 'signal_assignment_summary.csv'


def atlas_boundary_coords(atlas):
    """Coordinates of labelled atlas voxels that touch (26-connected) an unlabelled voxel.

    The nearest labelled voxel to any unlabelled point is always one of these, so a KD-tree built
    on them alone gives the same nearest distances as one built on every labelled voxel, at a
    fraction of the size.
    """
    boundary = (atlas > 0) & ndimage.binary_dilation(atlas == 0, structure=np.ones((3, 3, 3), dtype=bool))
    return np.argwhere(boundary)


def count_regions(region_ids):
    """Count signal voxels per atlas region, largest first. np.unique rather than np.bincount:
    atlas IDs run to ~6e8, so bincount would allocate a multi-GB array to count a few hundred regions."""
    ids, counts = np.unique(region_ids, return_counts=True)
    df = pd.DataFrame({'Region_ID': ids.astype(np.int64), 'Signal_Pixel_Count': counts})
    df = df[df['Region_ID'] > 0]
    return df.sort_values('Signal_Pixel_Count', ascending=False).reset_index(drop=True)


def decode_region_counts(counts_df, structures_file, volumes):
    """Attach structure names/paths and registration volumes to a Region_ID count table."""
    structures = pd.read_csv(structures_file)
    unmatched = counts_df.loc[~counts_df['Region_ID'].isin(structures['id'])]
    if not unmatched.empty:
        logger.warning(f"{unmatched['Signal_Pixel_Count'].sum()} signal voxels carry atlas IDs missing from "
                       f"{structures_file} and will not appear in decoded_region_counts.csv: "
                       f"{unmatched['Region_ID'].tolist()}")
    decoded = pd.merge(structures, counts_df, left_on='id', right_on='Region_ID', how='left')
    # brainreg's volumes.csv repeats id/structure_id_path from the atlas; merging it in as-is
    # makes pandas suffix the collisions (id_x/id_y) so the decoded table has no plain 'id'
    # column. Only the volume columns are wanted here.
    redundant = [c for c in volumes.columns
                 if c in decoded.columns and c != 'structure_name']
    if redundant:
        volumes = volumes.drop(columns=redundant)
    return pd.merge(decoded, volumes, left_on='name', right_on='structure_name', how='left')


class ImageAnalyser:
    def __init__(self, base_dir, max_fill_distance_um=None, voxel_size_um=20.0):
        """max_fill_distance_um caps how far signal lying outside the atlas may be from the nearest
        labelled voxel and still be assigned to it; None means no cap (the original behaviour).
        voxel_size_um is the isotropic voxel size of the signal/atlas images."""
        self.base_dir = base_dir
        self.max_fill_distance_um = np.inf if max_fill_distance_um is None else float(max_fill_distance_um)
        self.voxel_size_um = float(voxel_size_um)
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
        # The atlas voxel each signal voxel is counted against: itself unless filled from a neighbour
        self.assigned_coords = self.non_zero_coords.copy()
        self.filled = np.zeros(len(self.non_zero_coords), dtype=bool)

    # Some signal voxels land on atlas label 0 (surface signal, small registration mismatches, or
    # segmentation artefacts outside the brain). Those within max_fill_distance_um of a labelled
    # voxel take its label; anything further stays 0 and is reported, never counted in a region.
    def fill_zero_regions(self):
        zero_indices = np.flatnonzero(self.corresponding_regions == 0)
        self.n_in_atlas = len(self.corresponding_regions) - len(zero_indices)
        if len(zero_indices) == 0:
            self.n_excluded = 0
            return

        boundary = atlas_boundary_coords(self.registration)
        tree = cKDTree(boundary)
        max_distance_vox = self.max_fill_distance_um / self.voxel_size_um
        # One vectorised query; points with no labelled voxel within the cap come back as inf.
        # The small margin makes the cap inclusive (a voxel exactly at the cap is filled).
        distances, nearest = tree.query(self.non_zero_coords[zero_indices],
                                        distance_upper_bound=max_distance_vox * (1 + 1e-9) + 1e-9,
                                        workers=-1)
        within = np.isfinite(distances)

        filled_indices = zero_indices[within]
        nearest_coords = boundary[nearest[within]]
        self.corresponding_regions[filled_indices] = self.registration[tuple(nearest_coords.T)]
        self.assigned_coords[filled_indices] = nearest_coords
        self.filled[filled_indices] = True
        self.n_excluded = int((~within).sum())

    # Generate a dataframe describing the amount of tracer signal in a given brain region
    def count_regions(self):
        counted = self.corresponding_regions > 0
        self.counted_regions = self.corresponding_regions[counted]
        self.counted_coords = self.assigned_coords[counted]
        self.counted_filled = self.filled[counted]
        self.df = count_regions(self.counted_regions)

        self.summary = {
            'n_signal_voxels': len(self.corresponding_regions),
            'n_in_atlas': self.n_in_atlas,
            'n_filled': int(self.filled.sum()),
            'n_excluded_beyond_cap': self.n_excluded,
            'max_fill_distance_um': self.max_fill_distance_um,
            'voxel_size_um': self.voxel_size_um,
        }
        n = max(self.summary['n_signal_voxels'], 1)
        logger.info(f"Signal voxels: {self.summary['n_signal_voxels']:,} | in atlas {self.n_in_atlas:,} "
                    f"({self.n_in_atlas / n:.1%}) | filled within {self.max_fill_distance_um:g} um "
                    f"{self.summary['n_filled']:,} ({self.summary['n_filled'] / n:.1%}) | excluded "
                    f"{self.n_excluded:,} ({self.n_excluded / n:.1%})")

    # Save the dataframe
    def save_results(self, filename='region_counts.csv'):
        self.df.to_csv(filename, index=False)

    # Region_counts is coded, here we decode the regions in region counts and
        # combine the volumes file generated during registration
    def merge_data(self, structures_file, volumes_file):
        volumes = pd.read_csv(volumes_file)
        self.decoded_df = decode_region_counts(self.df, structures_file, volumes)
        self.decoded_df.to_csv('decoded_region_counts.csv', index=False)

    # Run the analysis
    def run_analysis(self, tracer_file, registration_file, structures_file, volumes_file):
        self.load_images(tracer_file, registration_file)
        self.process_images()
        self.fill_zero_regions()
        self.count_regions()
        self.save_results()
        self.merge_data(structures_file, volumes_file)

# Example usage:
# analyser = ImageAnalyser("path/to/your/parent/directory", max_fill_distance_um=100)
# analyser.run_analysis(
#     "thresholded_image.tif",
#     "registration/registered_atlas.tiff",
#     "path/to/structures.csv",
#     "registration/volumes.csv"
# )
# print(analyser.decoded_df)
