# Title: Splitting regional tracer counts into left and right hemispheres
# Author: Phillip Muza
# Date: 01/05/2025

# Import necessary libraries
import os
import numpy as np
import pandas as pd
from skimage import io

from signal_lookup import count_regions, decode_region_counts, SUMMARY_FILENAME

# Label values in brainreg's registered_hemispheres.tiff
HEMISPHERE_IDS = {'left': 1, 'right': 2}


class HemisphereAnalysis:
    def __init__(self, parent_dir, registration_dir=None):
        """Split one channel's whole-brain signal lookup into left and right hemispheres.

        parent_dir is the channel directory and receives the left_hemisphere/right_hemisphere
        output folders. registration_dir holds the brainreg output and may live outside parent_dir,
        so a single registration can be shared by several signal channels. Defaults to
        parent_dir/registration_dir.
        """
        self.parent_dir = parent_dir
        self.registration_dir = registration_dir or os.path.join(parent_dir, 'registration_dir')
        self.registered_hemispheres = io.imread(os.path.join(self.registration_dir, 'registered_hemispheres.tiff'))
        # Hemisphere volumes come straight from the registration - they don't depend on the signal
        self.volumes = pd.read_csv(os.path.join(self.registration_dir, 'volumes.csv'))

    def split(self, analyser, structures_file):
        """Split a completed ImageAnalyser run (whole brain) into hemispheres.

        Each counted signal voxel is assigned the hemisphere of the atlas voxel it was counted
        against (itself, or the labelled voxel it was filled from), so region and hemisphere always
        agree and no voxel is counted twice or dropped: left + right equals the whole-brain count
        for every region, which is verified before anything is written.
        """
        if analyser.tracer_signal.shape != self.registered_hemispheres.shape:
            raise ValueError(f"registered_hemispheres.tiff is {self.registered_hemispheres.shape} but the signal "
                             f"lookup ran on {analyser.tracer_signal.shape}")

        hemisphere = self.registered_hemispheres[tuple(analyser.counted_coords.T)]
        unassigned = ~np.isin(hemisphere, list(HEMISPHERE_IDS.values()))
        if unassigned.any():
            raise ValueError(f"{unassigned.sum()} counted signal voxels sit on atlas voxels with no hemisphere label "
                             f"(values {np.unique(hemisphere[unassigned]).tolist()}); left + right would not equal "
                             f"the whole brain")

        counts = {name: count_regions(analyser.counted_regions[hemisphere == hid])
                  for name, hid in HEMISPHERE_IDS.items()}
        self._verify_against_whole(analyser.df, counts)

        summary_rows = [{'compartment': 'whole', **analyser.summary}]
        for name, hid in HEMISPHERE_IDS.items():
            hemisphere_dir = os.path.join(self.parent_dir, f'{name}_hemisphere')
            os.makedirs(hemisphere_dir, exist_ok=True)
            counts[name].to_csv(os.path.join(hemisphere_dir, 'region_counts.csv'), index=False)
            volumes = self.volumes[['structure_name', f'{name}_volume_mm3']]
            decoded = decode_region_counts(counts[name], structures_file, volumes)
            decoded.to_csv(os.path.join(hemisphere_dir, 'decoded_region_counts.csv'), index=False)

            in_hemisphere = hemisphere == hid
            n_filled = int(analyser.counted_filled[in_hemisphere].sum())
            summary_rows.append({
                'compartment': name,
                'n_signal_voxels': np.nan,           # excluded voxels have no region, hence no hemisphere
                'n_in_atlas': int(in_hemisphere.sum()) - n_filled,
                'n_filled': n_filled,
                'n_excluded_beyond_cap': np.nan,
                'max_fill_distance_um': analyser.max_fill_distance_um,
                'voxel_size_um': analyser.voxel_size_um,
            })

        # Written last: its presence marks a complete whole-brain + hemisphere run for this cap
        pd.DataFrame(summary_rows).to_csv(os.path.join(self.parent_dir, SUMMARY_FILENAME), index=False)
        return counts

    @staticmethod
    def _verify_against_whole(whole_df, counts):
        whole = whole_df.set_index('Region_ID')['Signal_Pixel_Count']
        left = counts['left'].set_index('Region_ID')['Signal_Pixel_Count']
        right = counts['right'].set_index('Region_ID')['Signal_Pixel_Count']
        combined = left.add(right, fill_value=0).reindex(whole.index.union(left.index).union(right.index), fill_value=0)
        mismatch = combined != whole.reindex(combined.index, fill_value=0)
        if mismatch.any():
            raise AssertionError(f"left + right != whole brain for {mismatch.sum()} regions: "
                                 f"{combined.index[mismatch].tolist()[:10]}")

# Example usage:
# analyser = ImageAnalyser(channel_dir, max_fill_distance_um=100)
# analyser.run_analysis("thresholded_image.tif", registered_atlas, structures_csv, volumes_csv)
# HemisphereAnalysis(channel_dir, registration_dir=registration_dir).split(analyser, structures_csv)
