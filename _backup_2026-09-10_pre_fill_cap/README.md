# Brain Image Analysis Pipeline

This pipeline provides tools for analyzing brain images, including tracer segmentation, brain registration, signal lookup, and hemisphere-based analysis.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd tracer_uptake_analysis
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Install Brainreg (required for brain registration):
```bash
pip install brainreg
```

4. Download the required brain atlas:
```bash
# List available atlases
brainglobe list

# Install the Perens LSFM mouse brain atlas (20 micron resolution)
brainglobe install perens_lsfm_mouse_20um
```

For more information about available atlases and installation options, visit the [BrainGlobe Atlas API documentation](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html).

## Orientation Correction

LSM acquisitions are transverse, in one of two directions that varies by animal (dorsal→ventral
or ventral→dorsal), but registration needs a coronal stack. **This runs automatically as step 0**,
before downsampling — you no longer need to run `check_orientation.py` / `image_transformations.py`
/ `check_transform.py` yourself in sequence.

### Why a human still has to look

Which direction an animal was acquired in cannot be told apart algorithmically — both directions
produce the same small→big→small cross-sectional profile along the stack, since it's the same
physical brain either way round. This is the one manual step the pipeline can't remove.

### How it works

1. On each run, `main.py` checks every animal directory for a scored direction in
   `<parent_dir>/orientation_check/orientations.csv`.
2. If **any** animal is missing a score, it (re)writes that animal's contact sheet into
   `orientation_check/` and **stops without processing anything** — including animals that were
   already scored — so you get one QC pass over the whole batch rather than surprises trickling
   in animal by animal.
3. Open the sheets, fill in the `direction` column with `dorsal_to_ventral` or
   `ventral_to_dorsal`, and re-run the exact same command.
4. Once every animal is scored, each one is transformed automatically into
   `<animal_dir>/transformed/` (raw images are never modified, matching `downsampled/`), a single
   cross-animal `orientation_check/transform_check.png` grid is written so a wrong call is easy
   to spot before the expensive registration/segmentation steps run on it, and the pipeline
   continues straight through downsampling, registration and analysis in the same run.

Re-running after that is free: already-transformed animals and already-scored rows are reused,
so a batch with a few new animals only regenerates sheets for the new ones.

If your images are already coronal (not from this transverse LSM acquisition), skip this
entirely with `--source_orientation coronal`.

## Image Downsampling

Downsampling is **built into the pipeline** and runs automatically as step 0 — you no longer need to run `downsample_images.py` yourself.

### Why it is mandatory
Brainreg emits `registered_atlas.tiff` on the voxel grid of the image it was given, and signal lookup requires the segmented signal and the registered atlas to have identical shapes. Every image therefore has to be resampled to the atlas resolution (20x20x20 µm for `perens_lsfm_mouse_20um`) before registration and segmentation. Downsampling also cuts memory use and runtime substantially.

Raw images are never modified — 20 µm copies are written into a `downsampled/` subfolder, and existing ones are reused on re-runs.

Set your microscope's voxel size with `--source_voxel_size` (z, x, y in microns; default `20 6.55 6.55`). `downsample_images.py` remains usable as a standalone script if you want to downsample separately.

## Data Organization and File Naming

The pipeline recursively searches for **animal directories** — any directory containing at least one of the signal images named by `--signal_imgs`. Filenames are matched case-insensitively, so `FITC.tif` and `fitc.tif` both work.

### Required File Names
- Signal images: default `FITC.tif` and `TxR.tif`, customisable via `--signal_imgs`. All matching channels in a directory are processed, each into its own output folder.
- Registration image: **optional**. If no `registration.tif` is found, the pipeline copies the downsampled channel named by `--registration_source` (default `FITC.tif`) to `downsampled/registration.tif`. Supply your own `registration.tif` next to the raw images to override this.

### Directory Structure

Input, before running:
```
parent_directory/
└── ATX_5781_June26/Lights_OFF/LSM_analysis/an9/
    ├── FITC.tif
    └── TxR.tif
```

Output, after running:
```
an9/
├── FITC.tif  TxR.tif                    # raw images, untouched
├── transformed/
│   └── FITC.tif  TxR.tif                # coronal copies (orientation correction, step 0)
├── downsampled/
│   ├── FITC.tif  TxR.tif                # 20 µm copies, resampled from transformed/
│   └── registration.tif                 # copy of downsampled FITC
├── registration_dir/                    # ONE brainreg run, shared by all channels
│   ├── brain_mask.tif  preprocessed.tiff
│   ├── registered_atlas.tiff  registered_hemispheres.tiff
│   └── volumes.csv
├── FITC/
│   ├── brain_mask.tif  unsharp_image.tif  thresholded_image.tif
│   ├── tracer_percentage_area.csv  region_counts.csv  decoded_region_counts.csv
│   ├── left_hemisphere/   registration_mask.tiff  signal_mask.tiff  volumes.csv
│   └── right_hemisphere/                            decoded_region_counts.csv
└── TxR/                                 # same structure as FITC
```

Registration runs **once per animal** because both channels come from the same brain — the shared `registration_dir/` sits alongside the per-channel folders rather than inside them.

The script will automatically:
1. Search through all subdirectories, ignoring the `transformed/`, `orientation_check/`, `downsampled/`, `registration_dir/` and per-channel folders it creates
2. Transform any channel not already present in `transformed/`, once every animal has a scored orientation direction (see [Orientation Correction](#orientation-correction))
3. Downsample any channel not already present in `downsampled/`
4. Skip registration where `registration_dir/registered_atlas.tiff` already exists
5. Skip any channel already containing `decoded_region_counts.csv`

A top-level `orientation_check/` folder also appears next to your animal directories, holding the orientation contact sheets, `orientations.csv`, and `transform_check.png`.

## Usage

Run the main analysis script with default settings:
```bash
python main.py --parent_dir "path/to/your/parent/directory"
```

### Command Line Arguments

Required arguments:
- `--parent_dir`: Path to the parent directory containing your data

Optional arguments:
- `--signal_imgs`: One or more signal image names to look for (default: `FITC.tif TxR.tif`)
- `--registration_img`: Filename of the shared registration image inside `downsampled/` (default: `registration.tif`)
- `--registration_source`: Signal channel copied to build the registration image when none is supplied (default: `FITC.tif`)
- `--source_voxel_size`: Voxel size of the raw images in microns, z x y (default: `20 6.55 6.55`)
- `--target_voxel_size`: Voxel size to downsample to, matching the atlas (default: `20 20 20`)
- `--orientation`: Image orientation passed to brainreg (default: `asr`)
- `--source_orientation`: `transverse` (default) runs orientation scoring and the transverse→coronal transform as step 0; `coronal` skips it entirely for data that's already correctly oriented

Examples:

1. Typical two-channel run:
```bash
python main.py --parent_dir "D:/awake_tracer_infusions/ATX_5781_June26" \
               --signal_imgs FITC.tif TxR.tif \
               --registration_source FITC.tif \
               --source_voxel_size 20 6.55 6.55
```

2. Different acquisition z-step:
```bash
python main.py --parent_dir "path/to/data" --source_voxel_size 15 6.55 6.55
```

3. Register on the TxR channel instead:
```bash
python main.py --parent_dir "path/to/data" --registration_source TxR.tif
```

### Workflow

For each animal directory, the script performs the following steps in sequence:

0. **Orientation Correction** (see [Orientation Correction](#orientation-correction) above)
   - Skipped with `--source_orientation coronal`
   - Otherwise gates the whole run until every animal has a scored direction, then transforms into `transformed/`

1. **Downsampling**
   - Resamples every raw signal image from `--source_voxel_size` to `--target_voxel_size`
   - Outputs in `downsampled/`, reusing anything already present

2. **Registration Image Preparation**
   - Uses `downsampled/registration.tif` if present, otherwise downsamples a supplied raw `registration.tif`, otherwise copies the downsampled `--registration_source` channel

3. **Brain Registration** (once per animal, shared by all channels)
   - Preprocessing generates its own `brain_mask.tif` from the registration image and reduces bright surface signal
   - Registers to the Perens LSFM mouse brain atlas (20 micron resolution)
   - Outputs in `registration_dir/`: `registered_atlas.tiff`, `registered_hemispheres.tiff`, `volumes.csv`
   - Skipped entirely if `registered_atlas.tiff` already exists

4. **Per-Channel Analysis** (repeated for each signal image)
   - **Tracer Segmentation** → `<channel>/brain_mask.tif`, `unsharp_image.tif`, `thresholded_image.tif`, `tracer_percentage_area.csv`
   - **Signal Lookup** against the shared `registration_dir/` → `<channel>/decoded_region_counts.csv`
   - **Hemisphere Analysis** (always runs) → `<channel>/left_hemisphere/` and `<channel>/right_hemisphere/`, each with `registration_mask.tiff`, `signal_mask.tiff`, `volumes.csv` and its own `decoded_region_counts.csv`
   - Whole-brain and per-hemisphere counts are always produced together; drop the hemisphere results downstream if a given analysis doesn't need them
   - Both steps are skipped per channel where their `decoded_region_counts.csv` already exists, so re-running a finished dataset only fills in the hemispheres it is missing

## Dependencies

- `brainreg`: Brain registration tool
- `scikit-image`: Image processing
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `plumbum`: Command execution
- `numpy`: Numerical operations

## Troubleshooting

### Common Issues

1. **Missing Files**
   - Ensure your signal images match the names specified in `--signal_imgs` (matching is case-insensitive)
   - A registration image is not required; if absent it is built from `--registration_source`, which must name one of your signal images

2. **Shape Mismatch Errors**
   - "Signal and registration images must have the same shape" means the downsampled signal and the registered atlas disagree
   - Almost always caused by a wrong `--source_voxel_size`, or by a `downsampled/` folder left over from a run with different resolution settings
   - Fix by deleting `downsampled/` and `registration_dir/` for that animal and re-running with the correct voxel size

3. **Brain Registration Issues**
   - Verify brainreg is properly installed
   - Ensure the Perens LSFM mouse brain atlas is installed (`brainglobe install perens_lsfm_mouse_20um`)
   - Check the input image orientation matches `--orientation`
   - Ensure sufficient disk space for registration output
   - To force a re-registration, delete `registration_dir/` for that animal

4. **Hemisphere Analysis Issues**
   - Hemisphere analysis requires successful completion of brain registration
   - Check if `registered_hemispheres.tiff` exists in registration_dir
   - Missing inputs are logged and that channel's hemisphere step is skipped; the whole-brain results for the channel are unaffected
   - To redo it for a channel, delete that channel's `left_hemisphere/` and `right_hemisphere/` folders and re-run

5. **Run exits immediately with "still need orientation scoring"**
   - Expected the first time you run on new raw data — open the sheets in `orientation_check/`, fill in the `direction` column of `orientations.csv`, and re-run the same command
   - If a sheet failed to write (check the log for "Failed to write orientation sheet"), that animal counts as unscored too and blocks the run until it's fixed
   - Already coronal data doesn't need this — pass `--source_orientation coronal`

6. **Wrong orientation call**
   - Check `orientation_check/transform_check.png` after a successful run — an animal scored the wrong way round shows its row running backwards against the others
   - Fix the `direction` in `orientations.csv`, delete that animal's `transformed/` and `downsampled/` folders, and re-run

### Getting Help

If you encounter any issues:
1. Check the error messages in the console output
2. Verify all dependencies are correctly installed
3. Ensure your input files are in the correct format and named correctly
4. Check if you have sufficient disk space
5. Consider downsampling your images if processing is slow
6. Visit the [BrainGlobe documentation](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html) for more information about atlases and registration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included in the repository. 