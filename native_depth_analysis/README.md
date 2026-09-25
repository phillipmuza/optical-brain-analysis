# Native depth analysis

Measures **how much CSF tracer reaches the brain surface**, from cleared light-sheet volumes, without
an atlas and without any registration.

Point the pipeline at a dataset by selecting a **cohort** - a directory of animals, a data map and a
list of treatment groups - and nothing else in the code changes. See "Configuring it".

Each animal is measured against **its own brain surface**, in its own image space. Signal is
integrated in shells at a known depth below that surface, dorsal and ventral halves separately, and
compared between treatment groups in absolute units.

## Why it exists

The conventional pipeline registers each brain to an atlas, assigns tracer signal to anatomical
regions, and expresses everything as a proportion (% of a region's volume, share of the animal's
total). On the NS24122 cohort that approach found nothing, while an independent ex vivo IVIS
measurement showed a clear increase in surface fluorescence. The two disagreed for structural
reasons, not biological ones:

- **At 30 minutes the tracer is mostly a thin rim on the pial surface**, not spread through the
  parenchyma. Region-based metrics split that rim across whichever structures it abuts.
- **The atlas does not cover the subarachnoid space**, so a large share of the signal falls outside
  it — 20% in controls, 38% in treated animals in our data.
- **Registration is driven by the tracer itself** when the registration channel is a copy of a
  tracer channel, so the atlas fits treated brains less snugly. That makes the *accounting* of
  signal depend on the treatment group.
- **Every metric was relative**, so a change in how much tracer arrived was normalised away.

This pipeline removes all four problems by never leaving the animal's own image space.

When run on the NS24122 cohort it recovered the IVIS effect: FITC surface signal (0–0.5 mm) was
**3.3× higher dorsally and 2.8× higher ventrally at 30 mg** (ANOVA p = 0.056 / 0.064, Tukey vs
vehicle p = 0.048 / 0.055, Hedges' g = 1.3 / 1.2), against 1.9× and 1.6× measured by IVIS on the same
animals, and the two modalities correlated per animal (Spearman ρ = 0.48–0.84 across tracers and
surfaces, all p ≤ 0.044). The deep compartment showed nothing, so the effect is at the surface.
That is 18 animals (vehicle 7, 10 mg 5, 30 mg 6) after five exclusions.

## What you need

Python with: `numpy`, `pandas`, `scipy`, `scikit-image`, `tifffile`, `matplotlib`, `statsmodels`,
`pingouin`. The original runs used the `2D_cell_analysis` conda environment (Python 3.9).

Roughly 8 GB of RAM free, and enough disk for one bit-packed mask per animal (~25 MB each).

## Input data

One folder per animal under `DATA_DIR`, each containing:

```
<animal>/downsampled/<REFERENCE_IMAGE>     e.g. registration.tif  - the image the mask is built from
<animal>/downsampled/<channel>.tif         e.g. fitc.tif, txr.tif - one per tracer channel
```

All images must be the **same grid, same isotropic voxel size**, and — critically — **acquired with
identical settings across every animal in the cohort**, because the comparison is of absolute
intensity. Randomising treatment groups across imaging sessions is strongly advised; if group A was
imaged on a different day from group B with a different laser setting, this pipeline will happily
report that difference as biology.

You also need a data map (CSV) with one row per animal, giving an animal id column and a treatment
column.

A second modality (here IVIS) is optional. If you have one, step 4 will cross-validate against it per
animal; if not, set `second_modality_csv: None` for that cohort and that section is skipped.

## Configuring it

Everything dataset-specific is the `COHORTS` dictionary in `config.py`. A cohort is:

```python
'anaesthetic': {
    'data_dir':            r'D:\anaesthetic_experiments\cleared_brains_new_analysis',
    'data_map':            r'...\data_map.csv',
    'second_modality_csv': r'...\ivis.csv',          # or None; step 04 then skips the cross-check
    'atlas_annotation':    r'...\annotation.tiff',   # step 05 only
    'atlas_space_dir':     r'...\atlas_space',       # step 05 only
    'threshold_summary':   r'...\threshold_summary.csv',   # step 05 only
    'mask_channel_is_tracer': 'FITC',                # see below; '' if the reference is its own channel
    'groups': [
        {'name': 'Isoflurane',   'label': 'Iso', 'color': '#7f7f7f'},
        {'name': 'Medetomidine', 'label': 'Med', 'color': '#4C72B0'},
        {'name': 'KX',           'label': 'K/X', 'color': '#C1666B'},
    ],
    'reference_group':  'Isoflurane',
    'exclude_animals':  [],
    'example_animal':   'an17',
}
```

`groups[].name` must match the `treatment` column of the data map **exactly** - a mismatch is the one
mistake that would otherwise reach the statistics, so `validate()` stops on it rather than plotting
NaN bars. `reference_group` is what every contrast is tested against, and it differs between cohorts
(Vehicle for the drug series, Isoflurane for the anaesthetic one). The label and colour are only for
figures.

Select the cohort on any script:

```
python 03_depth_profiles.py --cohort anaesthetic
NATIVE_DEPTH_COHORT=anaesthetic python 03_depth_profiles.py     # same thing
```

Without either, `DEFAULT_COHORT` is used. Each script calls `config.select()`, then `config.validate()`
- which fails with a list of problems before any work starts, and prints the cohort census (how many
animals per group) when it passes. Read that census: it is the difference between "the analysis ran"
and "the analysis ran on the animals I think it did".

Results are namespaced per cohort: `results/<cohort>/{masks,depth,figures,outputs,qc}`. This is not
tidiness - animal ids repeat across cohorts (`an17` exists in both), so two cohorts sharing a results
directory would half-overwrite each other and produce a plausible figure mixing them. Set
`NATIVE_DEPTH_RESULTS` to one exact directory to override, which is how you point step 04 at results
computed earlier.

### What this assumes

These are constants, not per-cohort settings, because every cohort so far has shared them. A cohort
that breaks one needs a new constant and a look at the step that uses it - not a new special case:

- 20 um isotropic downsampled volumes, and the same acquisition settings across the cohort. The
  comparison is of **absolute** intensity, so a cohort imaged on two different days with different
  laser settings will report that difference as biology. Randomise groups across imaging days.
- `brainreg --orientation asr`, so axis 1 runs dorsal -> ventral. Another orientation swaps the
  dorsal and ventral halves silently.
- `registration.tif` as the mask-building channel, one `fitc.tif`/`txr.tif` per tracer, one folder
  per animal. `REFERENCE_IMAGE` and `CHANNELS` are the only handles on that layout.
- The Perens LSFM mouse atlas (`perens_lsfm_mouse_20um`), 20 um, so its distances and the images'
  distances are in the same units. `ATLAS_BIN` is the display binning (2 -> the 40 um grid).

The analysis parameters (`K_MAD`, `SURFACE_MM`, closing radii, depth bins) also have defaults
documented in the file; leave them unless you have a reason. `SURFACE_MM` is the single source for
the surface/deep boundary - the band names (`surface_0_500um`, `deep_500um_plus`) are derived from it,
so changing it changes the statistics and the labels together.

`mask_channel_is_tracer` records whether the reference image is a copy of a tracer channel. In every
cohort so far it is (a copy of FITC), which means the mask follows the tracer's own surface rim and
"depth 0" is not independent of the signal - see "Things to keep in mind when reading the results".

## Running it

In order. Each script is independent and re-runnable; steps 2 and 3 skip nothing, so delete outputs
if you want a clean run.

```
python 01_check_wraparound.py            # ~10 min   data integrity, read this before trusting anything
python 02_build_masks.py --qc an4 an7    # ~3 min/animal   look at the figures before committing
python 02_build_masks.py                 # ~45 min for 20 animals
python 03_depth_profiles.py --qc an4     # ~5 min    check the envelope and the profile shape
python 03_depth_profiles.py              # ~90 min for 20 animals
python 04_analysis.py                    # seconds   statistics and figures
python 05_atlas_maps.py                  # optional, atlas-space figures; needs the stage-05 inputs
```

Add `--cohort NAME` to any of them (or set `NATIVE_DEPTH_COHORT`) to run a dataset other than the
default. Outputs land in `results/<cohort>/`.

### 1. `01_check_wraparound.py`

Checks for 16-bit wraparound: stacks saved as **signed** int16 store any voxel brighter than 32,767
as a large negative number. Those are the brightest voxels — the tracer — and being negative they
fall below every threshold and vanish from the analysis, biasing against the animals with most
signal. It also breaks any rule that reads a bin position from a histogram.

In our cohort 17 of 23 animals were affected, but only a few hundred voxels each (< 0.1% of signal),
so we proceeded with thresholds estimated on non-negative voxels only. **Check the numbers for your
data before deciding.** If a stack has lost a meaningful fraction, re-export it as unsigned 16-bit,
or re-downsample from raw with the wrap undone (`value += 65536 where value < 0`) — it cannot be
repaired after averaging.

Outputs: `qc/int16_wraparound.csv`, `qc/int16_wraparound.png`.

### 2. `02_build_masks.py`

Builds the brain mask in native space: Otsu threshold on the reference image, 3D hole filling,
largest connected component, small closing. Not convexified, so it follows the real surface.

**Why not use the pipeline mask:** `mask_generation.py` in the older tracer pipeline applies a
per-slice **convex hull**, which cannot follow the concavities of a brain and came out 30–90% larger
than the brain (785 vs 593 mm³ in an4; 1119 vs 737 mm³ in an16).

`--qc <animals>` writes a comparison figure per animal — sections with every candidate mask outlined,
the intensity histogram with the threshold candidates, and a metrics table. **Look at these before
running the whole cohort.** What you want to see: a bimodal histogram with the Otsu line in the
valley, and a mask outline that hugs the tissue including its concavities.

Outputs: `masks/<animal>.npz` (bit-packed), `mask_summary.csv`, `qc/<animal>_mask_comparison.png`.

### 3. `03_depth_profiles.py`

The measurement. For each animal it builds an **outer envelope** from the mask (a 100 µm closing plus
hole filling, so ventricles and the midline fissure do not count as surface), computes the **signed
distance** from that envelope (negative outside, positive inside), splits the brain into dorsal and
ventral halves by each column's own mid-height, and bins the tracer signal by depth.

Signal is raw intensity above that animal's own background, counted where it exceeds that animal's
fixed threshold (background + `K_MAD` robust SDs). Both the thresholded sum and the all-voxel sum are
stored; the thresholded one is primary.

`--qc <animal>` prints the envelope volume, how much signal sits outside the surface, and writes a
figure of the signed-depth map plus the depth profile.

Outputs: `depth/<animal>.npz`, `qc/<animal>_native_depth.png`.

### 4. `04_analysis.py`

Statistics and figures:

- **absolute** — integrated signal in the surface (0–0.5 mm) and deep (> 0.5 mm) compartments,
  compared between groups. One-way Type II ANOVA of treatment per tracer × region, then Tukey
  post-hoc (statsmodels + pingouin, reporting Hedges' g).
- **relative** — the fraction of each animal's signal that is deep, i.e. the penetration question.
- **agreement** — per-animal correlation against the second modality, if configured.

Outputs in `outputs/`: `native_depth_per_animal.csv`, `native_depth_anova.csv`,
`native_depth_posthoc_tukey.csv`, and the correlation tables. Figures in `figures/`:
`native_depth_profiles.png`, `native_depth_headline.png`, `native_vs_ivis.png`,
`native_vs_ivis_average.png`.

## Things to keep in mind when reading the results

- **The mask is built from the reference image, which in our dataset was a copy of a tracer channel.**
  A brighter surface rim therefore nudges the mask boundary outward, so "depth 0" is not perfectly
  independent of the tracer. Integrating across the boundary absorbs this, but do not report "signal
  outside the surface" as an independent quantity. With a dedicated autofluorescence channel the
  problem disappears — worth acquiring one.
- **The mask is a threshold plus morphology, not a validated segmentation.** It will include dura,
  debris or anything bright stuck to the surface, all of which land in the outermost shell, which is
  exactly where the effect is expected. Treat the 0–100 µm shell with particular suspicion.
- **Multiplicity.** Two tracers × two regions × (number of groups − 1) comparisons. Our headline
  p-values were 0.03–0.05 uncorrected and would not survive Bonferroni across the whole family; the
  case rested on the direction being predicted in advance by an independent modality, and on the two
  modalities agreeing animal by animal.
- **Batch effects show up here.** Absolute intensity is sensitive to acquisition conditions. In our
  cohort one dose group was run months after the others and had a visibly different background, and
  an ANOVA can come out significant because of that group alone — check the post-hoc to see which
  pair is driving it, rather than reading the ANOVA p on its own.

## What is deliberately not here

The registration and region-based work - registration QC, regional cluster tests - lives in
`../NS24122_intensity/` and `../native_space_analysis/` alongside notes on the registration problems
(`registration_and_mask_issues.md`) and the reasoning behind this design
(`native_space_surface_analysis.md`).

`05_atlas_maps.py` is the one atlas-dependent stage here, and only for **display** alignment. Its
input, the per-animal atlas-space resampling, is still produced outside this repository
(`../NS24122_intensity/atlas_space_images.py`) - so stage 05 reproduces for a cohort whose
atlas-space volumes already exist, and not otherwise. Nothing else in the pipeline depends on it:
01-04 run with no atlas at all.
