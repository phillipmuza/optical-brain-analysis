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

Everything dataset-specific is the `DATASET` dictionary at the top of `config.py`. Point it at a
directory of animals, a data map and your groups:

```python
DATASET = {
    'name':                'anaesthetic',
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

Select it by editing that dictionary. One dataset at a time: to run another, edit `DATASET`, or keep
a copy of `config.py` per dataset. `config.py` also carries the anaesthetic series as
`ANAESTHETIC_DATASET`, an example of the other shape - three groups whose names are the anaesthetic
conditions, isoflurane as the control, everything else identical - so switching is a copy-paste of
that block.

Every script resolves the dictionary once when it imports `config`, then calls `config.validate()`:
- which fails with a list of problems before any work starts, and prints the cohort census (how many
animals per group) when it passes. Read that census: it is the difference between "the analysis ran"
and "the analysis ran on the animals I think it did".

Results are namespaced per dataset: `results/<name>/{masks,depth,figures,outputs,qc}`. This is not
tidiness - animal ids repeat across datasets (`an17` exists in both), so two datasets sharing a results
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
python 05_atlas_space.py                 # optional, atlas-space volumes; needs the brainglobe atlas
python 06_atlas_maps.py                  # optional, atlas-space figures; needs 05
```

Add `--qc ANIMAL` to any of the first three to check one animal before committing to the cohort.
Outputs land in `results/<name>/`.

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

### 5. `05_atlas_space.py`

Optional, and the only step that needs an atlas. Everything above measures each animal against its
own surface, which is right for **how much** tracer there is; to see **where** it is, animals have to
share a grid. Each tissue voxel is dropped into the atlas voxel its brainreg deformation field points
at, and each atlas voxel takes the mean raw intensity of the voxels that landed in it. Nothing is
normalised or thresholded in the saved volumes, so a figure can window all animals together
(absolute - valid only because acquisition settings were identical across the cohort).

Voxels are kept wherever there is tissue, not only inside the registered atlas, because the pial rim
and the cisterns - the compartment the atlas does not cover, and where the effect is expected - would
otherwise be lost. It also writes the per animal x channel detection threshold that step 06 applies:
background + `K_MAD` robust SDs over the atlas voxels that hold tissue (tissue being the registration
image above background by `TISSUE_K` robust SDs, which is what excludes atlas voxels overhanging the
sample).

Needs the brainglobe install of `config.ATLAS` (set `ATLAS_ANNOTATION` once, in config.py) and this
cohort's `registration_dir/`, so brainreg must have been run. `--qc <animal>` prints the coverage, the
tissue fraction and how much signal-bearing volume sits outside the atlas outline, and writes a
figure of the resampled planes.

Outputs, per cohort: `results/<cohort>/atlas_space/<animal>.npz` (raw intensities on the atlas grid
binned by `ATLAS_BIN`, the sample-voxel count per atlas voxel, the backgrounds, and the configuration
that produced them) and `threshold_summary.csv` next to them.

### 6. `06_atlas_maps.py`

The atlas-space figures, from step 05's outputs. Display only - no statistic in this pipeline is
computed from atlas space, and 01-04 run with no atlas at all:

- **group means** — signal across the brain with the surface/deep boundary drawn and the
  dose/reference ratio underneath (`compartment_maps_<tracer>.png`), and the integrated signal per
  coronal plane from hindbrain to olfactory bulb (`coronal_profile.png`).
- **per animal** — one row per animal, grouped by treatment, one column per coronal plane
  (`<tracer>_per_animal_<mode>.png`), in the two windowings that answer different questions.
  *absolute* puts every animal in one intensity window, so a brighter row is an animal with more
  tracer. That is only meaningful because acquisition settings were identical across the cohort and
  the groups were randomised within imaging day; it is the view that corresponds to what the second
  modality measures. *relative* divides each animal by its own background,
  `(intensity - background) / background`, which removes any residual per-animal difference in
  laser, detector and tissue autofluorescence and leaves only how the tracer is distributed within
  the brain. Voxels no sample reached are grey, not black, so a region an animal did not image is
  not read as a region with no signal. Where the two views disagree is where "more tracer" and
  "tracer arranged differently" part company.

Both windowings read the volumes raw — no threshold, no background subtraction — because that is
what 05 saves them for. The surface/deep statistics in 04 come from native space and are unaffected.

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

The registration and region-based work - regional cluster tests, the per-region intensity tables, the
notebooks that explore them - is not part of this repository. Steps 05 and 06 were ported from it so
that this pipeline depends on nothing outside itself: what was left behind is the analysis this
pipeline exists to avoid, measuring signal per atlas region, along with its own cohort constants and
cache. `git log` has it if a specific number needs tracing back.
