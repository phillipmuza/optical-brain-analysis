# Registration and brain-mask problems in the tracer pipeline

**Written 15/09/2026, from the NS24122 cohort analysis.** Pipeline under discussion:
`Z:\scripts\updated_tracer_uptake_analysis` (`mask_generation.py`, `tracer_segmentation.py`,
`brain_registration.py`, `signal_lookup.py`).

## Summary

Three separate defects interact, and together they make the *accounting* of tracer signal depend on
the treatment group rather than only on the biology:

1. The image used for registration is a copy of a tracer channel, so registration is driven by an
   image dominated by the bright tracer rim at the pial surface.
2. The brain mask is built with a per-slice **convex hull**, so it cannot follow concavities and is
   far larger than the brain.
3. The registered atlas both **overhangs** the sample (dark non-tissue voxels inside the atlas) and
   **under-covers** it (real tissue outside the atlas) - and the under-covered rim is exactly where
   most of the tracer signal sits.

The third is the one that changes conclusions. It is not a small effect: in the NS24122 cohort about
5% of tissue lies outside the registered atlas, and that 5% carries up to 38% of an animal's tracer
signal.

## Evidence

### The registration image contains the thing we are measuring

`main.prepare_registration_image()` copies a downsampled signal channel when no dedicated
registration image exists. In this cohort `anX/downsampled/registration.tif` is a **copy of the FITC
channel** - confirmed by identical background statistics in two animals (an4: 860 ± 197 in both;
an16: 752 in both).

CSF tracer at 30 min sits overwhelmingly on the pial surface (see
`figures/FITC_per_animal_absolute.png`): a bright rim around a comparatively dark parenchyma. The
atlas template (`perens_lsfm_mouse_20um`) is an autofluorescence average with no such rim. Registering
an image whose strongest feature is absent from the target is a poor conditioning of the problem, and
it plausibly pulls the atlas boundary outward onto the rim - the effect Phillip suspected.

### The fit is measurably worse in treated animals

From `registration_check/registration_qc.csv`, fraction of tissue left outside the registered atlas:

| group | tissue outside atlas | Dice |
|---|---|---|
| Vehicle | 2.3% (0.9-3.6%) | 0.967 |
| NS24122 10 mg | 5.6% (1.3-15.8%) | 0.957 |
| NS24122 30 mg | 4.6% (1.1-6.4%) | 0.962 |

30 mg vs Vehicle, Welch p = 0.048. The share of signal the pipeline has to "fill" from outside the
atlas tracks it closely (Spearman rho = 0.78 across animals, p = 4.9e-05): Vehicle 17.7%, 10 mg
28.7%, 30 mg 36.0% (30 mg vs Vehicle p = 0.035).

### A few percent of tissue carries a third of the signal

Integrated above-threshold tracer signal, split by whether it falls inside or outside the registered
atlas (atlas-space volumes, fixed per-animal threshold; `ivis_vs_lightsheet.py` and the ad-hoc
compartment analysis of 15/09/2026):

| FITC, integrated signal | Vehicle | 30 mg | ratio | Welch p |
|---|---|---|---|---|
| inside atlas, dorsal | 1.5e7 | 3.0e7 | 2.0x | 0.17 |
| inside atlas, ventral | 2.6e7 | 3.2e7 | 1.2x | 0.68 |
| **outside atlas, dorsal** | 2.2e6 | 1.9e7 | **8.9x** | **0.024** |
| **outside atlas, ventral** | 5.2e6 | 3.4e7 | **6.4x** | 0.051 |

TxR behaves the same way (outside-atlas 3.5x dorsal, 2.8x ventral, both p ~ 0.11).

Fraction of each animal's above-threshold signal lying outside the registered atlas:

| group | FITC | TxR |
|---|---|---|
| Vehicle | 20% | 14% |
| 10 mg | 22% | 18% |
| 30 mg | **38%** | **27%** |

### Why that is dangerous

`signal_lookup` assigns signal within 200 µm outside the atlas to the nearest labelled region and
discards anything further. So a group difference in how snugly the atlas fits becomes a group
difference in *where signal is attributed*, and in *how much is discarded*, without any difference in
the underlying brain. Every downstream metric inherits it.

Concretely, this is a live alternative explanation for the main regional result of
`../NS24122_analysis/NS24122_spatial_autocorrelation_analysis.ipynb`: NS24122 animals appeared to
have **less** tracer in ventral midbrain, pons and hypothalamus (TxR 30 mg, p_FWER = 0.001). Those
regions line the basal cisterns, which is where the ventral rim lands. If more of that rim falls
outside the atlas in treated animals, those regions lose share for a purely technical reason. That
finding should be treated as unresolved until the registration issue is settled.

### The mask itself

`mask_generation.MaskGenerator` blurs (Gaussian sigma = 2), thresholds at the background histogram
peak, then applies **per-slice `convex_hull_image`** plus hole filling. A convex hull cannot follow
the concavities of a brain - the cortex/cerebellum gap, the midline fissure, the ventral curvature -
so the mask is systematically inflated. The ATX-7851 penetration work measured it at 1023 mm³ against
586 mm³ for the registered atlas, which is why that analysis abandoned `brain_mask.tif` and derived
its own envelope from the atlas instead.

For comparison, `fixed_threshold_intensity.tissue_mask()` (written for this analysis) is a single
global threshold on the registration image at `median - 2 x robust SD`, with no hull and no per-slice
processing. It gives 94.7-98.2% tissue inside the atlas with no group difference (p = 0.44 / 0.10),
and the atlas-space figures show it tracks the brain outline closely. It is **not** validated as a
brain mask: being a plain threshold, it will include dura, debris or anything bright stuck to the
surface, and it has never been checked against manual segmentation.

### A fourth defect, found 15/09/2026: 16-bit wraparound breaks the mask threshold

The stacks are stored as **signed** int16, so voxels brighter than 32,767 wrap to large negative
values. This is in the raw data, not the pipeline. It affects 17 of 23 animals, but only a few
hundred voxels each (an12 TxR is the outlier at 54,707).

It matters here because `_calculate_threshold()` takes the mode of the first 50 bins of a 256-bin
histogram. With the range stretched from -26,281 to +30,141 those bins are meaningless and the rule
returns a **negative threshold** - for an36, -13,965 - so the mask thresholds the whole image and
the convex hull alone defines the brain. Estimating the same rule on non-negative voxels gives 220.

Full write-up: `../native_space_analysis/data_integrity_int16_wraparound.md`.

### A separate but related defect: atlas overhang

The registered atlas also covers voxels that hold no tissue (dark, ~600 intensity units below
background). They are only ~5% of atlas voxels but, left in an unclipped regional sum, they removed
about two thirds of the apparent FITC total (an4: 1.08e9 over the whole atlas vs 3.24e9 over tissue
only). All intensity work here is therefore restricted to atlas ∩ tissue.

## What to fix, in order of value

1. **Register on an image that does not contain the tracer.** Best: acquire a dedicated
   autofluorescence channel (e.g. 488 nm on an unlabelled band) and register on that. Without new
   acquisition, the next best options are to winsorise the registration image at a high percentile so
   the rim cannot dominate, or to register the tissue silhouette rather than the intensity image.
   This attacks the cause rather than the symptom.
2. **Replace the convex hull.** Fill holes, morphologically close, take the largest connected
   component - but do not convexify. Validate against the registered atlas volume and by eye on
   sagittal sections.
3. **Make the mask a first-class pipeline output and QC it per animal**: tissue volume, Dice with the
   registered atlas, fraction of tissue outside the atlas, and fraction of signal outside the atlas.
   These four numbers would have surfaced this problem immediately.
4. **Report the signal accounting explicitly.** `signal_assignment_summary.csv` already records
   `n_in_atlas` / `n_filled` / `n_excluded_beyond_cap`; those should be compared *between groups* as
   standard QC, not just inspected per animal. A group difference there invalidates regional
   comparisons until explained.

## How we will know it is fixed

- Fraction of tissue outside the registered atlas is comparable between groups (no significant Welch
  difference), and below ~2% for every animal.
- Fraction of signal filled from outside the atlas is comparable between groups.
- The regional results are re-run on the re-registered data and the ventral midbrain/pons cluster
  either survives or does not - either way we then know which it was.

## Open questions

- Is there any unlabelled channel in the raw acquisitions that could serve as an anatomy reference?
- Would brainreg accept a mask-driven or multi-channel registration without pipeline surgery?
- Should the 200 µm fill cap be revisited once the fit is better? It exists to rescue misregistered
  surface tissue; with a good fit less of it is needed, and the cap is currently doing a lot of
  silent work.

## Related documents

- `native_space_surface_analysis.md` - the registration-free measurement that can answer the
  biological question while this is being fixed.
- `../ATX7851_analysis/tracer_penetration_plan.md` - where `brain_mask.tif` was first found to be
  unusable.
