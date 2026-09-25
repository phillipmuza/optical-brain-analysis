# Measuring tracer from each animal's own tissue surface, in its own image space

**Written 15/09/2026, from the NS24122 cohort analysis.** Companion to
`registration_and_mask_issues.md`, which describes the problem this design avoids.

## The question this is for

Does NS24122 increase the amount of CSF tracer reaching the brain surface?

IVIS says yes for FITC (30 mg vs Vehicle: 1.9x dorsal, p = 0.059; 1.6x ventral, p = 0.046). Every
light-sheet analysis so far says nothing. The two modalities are not in contradiction so much as
measuring different things, and the light-sheet pipeline is built in a way that cannot answer this
particular question.

## Why atlas space is the wrong instrument here

Three reasons, in increasing order of seriousness.

**1. The atlas does not cover the compartment.** At 30 min the tracer is overwhelmingly a rim on the
pial surface and in the basal cisterns (`figures/*_per_animal_absolute.png`). The atlas annotation
stops at the brain surface; subarachnoid space is not a labelled structure. Signal there is either
pushed into whichever outermost region it abuts (within the 200 µm fill) or discarded. IVIS, imaging
the intact surface, counts all of it.

**2. Every regional metric is relative.** `% coverage of a region's volume`, `share of the animal's
total`, `% of signal deeper than 1 mm` - all proportions, by design, because absolute fluorescence was
assumed not to be comparable between animals. That assumption turns out to be wrong for this cohort:
acquisition settings were identical, and Vehicle and 30 mg animals were randomised within imaging
day. A change in *how much* tracer arrived is therefore measurable, but no metric we built can see
it - and a uniform increase would be normalised away entirely.

**3. Registration is affected by the thing being measured.** This is the serious one. The
registration image is a copy of the FITC channel, so the bright rim drives the alignment; the fit is
measurably worse in treated animals (tissue outside the atlas 4.6% vs 2.3%, p = 0.048); and that thin
rim carries a third of the signal. The result is that the atlas-based accounting differs by group:

| FITC, integrated tracer signal | Vehicle | 30 mg | ratio | p |
|---|---|---|---|---|
| inside atlas | 4.1e7 | 6.2e7 | 1.5x | ns |
| **outside atlas** | 7.4e6 | 5.3e7 | **7.2x** | 0.024 (dorsal) / 0.051 (ventral) |

Any measurement that depends on where the atlas boundary fell inherits a group-dependent bias. We
cannot use the atlas to adjudicate a question about the compartment that the atlas misplaces.

## The proposal

Measure depth from **each animal's own tissue surface, in its own image space**. No registration, no
atlas, no deformation fields, no region parcellation, no cross-animal normalisation.

Per animal:

1. **Tissue mask** from the registration image (`fixed_threshold_intensity.tissue_mask()`,
   `median - 2 x robust SD`), holes filled, largest connected component kept. This defines the
   brain's own boundary.
2. **Depth map**: Euclidean distance transform inward from the tissue boundary, in the animal's own
   20 µm grid. Every voxel gets its distance below that animal's own surface.
3. **Surface orientation**: label each surface voxel dorsal or ventral by which side of its own
   column's mid-height it sits on, matching how IVIS photographs the two faces. (Refinement if
   needed: the sign of the outward normal from the distance transform gradient.)
4. **Integrate signal in shells**: for each depth shell (0-100, 100-200, ... µm) and each surface,
   the sum of above-background intensity and the sum restricted to above-threshold voxels, plus the
   shell's tissue volume and surface area.
5. **Compare groups** on the absolute quantities, and **correlate per animal against IVIS**
   dorsal/ventral radiant efficiency.

Cost: one distance transform per animal on a 203 M-voxel volume, roughly 30 minutes for the cohort.
Everything needed is already on disk.

## Why this is decisive

The failure modes are independent of IVIS's. IVIS has no registration, no thresholding and a dark
baseline, but poor spatial resolution and no depth information. Native-space light-sheet has
excellent resolution and depth, no registration, and an autofluorescent baseline. If both show more
tracer in the surface shell of treated animals, the only shared assumption left is that the tracer
was in the tissue - which is the claim.

A pre-declared reading, so this cannot be rationalised afterwards:

- **Treated animals show more signal in their own outer 200 µm** -> the effect is biological;
  NS24122 increases tracer at the pial surface and in the cisterns, IVIS was right, and the
  atlas-based analyses missed it for the structural reasons above.
- **No difference in native space** -> the IVIS/light-sheet divergence is an accounting artefact of
  the registration, and the "outside atlas" excess is misattributed tissue rather than extra tracer.
- **Difference only in the outermost shell (0-100 µm) and not below** -> favours material on the
  surface (cisternal fluid, meningeal remnants) rather than uptake into tissue, which matters for
  interpreting what the drug does.

## What it cannot do

- **No anatomical attribution.** A native-space shell cannot say "substantia nigra". Regional
  questions still need the atlas; this measurement answers "how much, and how deep from the surface",
  not "where".
- **No cross-animal anatomical alignment**, so shells must be normalised by that animal's own tissue
  volume or surface area before comparison; brains differ in size (linear scale 1.28-1.46x the
  atlas).
- **It inherits the tissue mask.** If the mask includes dura or debris, that lands in the outermost
  shell - precisely where the effect is expected. The mask must be checked per animal (visual, plus
  tissue volume consistency) before the result is trusted, and the 0-100 µm shell treated with
  particular suspicion.
- **It does not fix the pipeline.** The registration problem still needs solving for every regional
  analysis; see `registration_and_mask_issues.md`.

## Checks to run alongside

- Total above-threshold signal per animal should reproduce `threshold_summary.csv` (same voxels,
  different geometry) - a direct arithmetic check that the native-space accounting is complete.
- Tissue volume per animal should be stable within group and consistent with the registered atlas
  volume scaled by the animal's linear factor.
- The dorsal/ventral split should be checked on sections, not assumed.
- Correlate the native-space shell measure against IVIS per animal; the atlas-space version currently
  gives dorsal Spearman rho = 0.79 (FITC) and 0.57-0.73 (TxR), ventral 0.20-0.50. Native space should
  do at least as well if it is the better instrument, particularly ventrally where the atlas-based
  agreement is weakest.

## Keeping both

This does not replace atlas space. The two answer different questions and should both be kept:

| question | instrument |
|---|---|
| how much tracer, how deep from the surface | native space, this document |
| which structures, laterality, spatial clusters | atlas space, the existing notebooks |
| does it agree with an independent modality | native space vs IVIS, per animal |

The atlas-space resampled volumes built for this work (`atlas_space/`, 40 µm, raw intensity) are
worth keeping regardless: they make every animal directly comparable voxel by voxel and are what
made the compartment problem visible in the first place.

## Status: run 15/09/2026

Implemented in `../native_space_analysis/` (`improve_mask.py` -> `native_depth.py` ->
`native_depth_analysis.py`). Masks 43 min, depth profiles 88 min for 20 animals.

### Result: the pre-declared "biological" branch

FITC, 30 mg vs Vehicle, integrated tracer signal in the animal's own surface shells:

| band | Vehicle | 30 mg | ratio | Welch p | exact p |
|---|---|---|---|---|---|
| dorsal 0-200 µm | 2.31e8 | 8.12e8 | **3.5x** | 0.046 | 0.021 |
| dorsal 0-500 µm | 2.65e8 | 8.75e8 | 3.3x | 0.047 | 0.026 |
| ventral 0-200 µm | 3.46e8 | 1.05e9 | 3.0x | 0.066 | 0.031 |
| ventral 0-500 µm | 4.40e8 | 1.24e9 | 2.8x | 0.070 | 0.036 |

Signal per surface voxel (so this is not simply a larger mask): dorsal 3.2x (exact p = 0.032),
ventral 2.7x (exact p = 0.042). IVIS measured 1.9x dorsal and 1.6x ventral for the same animals -
same direction, same order of magnitude, from an instrument with unrelated failure modes.

TxR at 30 mg goes the same way but does not reach significance (1.4-1.8x, p = 0.20-0.41); IVIS was
also weak for TxR (1.11-1.34x, ns). The 10 mg group shows nothing consistent.

### Agreement with IVIS improved, as the design predicted

Spearman rho, per animal (n = 18), native-space surface signal vs IVIS, against the atlas-space
version of the same comparison:

| | atlas space | native space |
|---|---|---|
| FITC dorsal | 0.79 | **0.83** (0.84 per-voxel) |
| FITC ventral | 0.33 | **0.54** (0.65) |
| TxR dorsal | 0.57 | 0.57 (0.70) |
| TxR ventral | 0.33 | **0.48** (0.49) |

All eight correlations are now significant (p <= 0.044); ventrally they were not, in atlas space.

### What it says, and what it does not

The extra tracer sits **at the surface**, not deeper: the relative measure moves the other way
(fraction of FITC signal deeper than 0.5 mm, ventral: 0.32 in Vehicle vs 0.18 at 30 mg, p = 0.0497).
So this is about how much tracer reaches and stays at the pial surface and cisterns, not about
enhanced penetration into parenchyma - which is exactly why the depth- and region-based analyses
found nothing.

Caveats that stand:
- Eight primary tests (2 tracers x 2 sides x 2 doses); Bonferroni over that family would need
  p < 0.006, and the best exact p is 0.021. The direction was pre-specified from IVIS, and the
  per-animal agreement is independent support, but this is not a corrected-significant result.
- TxR does not reach significance on its own.
- The background and threshold of the 10 mg batch differ from Vehicle (FITC background 863 vs 931),
  so 10 mg comparisons carry a batch signature. The 30 mg background matches Vehicle (925 vs 931),
  which is what makes the 30 mg comparison trustworthy.
- n = 6-7 per group.

Figures: `../native_space_analysis/figures/native_depth_profiles.png`,
`native_depth_headline.png`, `native_vs_ivis.png`. Tables in
`../native_space_analysis/outputs/`.

## Update 16/09/2026: an25 and an26 removed

Experimental errors were identified in an25 and an26, both **NS24122 10 mg**. They join an12, an14
and an35 in `EXCLUDE_ANIMALS` everywhere. Group sizes are now Vehicle 7, 10 mg **5**, 30 mg 6
(n = 18). Their mask and depth files are still on disk; the exclusion is a filter in
`native_depth_analysis.py` and `brain_maps.py`, so it can be reversed.

**Nothing in the Vehicle vs 30 mg comparison changes** - neither excluded animal is in either group,
so the group means, ratios and the pairwise tests above are exactly as reported. What changes is the
three-group ANOVA, which loses two animals from one arm.

One-way ANOVA of treatment, then Tukey (the IVIS procedure), on integrated signal:

| tracer, region | ANOVA p (n=20) | ANOVA p (n=18) | Tukey 30 mg vs Vehicle (n=20) | (n=18) |
|---|---|---|---|---|
| FITC dorsal 0-500 µm | 0.055 | 0.056 | 0.045 | **0.048** (g = 1.31) |
| FITC ventral 0-500 µm | 0.040 | 0.064 | 0.045 | 0.055 (g = 1.20) |
| FITC dorsal 0-200 µm | - | **0.046** | - | - |
| TxR ventral 0-500 µm | **0.049** | 0.154 | ns | ns |

Two things worth noting, in opposite directions:

- **The TxR ventral ANOVA has disappeared** (0.049 -> 0.154). It was flagged at the time as being
  driven by the 10 mg batch rather than by dose, and removing two 10 mg animals removed it. That is
  the result behaving as the caveat predicted, and it is one fewer thing to explain.
- **FITC ventral has slipped just above 0.05** (Tukey 0.045 -> 0.055) purely from the loss of power
  in a third arm that is not part of the contrast. The effect size is unchanged (Hedges' g = 1.20,
  2.8x), as is the Vehicle-vs-30 mg pairwise test. Dorsal still passes at 0.048.

The IVIS agreement is **unchanged** (rho 0.48-0.84, all p <= 0.044): an25 and an26 were never in the
IVIS brain table either, so those 18 animals were always the 18 being correlated. That is a small
independent confirmation that the same two animals were already set aside on the IVIS side.
