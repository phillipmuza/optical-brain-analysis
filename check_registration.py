# Title: Contact sheets for checking brainreg registration (boundaries over preprocessed)
# Author: Phillip Muza
# Date: 11/09/26

"""Overlay registration_dir/boundaries.tiff on preprocessed.tiff at sampled planes, per animal.

Replaces loading both stacks into napari and scrolling: each animal gets one PNG showing the
atlas boundaries (magenta) over the registration image at evenly spaced coronal planes plus a row
each of sagittal and horizontal planes. Misregistration that is easy to miss scrolling coronally
(an A-P shift, a tilted midline, a squashed cerebellum) is usually obvious in the orthogonal rows.

Planes are sampled across the brain's extent (tissue union atlas boundaries), not the whole stack,
and each tile is cropped to that extent, so no tiles are spent on empty background.

Each animal is also scored on how well the atlas outline matches the tissue outline. Tissue is an
Otsu threshold on log intensity of preprocessed.tiff, holes filled per plane. registration_dir/
brain_mask.tif is deliberately not used: it runs well out into the background (over half of an16's
stack), which swamps the score. Only the outer outline is compared, so a good score does not rule
out internal misalignment; it is a way to order which sheets to look at first, not a verdict:
  - dice                  overlap of tissue and registered atlas (1 = perfect)
  - tissue_outside_atlas  fraction of tissue the atlas does not cover (atlas too small/shifted)
  - atlas_outside_tissue  fraction of the atlas lying on background (atlas too big/shifted; a few
                          % is normal, as dim tissue at the surface falls under the threshold)

Sheets are only re-rendered when missing or older than boundaries.tiff, so re-registering an
animal and re-running refreshes just that one. The verdict/notes columns of registration_qc.csv
are kept across runs.

Usage:
  python check_registration.py --path "E:/tracer_uptake/wt_mice/new_analysis_0926"
  python check_registration.py --path "E:/tracer_uptake/wt_mice/new_analysis_0926/an16/registration_dir"
  python check_registration.py --path ".../an16/registration_dir" --napari
"""

import os
import csv
import argparse
import logging

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_otsu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REGISTRATION_DIRNAME = "registration_dir"
OUTPUT_DIRNAME = "registration_check"
CSV_NAME = "registration_qc.csv"
CSV_FIELDS = ["folder", "dice", "tissue_outside_atlas", "atlas_outside_tissue", "sheet",
              "verdict", "notes"]
BOUNDARY_RGB = np.array([1.0, 0.0, 1.0], dtype=np.float32)
BOUNDARY_ALPHA = 0.85
# Every Nth coronal plane of registered_atlas.tiff is read for the overlap score; it is the
# largest file (uint32) and the score barely changes with denser sampling.
ATLAS_PLANE_STRIDE = 4

# Axis 0 is coronal for brainreg's "asr" output: axis 1 runs dorsal -> ventral, axis 2 right -> left.
VIEWS = (
    # name,        axis, rows, cols
    ("coronal",    0,    2,    6),
    ("sagittal",   2,    1,    6),
    ("horizontal", 1,    1,    6),
)
N_COLS = 6


def find_registration_dirs(path):
    """Return every registration_dir under path holding boundaries.tiff, or path itself."""
    if os.path.exists(os.path.join(path, "boundaries.tiff")):
        return [path]

    matches = []
    for root, dirs, files in os.walk(path):
        if os.path.basename(root) == REGISTRATION_DIRNAME:
            if "boundaries.tiff" in files:
                matches.append(root)
            dirs[:] = []
            continue
        # Only registration_dir holds what we need; skip the other pipeline outputs
        dirs[:] = [d for d in dirs if d not in
                   ("downsampled", "transformed", "orientation_check", OUTPUT_DIRNAME, "debug")]
    return sorted(matches)


def load_volumes(reg_dir):
    """Load the stacks the sheet and score need, checking they share one voxel grid."""
    boundaries = tifffile.imread(os.path.join(reg_dir, "boundaries.tiff")) > 0
    preprocessed = tifffile.imread(os.path.join(reg_dir, "preprocessed.tiff"))

    atlas_path = os.path.join(reg_dir, "registered_atlas.tiff")
    with tifffile.TiffFile(atlas_path) as tif:
        n_planes = tif.series[0].shape[0]
        atlas_idx = np.arange(0, n_planes, ATLAS_PLANE_STRIDE)
        if len(tif.pages) == n_planes:
            atlas_sub = np.stack([tif.pages[int(i)].asarray() > 0 for i in atlas_idx])
        else:
            atlas_sub = tifffile.memmap(atlas_path, mode='r')[atlas_idx] > 0

    shapes = {"boundaries": boundaries.shape, "preprocessed": preprocessed.shape,
              "registered_atlas": (n_planes,) + atlas_sub.shape[1:]}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Stacks in {reg_dir} disagree in shape: {shapes}")

    return boundaries, preprocessed, atlas_idx, atlas_sub


def tissue_mask(image):
    """Tissue vs background: Otsu on log intensity, then per-plane opening and hole filling.

    Log intensity keeps dim tissue at the surface above threshold; linear Otsu is pulled up by the
    bright interior. Filling holes per plane puts ventricles back inside the tissue.
    """
    sample = np.log1p(np.clip(image[::4, ::4, ::4], 0, None).astype(np.float32))
    threshold = np.expm1(threshold_otsu(sample))
    return np.stack([ndimage.binary_fill_holes(ndimage.binary_opening(plane > threshold, iterations=2))
                     for plane in image])


def overlap_scores(tissue_sub, atlas_sub):
    """Dice and one-sided miss fractions between the tissue mask and the registered atlas."""
    n_tissue = int(tissue_sub.sum())
    n_atlas = int(atlas_sub.sum())
    n_both = int(np.logical_and(tissue_sub, atlas_sub).sum())
    if n_tissue == 0 or n_atlas == 0:
        return float("nan"), float("nan"), float("nan")
    dice = 2 * n_both / (n_tissue + n_atlas)
    return dice, 1 - n_both / n_tissue, 1 - n_both / n_atlas


def brain_bbox(tissue, boundaries, pad=10):
    """Bounding box (lo, hi) per axis of tissue union atlas boundaries, padded."""
    footprint = tissue | boundaries
    bbox = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        present = np.flatnonzero(footprint.any(axis=other))
        if present.size == 0:
            raise ValueError("Neither the tissue mask nor the boundaries contain any voxels")
        bbox.append((max(int(present[0]) - pad, 0),
                     min(int(present[-1]) + pad + 1, footprint.shape[axis])))
    return bbox


def take_plane(volume, axis, idx, bbox):
    """Plane idx along axis, cropped to bbox and oriented with dorsal up."""
    crops = [slice(lo, hi) for lo, hi in bbox]
    crops[axis] = int(idx)
    plane = volume[tuple(crops)]
    # Sagittal comes out (A-P, D-V); transpose so D-V is vertical like the coronal view
    return plane.T if axis == 2 else plane


def display_range(image, tissue):
    """One contrast range per animal: background floor to 1.3x the tissue 95th percentile.

    A plain high percentile is not safe as the ceiling: an12's very bright ventral hindbrain is the
    top ~2% of its tissue (up to 27000 vs a median of 930), so per-tile or tissue p99.5 stretching
    turned the rest of the brain black. 1.3x p95 sits near p99.5 in normal animals (an16: 1485 vs
    1471) and lets such objects saturate instead.
    """
    sample = image[::4, ::4, ::4]
    sample_tissue = tissue[::4, ::4, ::4]
    lo = float(np.percentile(sample, 1))
    hi = 1.3 * float(np.percentile(sample[sample_tissue], 95)) if sample_tissue.any() else float(sample.max())
    return lo, max(hi, lo + 1)


def composite(grey_plane, boundary_plane, lo, hi):
    """RGB tile: greyscale stretched to (lo, hi) with boundaries blended on top."""
    grey = np.clip((grey_plane.astype(np.float32) - lo) / (hi - lo), 0, 1)

    rgb = np.repeat(grey[..., None], 3, axis=2)
    rgb[boundary_plane] = (1 - BOUNDARY_ALPHA) * rgb[boundary_plane] + BOUNDARY_ALPHA * BOUNDARY_RGB
    return np.clip(rgb, 0, 1)


def sample_indices(lo, hi, n, margin):
    """n evenly spaced planes inside [lo, hi), skipping `margin` of the range at each end."""
    span = hi - lo
    return np.unique(np.linspace(lo + span * margin, hi - 1 - span * margin, n).round().astype(int))


def make_sheet(boundaries, preprocessed, tissue, label, sheet_path, scores, n_coronal):
    bbox = brain_bbox(tissue, boundaries)
    lo, hi = display_range(preprocessed, tissue)

    views = [(name, axis, rows, cols) if name != "coronal"
             else (name, axis, int(np.ceil(n_coronal / N_COLS)), N_COLS)
             for name, axis, rows, cols in VIEWS]
    n_rows = sum(rows for _, _, rows, _ in views)

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(3.2 * N_COLS, 3.0 * n_rows), facecolor="white")
    axes = np.atleast_2d(axes)
    row = 0
    for name, axis, rows, _ in views:
        n_planes = n_coronal if name == "coronal" else rows * N_COLS
        # Orthogonal views skip the thin edges, where there is little tissue to judge
        margin = 0.03 if name == "coronal" else 0.15
        indices = sample_indices(*bbox[axis], n_planes, margin)

        view_axes = axes[row:row + rows].ravel()
        for ax, idx in zip(view_axes, indices):
            tile = composite(take_plane(preprocessed, axis, idx, bbox),
                             take_plane(boundaries, axis, idx, bbox), lo, hi)
            ax.imshow(tile, interpolation="antialiased")
            ax.set_title(f"{name}  {['z', 'y', 'x'][axis]} = {idx}", fontsize=9)
        for ax in view_axes:
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in view_axes[len(indices):]:
            ax.axis("off")
        row += rows

    fig.suptitle(f"{label}\nDice {scores['dice']:.3f}   |   tissue outside atlas "
                 f"{scores['tissue_outside_atlas']:.1%}   |   atlas outside tissue "
                 f"{scores['atlas_outside_tissue']:.1%}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.5)
    fig.savefig(sheet_path, dpi=110)
    plt.close(fig)


def score_and_render(reg_dir, label, sheet_path, n_coronal):
    boundaries, preprocessed, atlas_idx, atlas_sub = load_volumes(reg_dir)
    tissue = tissue_mask(preprocessed)
    dice, tissue_out, atlas_out = overlap_scores(tissue[atlas_idx], atlas_sub)
    scores = {"dice": dice, "tissue_outside_atlas": tissue_out, "atlas_outside_tissue": atlas_out}
    make_sheet(boundaries, preprocessed, tissue, label, sheet_path, scores, n_coronal)
    return {k: round(v, 4) for k, v in scores.items()}


def dice_sort_key(row):
    """Sort rows by Dice ascending; unscored rows first."""
    try:
        return float(row["dice"])
    except (KeyError, TypeError, ValueError):
        return -1.0


def read_existing_rows(csv_path):
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="") as fh:
        return {r["folder"]: r for r in csv.DictReader(fh)}


def open_in_napari(reg_dir):
    """Load preprocessed + boundaries into napari the way they are usually checked by hand."""
    import napari

    preprocessed = tifffile.imread(os.path.join(reg_dir, "preprocessed.tiff"))
    boundaries = tifffile.imread(os.path.join(reg_dir, "boundaries.tiff"))
    sample = preprocessed[::4, ::4, ::4]
    lo, hi = np.percentile(sample[sample > 0] if (sample > 0).any() else sample, (1, 99.5))

    viewer = napari.Viewer(title=reg_dir)
    viewer.add_image(preprocessed, name="preprocessed", colormap="gray", contrast_limits=(lo, hi))
    viewer.add_image(boundaries, name="boundaries", colormap="magenta", blending="additive",
                     contrast_limits=(0, 1), opacity=0.8)
    viewer.dims.set_point(0, preprocessed.shape[0] // 2)
    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="Render contact sheets of atlas boundaries over the registration image")
    parser.add_argument('--path', type=str, required=True,
                        help='A registration_dir, or a directory searched recursively for them')
    parser.add_argument('--output_dir', type=str, default=None,
                        help=f'Where sheets are written (default: {OUTPUT_DIRNAME}/ beside the animal '
                             'folders, i.e. <path>/{OUTPUT_DIRNAME} for a parent directory)')
    parser.add_argument('--n_coronal', type=int, default=12,
                        help='Coronal planes per sheet (default: 12)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-render every sheet, not just missing or out-of-date ones')
    parser.add_argument('--napari', action='store_true',
                        help='Open a single registration_dir in napari instead of writing a sheet')
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    reg_dirs = find_registration_dirs(path)
    if not reg_dirs:
        logging.error(f"No {REGISTRATION_DIRNAME} containing boundaries.tiff found under {path}")
        return

    if args.napari:
        if len(reg_dirs) != 1:
            logging.error(f"--napari needs a single registration_dir; {path} holds {len(reg_dirs)}")
            return
        open_in_napari(reg_dirs[0])
        return

    # Animal folders are labelled relative to the directory that holds them all, so single-animal
    # and whole-batch runs write into the same registration_check/ and the same CSV.
    single = reg_dirs == [path]
    base_dir = os.path.dirname(os.path.dirname(path)) if single else path
    output_dir = args.output_dir or os.path.join(base_dir, OUTPUT_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, CSV_NAME)
    rows = read_existing_rows(csv_path)
    logging.info(f"Found {len(reg_dirs)} registration_dir(s)")

    for reg_dir in reg_dirs:
        label = os.path.relpath(os.path.dirname(reg_dir), base_dir)
        safe_label = label.replace(os.sep, "__").replace(" ", "_") or "root"
        sheet_path = os.path.join(output_dir, f"{safe_label}.png")

        up_to_date = (os.path.exists(sheet_path) and label in rows and
                      os.path.getmtime(sheet_path) >= os.path.getmtime(os.path.join(reg_dir, "boundaries.tiff")))
        if up_to_date and not args.overwrite:
            logging.info(f"Up to date, skipping: {label}")
            continue

        try:
            scores = score_and_render(reg_dir, label, sheet_path, args.n_coronal)
        except Exception as e:
            logging.error(f"Failed on {reg_dir}: {e}")
            continue

        previous = rows.get(label, {})
        rows[label] = {"folder": label, **scores, "sheet": os.path.basename(sheet_path),
                       "verdict": previous.get("verdict", ""), "notes": previous.get("notes", "")}
        logging.info(f"Wrote {sheet_path}  (Dice {scores['dice']:.3f})")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        # Worst overlap first, so the animals most worth checking are at the top
        writer.writerows(sorted(rows.values(), key=dice_sort_key))

    logging.info(f"Scores in {csv_path}; fill 'verdict' (pass / fail / redo) and 'notes' as you go")


if __name__ == "__main__":
    main()
