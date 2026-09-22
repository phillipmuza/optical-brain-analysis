# Title: Contact sheets for checking transverse stack direction (dorsal->ventral vs ventral->dorsal)
# Author: Phillip Muza
# Date: 04/08/26

"""Sample evenly spaced slices from one image per folder and write a labelled contact sheet.

All stacks are acquired in the transverse plane, but some run dorsal -> ventral and others
ventral -> dorsal. That direction determines which transform in image_transformations.py is
correct, and the only reliable way to tell is to look. This script renders a sheet per animal
so the whole dataset can be scored quickly, and writes a CSV to record the call.

Only the sampled planes are read from disk, so multi-GB stacks cost almost nothing.

Reading the sheets:
  Slices run left to right, top to bottom, from the FIRST plane in the file to the LAST.
  - dorsal -> ventral: starts at the brain surface (cortex, small cross-section), the section
    grows through the midbrain, then narrows towards the ventral surface / brainstem.
  - ventral -> dorsal: the reverse -- starts ventral, ends at the cortical surface.

Usage:
  python check_orientation.py --parent_dir "D:/awake_tracer_infusions"
  python check_orientation.py --parent_dir "D:/awake_tracer_infusions" --image_name TxR.tif --n_slices 12
"""

import os
import csv
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import exposure

from tiff_io import open_stack

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_images(parent_dir, image_name):
    """Find one matching image per directory, matched case-insensitively."""
    matches = []
    for root, dirs, files in os.walk(parent_dir):
        # Skip output folders this or the analysis pipeline creates
        dirs[:] = [d for d in dirs if d not in
                   ("orientation_check", "downsampled", "registration_dir", "debug", "transformations")]
        for f in files:
            if f.lower() == image_name.lower():
                matches.append(os.path.join(root, f))
                break
    return sorted(matches)


def read_slices(image_path, n_slices, margin, max_px):
    """Read n_slices evenly spaced planes along axis 0, without loading the whole stack."""
    with open_stack(image_path) as (n_pages, read_plane):
        lo = int(round(n_pages * margin))
        hi = int(round(n_pages * (1 - margin))) - 1
        hi = max(hi, lo)
        indices = np.unique(np.linspace(lo, hi, n_slices).round().astype(int))

        planes = []
        for idx in indices:
            plane = read_plane(idx).astype(np.float32)

            # Downscale for display, preserving aspect ratio
            scale = max_px / max(plane.shape)
            if scale < 1:
                target = (max(int(plane.shape[0] * scale), 1), max(int(plane.shape[1] * scale), 1))
                plane = resize(plane, target, anti_aliasing=True, preserve_range=True)

            # Per-slice percentile stretch so dim ventral/dorsal ends stay visible
            p_lo, p_hi = np.percentile(plane, (1, 99.5))
            if p_hi <= p_lo:
                p_lo, p_hi = float(plane.min()), float(max(plane.max(), plane.min() + 1))
            planes.append(exposure.rescale_intensity(plane, in_range=(p_lo, p_hi), out_range=(0.0, 1.0)))

    return indices, planes, n_pages


def make_sheet(image_path, label, output_path, n_slices, margin, max_px):
    indices, planes, n_pages = read_slices(image_path, n_slices, margin, max_px)

    n_cols = 5
    n_rows = int(np.ceil(len(planes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.1 * n_rows), facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax, idx, plane in zip(axes, indices, planes):
        ax.imshow(plane, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"z = {idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Mark the ends, since direction is the whole question
    axes[0].set_title(f"z = {indices[0]}  (FIRST)", fontsize=10, color="tab:red", fontweight="bold")
    axes[len(planes) - 1].set_title(f"z = {indices[-1]}  (LAST)", fontsize=10, color="tab:blue", fontweight="bold")

    for ax in axes[len(planes):]:
        ax.axis("off")

    fig.suptitle(f"{label}\n{n_pages} slices along axis 0   |   FIRST (red) -> LAST (blue)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5)
    fig.savefig(output_path, dpi=90)
    plt.close(fig)

    return n_pages


def main():
    parser = argparse.ArgumentParser(
        description="Render contact sheets to check transverse stack direction (D->V vs V->D)")
    parser.add_argument('--parent_dir', type=str, required=True,
                        help='Directory to search recursively for images')
    parser.add_argument('--image_name', type=str, default='FITC.tif',
                        help='Image sampled in each folder (default: FITC.tif)')
    parser.add_argument('--n_slices', type=int, default=10,
                        help='Number of slices per sheet (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where sheets are written (default: <parent_dir>/orientation_check)')
    parser.add_argument('--margin', type=float, default=0.0,
                        help='Fraction of the stack skipped at each end, e.g. 0.05 to drop empty '
                             'leading/trailing planes (default: 0.0, full range)')
    parser.add_argument('--max_px', type=int, default=420,
                        help='Longest edge of each displayed slice in pixels (default: 420)')
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.parent_dir)
    output_dir = args.output_dir or os.path.join(parent_dir, "orientation_check")
    os.makedirs(output_dir, exist_ok=True)

    images = find_images(parent_dir, args.image_name)
    if not images:
        logging.error(f"No {args.image_name} found under {parent_dir}")
        return

    logging.info(f"Found {len(images)} image(s) named {args.image_name}")

    rows = []
    for image_path in images:
        label = os.path.relpath(os.path.dirname(image_path), parent_dir)
        safe_label = label.replace(os.sep, "__").replace(" ", "_") or "root"
        sheet_path = os.path.join(output_dir, f"{safe_label}.png")

        try:
            n_pages = make_sheet(image_path, label, sheet_path, args.n_slices, args.margin, args.max_px)
            logging.info(f"Wrote {sheet_path}")
            rows.append({"folder": label, "image": os.path.basename(image_path),
                         "n_slices": n_pages, "sheet": os.path.basename(sheet_path), "direction": ""})
        except Exception as e:
            logging.error(f"Failed on {image_path}: {e}")

    # Scoring sheet: fill the direction column with dorsal_to_ventral or ventral_to_dorsal
    csv_path = os.path.join(output_dir, "orientations.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["folder", "image", "n_slices", "sheet", "direction"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Wrote {len(rows)} sheet(s) to {output_dir}")
    logging.info(f"Now fill the 'direction' column in {csv_path} "
                 "with dorsal_to_ventral or ventral_to_dorsal")


if __name__ == "__main__":
    main()
