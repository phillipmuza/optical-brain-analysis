# Title: Verify transverse -> coronal transformation and slice direction consistency
# Author: Phillip Muza
# Date: 05/08/26

"""Render one comparison grid across animals to confirm image_transformations.py worked.

Two things need checking after transformation:
  1. The plane is now coronal. Slices along axis 0 should look like coronal sections
     (roughly symmetric left/right about a vertical midline), not horizontal ones.
  2. Every animal runs the same way. Both transforms are meant to normalise the two
     acquisition directions to a single common orientation, so the same column position
     should show the same anatomical level in every row. An animal scored the wrong way
     round appears as a row running backwards against the others.

One row per animal makes disagreement obvious in a way per-animal sheets do not.

Usage:
  python check_transform.py --parent_dir "D:/.../LSM_analysis/transformed"
"""

import os
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
        dirs[:] = [d for d in dirs if d not in
                   ("orientation_check", "transform_check", "downsampled", "registration_dir")]
        for f in files:
            if f.lower() == image_name.lower():
                matches.append(os.path.join(root, f))
                break
    return sorted(matches)


def read_planes(image_path, fractions, max_px):
    """Read planes at the given fractional depths along axis 0, without loading the stack."""
    with open_stack(image_path) as (n_pages, read_plane):
        indices = [min(int(round(f * (n_pages - 1))), n_pages - 1) for f in fractions]

        planes = []
        for idx in indices:
            plane = read_plane(idx).astype(np.float32)

            scale = max_px / max(plane.shape)
            if scale < 1:
                target = (max(int(plane.shape[0] * scale), 1), max(int(plane.shape[1] * scale), 1))
                plane = resize(plane, target, anti_aliasing=True, preserve_range=True)

            p_lo, p_hi = np.percentile(plane, (1, 99.5))
            if p_hi <= p_lo:
                p_lo, p_hi = float(plane.min()), float(max(plane.max(), plane.min() + 1))
            planes.append(exposure.rescale_intensity(plane, in_range=(p_lo, p_hi), out_range=(0.0, 1.0)))

    return indices, planes, n_pages


def main():
    parser = argparse.ArgumentParser(
        description="Comparison grid to verify the transverse -> coronal transform")
    parser.add_argument('--parent_dir', type=str, required=True,
                        help='Directory of transformed images to check')
    parser.add_argument('--image_name', type=str, default='FITC.tif',
                        help='Image sampled in each folder (default: FITC.tif)')
    parser.add_argument('--n_slices', type=int, default=6,
                        help='Slices per animal (default: 6)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where the grid is written (default: <parent_dir>/transform_check)')
    parser.add_argument('--margin', type=float, default=0.08,
                        help='Fraction skipped at each end, avoiding empty leading/trailing '
                             'planes (default: 0.08)')
    parser.add_argument('--max_px', type=int, default=300,
                        help='Longest edge of each displayed slice in pixels (default: 300)')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[6.55, 20, 6.55],
                        metavar=('AX0', 'AX1', 'AX2'),
                        help='Voxel size of the TRANSFORMED images in microns. The transform '
                             'reorders the raw (20, 6.55, 6.55) to (6.55, 20, 6.55), and the '
                             'displayed slices are anisotropic without this correction '
                             '(default: 6.55 20 6.55)')
    args = parser.parse_args()

    parent_dir = os.path.abspath(args.parent_dir)
    output_dir = args.output_dir or os.path.join(parent_dir, "transform_check")
    os.makedirs(output_dir, exist_ok=True)

    images = find_images(parent_dir, args.image_name)
    if not images:
        logging.error(f"No {args.image_name} found under {parent_dir}")
        return

    logging.info(f"Found {len(images)} image(s) named {args.image_name}")

    # Sample the same fractional depths in every animal so columns are comparable
    fractions = np.linspace(args.margin, 1 - args.margin, args.n_slices)

    rows = []
    for image_path in images:
        label = os.path.relpath(os.path.dirname(image_path), parent_dir)
        try:
            indices, planes, n_pages = read_planes(image_path, fractions, args.max_px)
            rows.append((label, indices, planes, n_pages))
            logging.info(f"{label}: {n_pages} slices along axis 0")
        except Exception as e:
            logging.error(f"Failed on {image_path}: {e}")

    if not rows:
        logging.error("Nothing to plot")
        return

    # A displayed slice spans axes 1 and 2, whose voxel sizes differ; without this the sections
    # render squashed and cannot be judged anatomically
    display_aspect = args.voxel_size[1] / args.voxel_size[2]

    n_cols = args.n_slices
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.3 * n_cols, 2.2 * len(rows)), facecolor="white")
    axes = np.atleast_2d(axes)

    for r, (label, indices, planes, n_pages) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(planes[c], cmap="gray", vmin=0, vmax=1, aspect=display_aspect)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{fractions[c] * 100:.0f}% depth", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{label}\n({n_pages} slices)", fontsize=9, rotation=0,
                              ha='right', va='center', labelpad=42)

    fig.suptitle("Transformed stacks: slices along axis 0 (should be CORONAL)\n"
                 "Columns are matched depths -- every row should progress the same way",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)

    output_path = os.path.join(output_dir, "transform_check.png")
    fig.savefig(output_path, dpi=95)
    plt.close(fig)

    logging.info(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
