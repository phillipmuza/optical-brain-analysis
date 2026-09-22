# Title: Orientation scoring and transverse -> coronal transform, orchestrated per animal
# Author: Phillip Muza
# Date: 08.09.26

"""Wires check_orientation.py / image_transformations.py / check_transform.py into main.py.

Raw LSM stacks are acquired in the transverse plane, in one of two directions (dorsal->ventral
or ventral->dorsal) that varies by animal. Which one applies cannot be told apart
algorithmically -- both directions produce the same small/big/small cross-sectional area
profile, so it takes a human glance at a contact sheet. That is the only manual step this
module leaves in place: main.py generates sheets for whatever hasn't been scored yet, refuses
to process anything until every animal in the batch has a direction on record (so the whole
dataset gets one QC pass rather than surprises trickling in animal by animal), then applies the
transform and a cross-animal verification grid automatically.

check_orientation.py, image_transformations.py and check_transform.py remain usable standalone
for one-off / manual work; this module reuses their side-effect-free functions rather than
duplicating them.
"""

import os
import csv
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_orientation import make_sheet
from image_transformations import read_stack, transformed_view, save_stack_streaming, DIRECTION_TO_TRANSFORM
from check_transform import read_planes

ORIENTATION_DIRNAME = "orientation_check"
TRANSFORMED_DIRNAME = "transformed"
ORIENTATIONS_CSV_NAME = "orientations.csv"

CSV_FIELDS = ["folder", "image", "n_slices", "sheet", "direction"]


def transformed_voxel_size(source_voxel_size):
    """Reorder (z, y, x) voxel size to match transformed_view's transpose(1, 0, 2).

    Both transform directions share the same leading transpose, so this reordering is the same
    regardless of which one was applied: axis 0 (z) and axis 1 (y) swap, axis 2 (x) is unchanged.
    Anything reading a transformed image (downsampling, registration) must use this, not the raw
    source_voxel_size, or the result comes out anisotropic -- stretched along one axis and
    squashed along another instead of isotropic.
    """
    z, y, x = source_voxel_size
    return np.array([y, z, x], dtype=float)


def pick_sample_image(signal_imgs, registration_source):
    """The channel used to represent an animal on the orientation/transform-check sheets."""
    match = next((img for img in signal_imgs if img.lower() == registration_source.lower()), None)
    return match or signal_imgs[0]


def _read_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return {}
    # utf-8-sig tolerates a BOM, which Excel adds when a user saves this CSV as "UTF-8"
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        return {row["folder"]: row for row in csv.DictReader(fh)}


def _write_csv_rows(csv_path, rows_by_folder):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for folder in sorted(rows_by_folder):
            writer.writerow(rows_by_folder[folder])


def check_orientation_status(parent_dir, animal_dirs, registration_source,
                              n_slices=10, margin=0.0, max_px=420):
    """Resolve every animal's transform status against orientation_check/orientations.csv.

    Already-transformed animals need no direction lookup. Everything else is looked up by its
    path relative to parent_dir; anything unscored (or missing/garbage in the direction column)
    gets its contact sheet (re)written, merging into the existing CSV so previously scored rows
    are never lost or overwritten.

    Returns (ready, unscored, csv_path):
      ready    -- {animal_dir: direction_or_None}; None means already transformed, nothing to do
      unscored -- animal_dirs still missing a usable direction, after writing their sheets
    """
    logger = logging.getLogger(__name__)
    output_dir = os.path.join(parent_dir, ORIENTATION_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, ORIENTATIONS_CSV_NAME)
    rows_by_folder = _read_csv_rows(csv_path)

    ready, unscored = {}, []
    for animal_dir, signal_imgs in animal_dirs:
        sample_image = pick_sample_image(signal_imgs, registration_source)
        transformed_path = os.path.join(animal_dir, TRANSFORMED_DIRNAME, sample_image)
        if os.path.exists(transformed_path):
            ready[animal_dir] = None
            continue

        folder = os.path.relpath(animal_dir, parent_dir)
        direction = (rows_by_folder.get(folder, {}).get("direction") or "").strip().lower()
        if direction in DIRECTION_TO_TRANSFORM:
            ready[animal_dir] = direction
            continue

        image_path = os.path.join(animal_dir, sample_image)
        if not os.path.exists(image_path):
            logger.warning(f"No {sample_image} in {animal_dir}, cannot check orientation; skipping")
            continue

        # relpath is "." when parent_dir is the animal directory itself
        label = folder if folder not in ("", ".") else             (os.path.basename(os.path.normpath(animal_dir)) or "root")
        safe_label = label.replace(os.sep, "__").replace(" ", "_")
        sheet_path = os.path.join(output_dir, f"{safe_label}.png")
        try:
            n_pages = make_sheet(image_path, label, sheet_path, n_slices, margin, max_px)
        except Exception as e:
            logger.error(f"Failed to write orientation sheet for {animal_dir}: {e}")
            # Still unscored -- block the run rather than silently skipping this animal
            unscored.append(animal_dir)
            continue

        rows_by_folder[folder] = {
            "folder": folder, "image": sample_image, "n_slices": n_pages,
            "sheet": os.path.basename(sheet_path),
            "direction": rows_by_folder.get(folder, {}).get("direction", ""),
        }
        logger.info(f"Wrote orientation sheet: {sheet_path}")
        unscored.append(animal_dir)

    if rows_by_folder:
        _write_csv_rows(csv_path, rows_by_folder)

    return ready, unscored, csv_path


def apply_orientation_transform(animal_dir, signal_imgs, direction):
    """Write a coronal copy of every raw signal image to animal_dir/transformed/.

    Raw images are never modified. Existing transformed copies are reused on re-runs.
    Returns {img_name: path_in_transformed_dir}.
    """
    logger = logging.getLogger(__name__)
    transformed_dir = os.path.join(animal_dir, TRANSFORMED_DIRNAME)
    os.makedirs(transformed_dir, exist_ok=True)
    transform_type = DIRECTION_TO_TRANSFORM[direction]

    transformed = {}
    for img_name in signal_imgs:
        destination = os.path.join(transformed_dir, img_name)
        if os.path.exists(destination):
            logger.info(f"Transformed image already exists, reusing: {destination}")
        else:
            img = read_stack(os.path.join(animal_dir, img_name))
            view = transformed_view(img, transform_type)
            # Write to a temporary name and rename, so an interrupted write cannot leave a
            # truncated file that the exists-check above would treat as complete
            partial = destination + ".partial"
            try:
                save_stack_streaming(view, partial)
                os.replace(partial, destination)
            finally:
                if os.path.exists(partial):
                    os.remove(partial)
            del img, view
            logger.info(f"Transformed ({direction}) {img_name} -> {destination}")
        transformed[img_name] = destination

    return transformed


def write_transform_check_grid(parent_dir, transformed_by_animal, registration_source,
                                n_slices=6, margin=0.08, max_px=300,
                                voxel_size=(6.55, 20, 6.55)):
    """One comparison grid across every animal transformed this run.

    Lets a wrong direction call be caught by eye before the expensive registration/segmentation
    steps run on it, without needing to open each animal's sheet individually.
    """
    logger = logging.getLogger(__name__)
    fractions = np.linspace(margin, 1 - margin, n_slices)

    rows = []
    for animal_dir, images in transformed_by_animal.items():
        sample_image = pick_sample_image(list(images.keys()), registration_source)
        image_path = images[sample_image]
        label = os.path.relpath(animal_dir, parent_dir)
        try:
            _, planes, n_pages = read_planes(image_path, fractions, max_px)
            rows.append((label, planes, n_pages))
        except Exception as e:
            logger.error(f"Transform check failed on {image_path}: {e}")

    if not rows:
        return None

    display_aspect = voxel_size[1] / voxel_size[2]
    n_cols = n_slices
    fig, axes = plt.subplots(len(rows), n_cols,
                              figsize=(2.3 * n_cols, 2.2 * len(rows)), facecolor="white")
    axes = np.atleast_2d(axes)

    for r, (label, planes, n_pages) in enumerate(rows):
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

    output_dir = os.path.join(parent_dir, ORIENTATION_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "transform_check.png")
    fig.savefig(output_path, dpi=95)
    plt.close(fig)
    logger.info(f"Wrote transform verification grid: {output_path}")
    return output_path
