r"""
Depth and front-to-back distribution of tracer *signal*, not of tracer voxels.

`../NS24122_analysis/NS24122_tracer_penetration_analysis.ipynb` asks how deep below the brain surface
and how far forward the tracer reaches, counting voxels of the pipeline's per-slice adaptive mask.
This script asks the same two questions of the amount of signal, using the fixed per-animal threshold
validated in fixed_threshold_intensity.py, so the answer does not depend on a mask whose threshold
moves with each slice's contrast.

Per animal x channel it accumulates 2D histograms over (atlas coronal plane x 20 um depth bin):
  counts     voxels above the fixed threshold                   -> the coverage-style readout
  intensity  their raw intensity above the animal's background  -> the amount readout
and, as the denominator for coverage curves, the same histogram for every tissue voxel of the brain.
Depth and position come from each animal's deformation fields, i.e. in atlas coordinates, exactly as
in the existing penetration notebook, so "1 mm deep" means the same anatomical place in every brain.

Signal that registration places just outside the atlas is kept, at its nearest atlas voxel, within
FILL_CAP_UM - the same convention as the pipeline (--no-fill drops it).

Run:
  python penetration_intensity.py --qc an4      # one animal, prints checks, writes nothing
  python penetration_intensity.py               # all animals -> pen_cache/<animal>.npz
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

from fixed_threshold_intensity import (PARENT, DATA_MAP, ATLAS_DIR, EXCLUDE_ANIMALS, TRACER_DIRS,
                                       K_MAD, DETECT, FILL_CAP_UM, VOXEL_UM, VOXEL_MM3,
                                       robust_background, tissue_mask, channel_threshold, read_channel)

HERE = os.path.dirname(os.path.abspath(__file__))
PEN_CACHE = os.path.join(HERE, 'pen_cache')
VOXEL_MM = VOXEL_UM / 1000
CLOSING_RADIUS_VOX = 5        # 100 um, as chosen for the ATX-7851 penetration analysis
DEPTH_MAX_MM = 3.0


def envelope_and_depth(mask, closing_radius=CLOSING_RADIUS_VOX, pad_posterior=100):
    """
    Depth (mm) below the brain's outer envelope for every atlas voxel.

    The envelope is the atlas brain mask with holes filled and a morphological closing that seals the
    midline fissure and the cortex-cerebellum gap, so tracer entering along those does not count as
    surface; the caudal cut face is extended so it is not treated as a surface either. Identical to
    the ATX-7851 penetration notebook.
    """
    r, pad = closing_radius, closing_radius + 2
    m = np.pad(ndimage.binary_fill_holes(mask), ((pad, pad_posterior + pad), (pad, pad), (pad, pad)))
    last = pad + np.flatnonzero(mask.any(axis=(1, 2)))[-1]
    m[last + 1:last + 1 + pad_posterior + r] = m[last]
    if r > 0:
        m = ndimage.distance_transform_edt(ndimage.distance_transform_edt(~m) <= r) > r
    m = ndimage.binary_fill_holes(m)
    depth = ndimage.distance_transform_edt(m).astype(np.float32) * VOXEL_MM
    crop = (slice(pad, pad + mask.shape[0]), slice(pad, pad + mask.shape[1]), slice(pad, pad + mask.shape[2]))
    return depth[crop]


def atlas_reference():
    """Atlas depth map, the front edge of the midbrain, and the caudal edge of the thalamus."""
    annotation = tifffile.imread(os.path.join(ATLAS_DIR, 'annotation.tiff'))
    import json
    with open(os.path.join(ATLAS_DIR, 'structures.json')) as f:
        path_of = {s['id']: s['structure_id_path'] for s in json.load(f)}
    depth = envelope_and_depth(annotation > 0)

    def first_plane(root_id):
        ids = [i for i, p in path_of.items() if root_id in p]
        return int(np.flatnonzero(np.isin(annotation, ids).any(axis=(1, 2)))[0])

    def last_plane(root_id):
        ids = [i for i, p in path_of.items() if root_id in p]
        return int(np.flatnonzero(np.isin(annotation, ids).any(axis=(1, 2)))[-1])

    return annotation, depth, first_plane(313), last_plane(549)     # MB, TH


def animal_histograms(animal, annotation, depth_atlas, ap_edges, depth_edges, fill=True, k=K_MAD):
    """
    2D histograms (atlas plane x depth) of tissue voxels and, per channel, of above-threshold voxels
    weighted by count and by above-background intensity.
    """
    reg_dir = os.path.join(PARENT, animal, 'registration_dir')
    labels = tifffile.imread(os.path.join(reg_dir, 'registered_atlas.tiff'))
    inside = labels > 0
    tissue, tissue_fraction = tissue_mask(animal, inside)
    measured = inside & tissue
    del labels

    # voxel sets to look up, and the weight carried by each voxel
    voxel_sets = {'brain': (np.flatnonzero(measured.ravel()), None)}
    meta = {'tissue_fraction': tissue_fraction}
    boundary = tree = None
    for tracer in TRACER_DIRS:
        raw, unsharp = read_channel(animal, tracer)
        detect_image = raw if DETECT == 'raw' else unsharp
        threshold, bg_detect, sd_detect = channel_threshold(detect_image, measured, k, DETECT)
        bg_raw, _ = robust_background(raw[measured].astype(np.float64))
        mask = (detect_image > threshold) & tissue
        above = np.maximum(raw.astype(np.float32) - bg_raw, 0)   # weights inside the mask are positive by construction

        inside_mask = mask & measured
        voxel_sets[f'{tracer}_in'] = (np.flatnonzero(inside_mask.ravel()), above.ravel()[np.flatnonzero(inside_mask.ravel())])
        if fill:
            outside = mask & ~inside
            if outside.any():
                if tree is None:
                    surface = inside & ~ndimage.binary_erosion(inside, border_value=1)
                    boundary = np.argwhere(surface)
                    tree = cKDTree(boundary)
                coords = np.argwhere(outside)
                dist, nearest = tree.query(coords, distance_upper_bound=FILL_CAP_UM / VOXEL_UM, workers=-1)
                keep = np.isfinite(dist)
                # counted at the nearest atlas voxel, so it lands in that voxel's depth/AP bin
                flat = np.ravel_multi_index(tuple(boundary[nearest[keep]].T), inside.shape)
                weights = above[tuple(coords[keep].T)]
                voxel_sets[f'{tracer}_shell'] = (flat, weights)
        meta.update({f'{tracer}_threshold': threshold, f'{tracer}_bg_raw': bg_raw,
                     f'{tracer}_n_in': int(inside_mask.sum()),
                     f'{tracer}_signal_total': float(above[inside_mask].sum())})
        del raw, unsharp, mask, above, inside_mask
    del inside, tissue, measured

    # atlas coordinate of every voxel of interest, one deformation field at a time
    coords = {key: [None] * 3 for key in voxel_sets}
    for axis in range(3):
        field = tifffile.imread(os.path.join(reg_dir, f'deformation_field_{axis}.tiff')).ravel()
        for key, (flat, _) in voxel_sets.items():
            coords[key][axis] = field[flat]
        del field

    result = {}
    for key, (flat, weights) in voxel_sets.items():
        values = coords[key]
        idx = [np.clip(np.rint(v / VOXEL_MM).astype(np.int32), 0, n - 1) for v, n in zip(values, annotation.shape)]
        ap, dep = values[0], depth_atlas[idx[0], idx[1], idx[2]]
        result[f'{key}_counts'] = np.histogram2d(ap, dep, bins=[ap_edges, depth_edges])[0]
        if weights is not None:
            result[f'{key}_intensity'] = np.histogram2d(ap, dep, bins=[ap_edges, depth_edges], weights=weights)[0]
    return result, meta


def run(animals, fill=True, k=K_MAD):
    annotation, depth_atlas, y_plane, thalamus_plane = atlas_reference()
    ap_edges = (np.arange(annotation.shape[0] + 1) - 0.5) * VOXEL_MM
    depth_edges = np.arange(0, DEPTH_MAX_MM + VOXEL_MM / 2, VOXEL_MM)
    os.makedirs(PEN_CACHE, exist_ok=True)
    np.savez(os.path.join(PEN_CACHE, 'reference.npz'), y_plane=y_plane, thalamus_plane=thalamus_plane,
             ap_edges=ap_edges, depth_edges=depth_edges,
             atlas_per_plane=(annotation > 0).sum(axis=(1, 2)).astype(np.float64))
    print(f'front edge of the midbrain: plane {y_plane} ({y_plane * VOXEL_MM:.2f} mm); '
          f'caudal thalamus plane {thalamus_plane} ({thalamus_plane * VOXEL_MM:.2f} mm)')

    meta_rows, start = [], time.time()
    for animal in animals:
        t0 = time.time()
        hist, meta = animal_histograms(animal, annotation, depth_atlas, ap_edges, depth_edges, fill, k)
        np.savez_compressed(os.path.join(PEN_CACHE, f'{animal}.npz'), **hist)
        meta_rows.append({'animal_number': animal, **meta})
        print(f'{animal}: {time.time() - t0:.0f} s', flush=True)
    pd.DataFrame(meta_rows).to_csv(os.path.join(HERE, 'penetration_intensity_meta.csv'), index=False)
    print(f'\nwrote {len(animals)} histograms to {PEN_CACHE} in {(time.time() - start) / 60:.1f} min')


def qc(animal):
    """One animal: do the intensity-weighted profiles behave, and how do they compare with counts?"""
    annotation, depth_atlas, y_plane, thalamus_plane = atlas_reference()
    ap_edges = (np.arange(annotation.shape[0] + 1) - 0.5) * VOXEL_MM
    depth_edges = np.arange(0, DEPTH_MAX_MM + VOXEL_MM / 2, VOXEL_MM)
    t0 = time.time()
    hist, meta = animal_histograms(animal, annotation, depth_atlas, ap_edges, depth_edges)
    print(f'{animal}: histograms in {time.time() - t0:.0f} s')
    print(f"  tissue fraction of atlas {meta['tissue_fraction']:.3f}")

    brain = hist['brain_counts']
    for tracer in TRACER_DIRS:
        counts = hist[f'{tracer}_in_counts'] + hist.get(f'{tracer}_shell_counts', 0)
        signal = hist[f'{tracer}_in_intensity'] + hist.get(f'{tracer}_shell_intensity', 0)
        print(f'\n--- {tracer}: threshold {meta[f"{tracer}_threshold"]:.0f}, background {meta[f"{tracer}_bg_raw"]:.0f}')
        print(f'  voxels above threshold inside the atlas: {meta[f"{tracer}_n_in"]:,}; '
              f'binned (with shell) {counts.sum():,.0f}')
        print(f'  signal inside the atlas {meta[f"{tracer}_signal_total"]:.3e}; binned (with shell) {signal.sum():.3e}')
        by_depth_c, by_depth_s = counts.sum(axis=0), signal.sum(axis=0)
        deep = depth_edges[:-1] >= 1.0 - 1e-9
        print(f'  deeper than 1 mm: {100 * by_depth_c[deep].sum() / by_depth_c.sum():.2f}% of voxels, '
              f'{100 * by_depth_s[deep].sum() / by_depth_s.sum():.2f}% of signal')
        by_plane_c, by_plane_s = counts.sum(axis=1), signal.sum(axis=1)
        print(f'  anterior to the midbrain: {100 * by_plane_c[:y_plane].sum() / by_plane_c.sum():.2f}% of voxels, '
              f'{100 * by_plane_s[:y_plane].sum() / by_plane_s.sum():.2f}% of signal')
        mean_depth_c = (by_depth_c * depth_edges[:-1]).sum() / by_depth_c.sum()
        mean_depth_s = (by_depth_s * depth_edges[:-1]).sum() / by_depth_s.sum()
        print(f'  mean depth: {mean_depth_c:.3f} mm by voxel, {mean_depth_s:.3f} mm by signal '
              '(signal-weighted should sit shallower if bright voxels are superficial)')
    print(f'\nbrain voxels binned: {brain.sum():,.0f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qc', metavar='ANIMAL', help='check one animal, write nothing')
    parser.add_argument('--no-fill', action='store_true', help='drop signal lying outside the registered atlas')
    parser.add_argument('--k', type=float, default=K_MAD, help=f'threshold in robust SDs above background (default {K_MAD:g})')
    args = parser.parse_args()

    if args.qc:
        qc(args.qc)
        return
    data_map = (pd.read_csv(DATA_MAP, encoding='utf-8-sig')
                  .rename(columns={'blinded_number': 'animal_number'}).set_index('animal_number'))
    animals = sorted([a for a in os.listdir(PARENT)
                      if a.startswith('an') and os.path.isdir(os.path.join(PARENT, a, 'registration_dir'))
                      and a not in EXCLUDE_ANIMALS], key=lambda a: int(a[2:]))
    assert all(a in data_map.index for a in animals)
    run(animals, fill=not args.no_fill, k=args.k)


if __name__ == '__main__':
    main()
