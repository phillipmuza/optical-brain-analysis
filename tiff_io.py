# Title: Lazy plane access for large TIFF stacks
# Author: Phillip Muza
# Date: 09/09/26

"""Read individual planes from a 3D TIFF without loading the whole stack.

The page count is not a reliable depth. ImageJ writes stacks past ~4 GB as a single IFD
followed by all the plane data laid out contiguously, so a 634-slice MesoSPIM stack reports
len(tif.pages) == 1 while tif.series[0].shape is (634, 2048, 2048). Depth therefore comes from
the series, and planes are memory-mapped where the data is contiguous, falling back to
per-page reads for stacks written one IFD per plane.
"""

import contextlib

import numpy as np
import tifffile


@contextlib.contextmanager
def open_stack(image_path):
    """Yield (n_planes, read_plane) for a 3D TIFF, reading only the planes asked for.

    read_plane(idx) returns plane idx along axis 0 as a numpy array.
    """
    with tifffile.TiffFile(image_path) as tif:
        shape = tuple(tif.series[0].shape)
        if len(shape) != 3:
            raise ValueError(f"{image_path} has shape {shape}; expected a 3D stack")

        n_planes = shape[0]
        if n_planes < 2:
            raise ValueError(f"{image_path} has {n_planes} plane(s); not a stack")

        if len(tif.pages) == n_planes:
            def read_plane(idx):
                return tif.pages[int(idx)].asarray()

            yield n_planes, read_plane
            return

    # One IFD covering many planes (ImageJ contiguous stack): index the mapped data instead.
    memmap = tifffile.memmap(image_path, mode='r')
    try:
        if memmap.shape != shape:
            raise ValueError(f"{image_path}: memory map is {memmap.shape}, expected {shape}")

        def read_plane(idx):
            return np.asarray(memmap[int(idx)])

        yield n_planes, read_plane
    finally:
        del memmap
