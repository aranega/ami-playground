import io
from base64 import b64encode

import mne as mne
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from nibabel.spatialimages import SpatialImage
from dipy.align.imaffine import AffineMap

from backend.helpers.transforms_helpers import get_affine_matrix
from backend.settings import orientation_map, base_data, VOXEL_SIZE


base = nib.orientations.apply_orientation(
    np.asarray(base_data.dataobj), nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(base_data.affine))).astype(np.float32)


## Util, to remove
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
##


def get_slices(V, slice_indices=None):
    if slice_indices is None:
        slice_indices = np.array(V.shape) // 2

    # Normalize the intensities to [0, 255]
    V = np.asarray(V, dtype=np.float64)
    V = 255 * (V - V.min()) / (V.max() - V.min())

    # Extract the middle slices
    axial = np.asarray(V[:, :, slice_indices[2]]).astype(np.uint8).T
    coronal = np.asarray(V[:, slice_indices[1], :]).astype(np.uint8).T
    sagittal = np.asarray(V[slice_indices[0], :, :]).astype(np.uint8).T

    return axial, coronal, sagittal
    # return sagittal, coronal, axial


def overlay_slices(L, R):
    la, lc, ls = get_slices(L)
    ra, rc, rs = get_slices(R)
    return ((la, ra), (lc, rc), (ls, rs))

    # sh = L.shape
    # c1 = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    # c2 = np.zeros(shape=(sh[2], sh[0], 3), dtype=np.uint8)
    # c3 = np.zeros(shape=(sh[1], sh[0], 3), dtype=np.uint8)

    # c1[..., 0] = la * (la > la[0, 0])
    # c1[..., 1] = la * (la > la[0, 0])
    # c1[..., 2] = la * (la > la[0, 0])

    # # c1[..., 0] = ra * (ra > ra[0, 0])
    # # c1[..., 1] = ra * (ra > ra[0, 0])
    # c1[..., 2] = ra * (ra > ra[0, 0])

    # c2[..., 0] = lc * (lc > lc[0, 0])
    # c2[..., 1] = lc * (lc > lc[0, 0])
    # c2[..., 2] = lc * (lc > lc[0, 0])

    # # c2[..., 0] = rc * (rc > rc[0, 0])
    # c2[..., 1] = rc * (rc > rc[0, 0])
    # # c2[..., 2] = rc * (rc > rc[0, 0])

    # c3[..., 0] = ls * (ls > ls[0, 0])
    # c3[..., 1] = ls * (ls > ls[0, 0])
    # c3[..., 2] = ls * (ls > ls[0, 0])

    # c3[..., 0] = rs * (rs > rs[0, 0])

    # c3[..., 1] = rs * (rs > rs[0, 0])
    # c3[..., 2] = rs * (rs > rs[0, 0])

    # c1[..., 0] = la * (la > la[0, 0])
    # c1[..., 1] = ra * (ra > ra[0, 0])

    # c2[..., 0] = lc * (lc > lc[0, 0])
    # c2[..., 1] = rc * (rc > rc[0, 0])

    # c3[..., 0] = ls * (ls > ls[0, 0])
    # c3[..., 1] = rs * (rs > rs[0, 0])

    # c1 = 0.5 * la  + 0.5 * ra
    # c2 = 0.5 * lc  + 0.5 * rc
    # c3 = 0.5 * ls  + 0.5 * rs

    # return c1, c2, c3




@timeit
def get_image(overlay, alpha=0.5, size=25.4):
    overlay = nib.orientations.apply_orientation(
        np.asarray(overlay.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(overlay.affine))).astype(np.float32)
    c1, c2, c3 = overlay_slices(base, overlay)

    # Save the figure to a BytesIO object
    buffers = [io.BytesIO(), io.BytesIO(), io.BytesIO()]
    img_size = len(base)
    voxel_sizes = base_data.header.get_zooms()
    orientations = [
        (1, 0),  # Axial: y/x
        (1, 2),  # Coronal: y/z
        (0, 2),  # Sagittal: x/z
    ]
    for i, (buf, (l, r)) in enumerate(zip(buffers, (c1, c2, c3))):

        # get the voxel size depending on the orientation
        a0, a1 = orientations[i]
        voxel_size_a, voxel_size_b = voxel_sizes[a0], voxel_sizes[a1]

        # compute the fig size
        figsize = (img_size * voxel_size_a / size, img_size * voxel_size_b / size)

        # create the fig and add the two planes
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(l, cmap="gray")
        ax.imshow(r, cmap="gist_heat", alpha=alpha)
        ax.invert_yaxis()
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(buf)

    return [b64encode(buf.getvalue()).decode('utf-8') for buf in buffers]




@timeit
def get_aligned_overlay(overlay, transformations, init):

    shape = overlay.shape
    affine = overlay.affine
    # overlay = nib.orientations.apply_orientation(
    #     np.asarray(overlay.dataobj), nib.orientations.axcodes2ornt(
    #         nib.orientations.aff2axcodes(overlay.affine))).astype(np.float32)

    matrix = init
    for transfo, axis in transformations.items():
        matrix = matrix @ get_affine_matrix(base_data, transfo, **axis)

    print(matrix)
    affine_map = AffineMap(matrix,
                           shape, affine,
                           base_data.dataobj.shape, base_data.affine)
    reg_data = affine_map.transform(base_data.get_fdata(), interpolation='linear')

    img = SpatialImage(reg_data, affine)

    return img