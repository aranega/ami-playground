import base64
import io

import mne as mne
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from nibabel.spatialimages import SpatialImage

from backend.helpers.transforms_helpers import get_affine_matrix
from backend.settings import orientation_map, base_data, VOXEL_SIZE

base = nib.orientations.apply_orientation(
    np.asarray(base_data.dataobj), nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(base_data.affine))).astype(np.float32)


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


@timeit
def _generate_image(overlay, orientation, alpha=0.5):
    """Define a helper function for comparing plots."""

    overlay = nib.orientations.apply_orientation(
        np.asarray(overlay.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(overlay.affine))).astype(np.float32)

    # Set the physical size of each voxel (in millimeters)
    voxel_size = VOXEL_SIZE

    image_size = len(base)

    # Calculate the appropriate figsize in inches
    # figsize = (image_size * voxel_size / 25.4, image_size * voxel_size / 25.4)

    fig, ax = plt.subplots(1, 1)
    i = orientation_map[orientation]
    ax.imshow(np.take(base, [base.shape[i] // 2], axis=i).squeeze().T,
              cmap='gray')
    ax.imshow(np.take(overlay, [overlay.shape[i] // 2],
                      axis=i).squeeze().T, cmap='gist_heat', alpha=alpha)
    ax.invert_yaxis()
    ax.axis('off')

    fig.tight_layout()
    return fig


from dipy.viz import regtools


@timeit
def overlay_slices(L, R, slice_index=None, slice_type=1, fname=None, **fig_kwargs):
    # Normalize the intensities to [0,255]
    sh = L.shape
    L = np.asarray(L, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    L = 255 * (L - L.min()) / (L.max() - L.min())
    R = 255 * (R - R.min()) / (R.max() - R.min())

    # Create the color image to draw the overlapped slices into, and extract
    # the slices (note the transpositions)
    if slice_type == 0:
        if slice_index is None:
            slice_index = sh[0] // 2
        colorImage = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
        ll = np.asarray(L[slice_index, :, :]).astype(np.uint8).T
        rr = np.asarray(R[slice_index, :, :]).astype(np.uint8).T
    elif slice_type == 1:
        if slice_index is None:
            slice_index = sh[1] // 2
        colorImage = np.zeros(shape=(sh[2], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, slice_index, :]).astype(np.uint8).T
        rr = np.asarray(R[:, slice_index, :]).astype(np.uint8).T
    elif slice_type == 2:
        if slice_index is None:
            slice_index = sh[2] // 2
        colorImage = np.zeros(shape=(sh[1], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, :, slice_index]).astype(np.uint8).T
        rr = np.asarray(R[:, :, slice_index]).astype(np.uint8).T
    else:
        print("Slice type must be 0, 1 or 2.")
        return

    # Draw the intensity images to the appropriate channels of the color image
    # The "(ll > ll[0, 0])" condition is just an attempt to eliminate the
    # background when its intensity is not exactly zero (the [0,0] corner is
    # usually background)
    colorImage[..., 0] = ll * (ll > ll[0, 0])
    colorImage[..., 1] = rr * (rr > rr[0, 0])

    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    ax.imshow(colorImage, cmap=plt.cm.gray, origin='lower')

    # Save the figure to disk, if requested
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', **fig_kwargs)

    return fig

# @timeit
# def _generate_image(overlay, orientation, alpha=0.5):
#     img = SpatialImage(overlay.dataobj, overlay.affine)
#     return img.orthoview().figs


# @timeit
# def get_image(overlay, orientation, alpha=0.5):
#     fig = _generate_image(overlay, orientation, alpha)
#     # fig = fig[0]
#     # Save the figure to a BytesIO object
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', transparent=True)
#     buf.seek(0)
#     plt.close('all')
#     return base64.b64encode(buf.getvalue()).decode('utf-8')


# @timeit
def get_image(overlay, orientation, alpha=0.5):
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    i = orientation_map[orientation]
    overlay = nib.orientations.apply_orientation(
        np.asarray(overlay.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(overlay.affine))).astype(np.float32)
    overlay_slices(base, overlay, slice_type=i, fname=buf)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@timeit
def get_aligned_overlay(overlay, transform, axis, value):

    from dipy.align.imaffine import AffineMap

    affine_map = AffineMap(get_affine_matrix(transform, **{axis: value}),
                           overlay.dataobj.shape, overlay.affine,
                           base_data.dataobj.shape, base_data.affine)
    reg_data = affine_map.transform(np.array(base_data.dataobj), interpolation='linear')
    img = SpatialImage(reg_data, overlay.affine)

    return img
    # return mne.transforms.apply_volume_registration(base_data, overlay,
                                                    # get_affine_matrix(transform, **{axis: value}),
                                                    # cval='1%')
