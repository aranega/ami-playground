import numpy as np

from backend.settings import VOXEL_SIZE

# copied from mne

def rotation(x=0, y=0, z=0):
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    cos_z = np.cos(z)
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    sin_z = np.sin(z)
    r = np.array([[cos_y * cos_z, -cos_x * sin_z + sin_x * sin_y * cos_z,
                   sin_x * sin_z + cos_x * sin_y * cos_z, 0],
                  [cos_y * sin_z, cos_x * cos_z + sin_x * sin_y * sin_z,
                   - sin_x * cos_z + cos_x * sin_y * sin_z, 0],
                  [-sin_y, sin_x * cos_y, cos_x * cos_y, 0],
                  [0, 0, 0, 1]], dtype=float)
    return r


def scaling(x=1, y=1, z=1):
    s = np.array([[x, 0, 0, 0],
                  [0, y, 0, 0],
                  [0, 0, z, 0],
                  [0, 0, 0, 1]], dtype=float)
    return s


def translation(x=0, y=0, z=0):
    m = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return m


transfos = {
    'rotation': rotation,
    'scale': scaling,
    'position': translation,  # is not using yet _get_effective_value, just to test here
}

def get_affine_matrix(transform_type, **kwargs):
    try:
        return transfos[transform_type](**kwargs)
    except KeyError:
        raise ValueError(f"Invalid transform type: {transform_type}")


def _get_effective_value(voxels, percentage_value, voxel_size=VOXEL_SIZE):
    # Calculate the physical dimensions of the object in each axis
    num_voxels = len(voxels)
    dimensions = num_voxels * voxel_size

    # Calculate the effective value in distance units
    effective_value = dimensions * percentage_value / 100.0

    return effective_value
