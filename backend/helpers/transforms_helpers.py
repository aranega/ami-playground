import numpy as np


def rotation(x=0, y=0, z=0):
    """
    Takes an angle (in radian) for each axis,
    Returns the corresponding 3D rotation matrix.
    """
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
    """
    Takes a scalar value or a 3-element array of scaling factors as input for each axis.
    Returns the corresponding 3D scaling matrix.
    """
    s = np.array([[1/x, 0, 0, 0],
                  [0, 1/y, 0, 0],
                  [0, 0, 1/z, 0],
                  [0, 0, 0, 1]], dtype=float)
    return s


def translation(x=0, y=0, z=0):
    """
    Takes a 3-element array of translation values as input for each axis.
    Returns the corresponding 3D translation matrix.
    """
    m = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return m


def translation_valuemapping(value, axis, img):
    # Calculate the physical dimensions of the object in each axis
    axis_index = {'x': 0, 'y': 1, 'z': 2}
    voxel_size = img.header.get_zooms()[axis_index[axis]]
    num_voxels = len(img.get_fdata())
    dimensions = num_voxels * voxel_size

    # Calculate the effective value in distance units
    effective_value = dimensions * value / 100.0
    return -effective_value  # the value is negative here as we turn the y axis for display


transfos = {
    'rotation': (
        rotation,  # Matrix production function (signature (x, y, z))
        lambda value, *_: np.deg2rad(value)  # value conversion function (signature (value, axis: str, img: Niftii image))
    ),
    'scale': (
        scaling,
        lambda value, *_: value
    ),
    'position': (
        translation,
        translation_valuemapping
    )
}


def get_affine_matrix(img, transfo_type, **kwargs):
    try:
        T, mapping = transfos[transfo_type]
        args = {k: mapping(v, k, img) for k, v in kwargs.items()}
        return T(**args)
    except KeyError:
        raise ValueError(f"Invalid transform type: {transfo_type}")

