from pathlib import Path

import nibabel as nib
from dipy.align import resample
from nibabel.processing import resample_from_to

current_path = Path(__file__).absolute().parent

base_data = nib.load(current_path / "data" / "base.mgz")
overlay_data = nib.load(current_path / "data" / "overlay.nii.gz")

# overlay_data = resample(overlay_data, base_data, static_affine=base_data.affine)

overlay_data = resample_from_to(overlay_data, base_data)

orientation_map = {
    'axial': 2,
    'sagittal': 0,
    'coronal': 1
}

# We get the voxel size from the base_data
VOXEL_SIZE = base_data.header.get_zooms()[0]
