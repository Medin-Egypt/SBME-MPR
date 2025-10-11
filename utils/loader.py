# loader.py

from scipy.ndimage import map_coordinates
import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import nibabel as nib
import os

def load_dicom_data(folder_path):
    """
    Loads a DICOM series, correctly handling orientation, spacing, and intensity windowing.
    """
    try:
        if os.path.isdir(folder_path):
            dicom_files = [os.path.join(folder_path, f)
                           for f in os.listdir(folder_path)
                           if f.lower().endswith('.dcm')]
        else:
            dicom_files = [folder_path]

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {folder_path}")

        slices = []
        for f in dicom_files:
            try:
                ds = dcmread(f)
                if hasattr(ds, 'pixel_array') and hasattr(ds, 'ImagePositionPatient'):
                    slices.append(ds)
            except Exception:
                continue

        if not slices:
            raise ValueError("No readable DICOM slices with pixel data and position info found.")

        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

        text_data = []
        for field in ["BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName"]:
            value = slices[0].get(field)
            if value:
                text_data.append(field + ": " + str(value).upper())

        image_stack = np.stack([s.pixel_array * float(getattr(s, 'RescaleSlope', 1)) +
                                float(getattr(s, 'RescaleIntercept', 0))
                                for s in slices])

        # Transpose from (Z, Y, X) to (X, Y, Z)
        image_stack = image_stack.transpose((2, 1, 0))

        # --- FIX: Flip the volume along the X-axis ---
        # This mirrors the sagittal view to face the other direction and
        # corrects the horizontal orientation of the axial view.
        image_stack = image_stack[::, ::-1, ::]

        pixel_spacing = [float(x) for x in slices[0].PixelSpacing]
        if len(slices) > 1:
            pos1 = np.array(slices[0].ImagePositionPatient)
            pos2 = np.array(slices[1].ImagePositionPatient)
            slice_spacing = np.linalg.norm(pos2 - pos1)
        else:
            slice_spacing = float(getattr(slices[0], 'SliceThickness', 1.0))

        affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_spacing, 1])

        if hasattr(slices[0], 'WindowCenter') and hasattr(slices[0], 'WindowWidth'):
            center = float(slices[0].WindowCenter[0] if isinstance(slices[0].WindowCenter, pydicom.multival.MultiValue) else slices[0].WindowCenter)
            width = float(slices[0].WindowWidth[0] if isinstance(slices[0].WindowWidth, pydicom.multival.MultiValue) else slices[0].WindowWidth)
            intensity_min = center - width / 2
            intensity_max = center + width / 2
        else:
            foreground_pixels = image_stack[image_stack > image_stack.min()]
            if foreground_pixels.size > 0:
                intensity_min = np.percentile(foreground_pixels, 1)
                intensity_max = np.percentile(foreground_pixels, 99)
            else:
                intensity_min, intensity_max = image_stack.min(), image_stack.max()

        dims = image_stack.shape
        return image_stack, affine, dims, intensity_min, intensity_max, text_data
    except Exception as e:
        print(f"Error loading DICOM data: {e}")
        return None, None, None, 0, 1, None


def load_nifti_data(file_path):
    """
    Loads NIfTI data, flips it horizontally to correct mirroring, and returns it.
    """
    try:
        nifti_file = nib.load(file_path)
        data = nifti_file.get_fdata()
        affine = nifti_file.affine

        # --- FIX: Flip the volume along the X-axis ---
        # This corrects left-right mirroring issues, such as the heart
        # appearing on the right side of the scan.
        data = data[::-1, :, :]

        if data.ndim < 3:
            raise ValueError(f"Data has fewer than 3 dimensions ({data.ndim}).")

        dims = data.shape

        # Calculate intensity window after flipping, focusing on foreground pixels
        # for better contrast.
        foreground_pixels = data[data > np.min(data)]
        if foreground_pixels.size > 0:
            intensity_min = np.percentile(foreground_pixels, 1)
            intensity_max = np.percentile(foreground_pixels, 99)
        else:
            # Fallback in case there are no foreground pixels
            intensity_min, intensity_max = np.min(data), np.max(data)

        return data, affine, dims, intensity_min, intensity_max

    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None, None, None, 0, 1


def get_slice_data(data, dims, slices, affine, intensity_min=0, intensity_max=1000, rot_x_deg=0, rot_y_deg=0, view_type='axial'):
    # ... (This function is unchanged)
    if data is None:
        return np.zeros((10, 10), dtype=np.uint8)

    if view_type == 'axial':
        slice_data = np.flipud(np.rot90(data[:, :, slices['axial']]))
        x_spacing, y_spacing = affine[0, 0], affine[1, 1]
    elif view_type == 'coronal':
        slice_data = np.rot90(data[:, slices['coronal'], :])
        x_spacing, y_spacing = affine[0, 0], affine[2, 2]
    elif view_type == 'sagittal':
        slice_data = np.rot90(data[slices['sagittal'], :, :])
        x_spacing, y_spacing = affine[1, 1], affine[2, 2]
    elif view_type == 'oblique':
        slice_data = _get_oblique_slice(data, rot_x_deg, rot_y_deg, slices['oblique'])
        x_spacing, y_spacing = 1, 1
    else:
        return np.zeros((10, 10), dtype=np.uint8)

    if slice_data.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    if x_spacing > 0 and y_spacing > 0:
        aspect_ratio = y_spacing / x_spacing
        new_height = int(slice_data.shape[0] * aspect_ratio)
        if new_height > 0:
            y_coords = np.linspace(0, slice_data.shape[0] - 1, new_height)
            x_coords = np.arange(slice_data.shape[1])
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

            resampled_slice = map_coordinates(slice_data, [yy, xx], order=1, mode='constant', cval=slice_data.min())
            slice_data = resampled_slice

    if intensity_max > intensity_min:
        slice_data = np.clip(slice_data, intensity_min, intensity_max)
        slice_data = 255 * (slice_data - intensity_min) / (intensity_max - intensity_min)
    else:
        slice_data = np.zeros_like(slice_data)

    return slice_data.astype(np.uint8)


def _get_oblique_slice(data, rot_x_deg, rot_y_deg, slice_idx):
    # ... (This function is unchanged)
    center_voxel = np.array(data.shape) / 2.0
    slice_dim = int(np.linalg.norm(data.shape))

    theta_x = np.deg2rad(rot_x_deg)
    theta_y = np.deg2rad(rot_y_deg)
    rot_x_mat = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y_mat = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    transform_mat = rot_y_mat @ rot_x_mat

    u_vec = transform_mat @ np.array([1, 0, 0])
    v_vec = transform_mat @ np.array([0, 1, 0])
    w_vec = transform_mat @ np.array([0, 0, 1])

    x_range = np.arange(-slice_dim / 2, slice_dim / 2)
    y_range = np.arange(-slice_dim / 2, slice_dim / 2)
    xx, yy = np.meshgrid(x_range, y_range)

    slice_offset = slice_idx - (int(np.linalg.norm(data.shape)) / 2)
    points_3d = center_voxel[:, np.newaxis] \
                + xx.ravel() * u_vec[:, np.newaxis] \
                + yy.ravel() * v_vec[:, np.newaxis] \
                + slice_offset * w_vec[:, np.newaxis]

    coords = [points_3d[0], points_3d[1], points_3d[2]]
    oblique_slice = map_coordinates(data, coords, order=1, cval=data.min(), mode='constant')
    return oblique_slice.reshape((slice_dim, slice_dim))