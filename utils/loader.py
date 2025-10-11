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

    Args:
        folder_path (str): Path to the DICOM folder or file.

    Returns:
        tuple: (data, affine, dims, intensity_min, intensity_max)
    """
    try:
        # 1️⃣ Collect all DICOM files
        if os.path.isdir(folder_path):
            dicom_files = [os.path.join(folder_path, f)
                           for f in os.listdir(folder_path)
                           if f.lower().endswith('.dcm')]
        else:
            dicom_files = [folder_path]

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {folder_path}")

        # 2️⃣ Read all slices and filter for readable ones with pixel data
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

        # FIX #3: Sort slices by their 3D position, not just instance number
        # This is far more robust for correct anatomical ordering.
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

        text_data = []
        for field in ["BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName"]:
            value = slices[0].get(field)
            if value:
                text_data.append(field + ": " + str(value).upper())

        # 3️⃣ Apply Rescale Slope/Intercept and stack slices
        # It's important to apply these *before* windowing to get Hounsfield Units (HU) or other quantitative values.
        image_stack = np.stack([s.pixel_array * float(getattr(s, 'RescaleSlope', 1)) +
                                float(getattr(s, 'RescaleIntercept', 0))
                                for s in slices])

        # FIX #1: Correct the data orientation
        # The raw stack is often (Z, Y, X). Transposing to (X, Y, Z) is a common
        # convention that matches tools like ITK-SNAP and aligns with NIfTI loaders.
        image_stack = image_stack.transpose((2, 1, 0))


        # 4️⃣ Calculate the affine matrix correctly
        pixel_spacing = [float(x) for x in slices[0].PixelSpacing]

        # FIX #3: Calculate slice spacing from ImagePositionPatient for accuracy
        if len(slices) > 1:
            pos1 = np.array(slices[0].ImagePositionPatient)
            pos2 = np.array(slices[1].ImagePositionPatient)
            slice_spacing = np.linalg.norm(pos2 - pos1)
        else:
            # Fallback if there's only one slice
            slice_spacing = float(getattr(slices[0], 'SliceThickness', 1.0))

        affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_spacing, 1])


        # 5️⃣ FIX #2: Smarter intensity windowing
        # First, try to use the WindowCenter and WindowWidth from the DICOM header.
        # This is what medical viewers use and is usually the intended display setting.
        if hasattr(slices[0], 'WindowCenter') and hasattr(slices[0], 'WindowWidth'):
            center = float(slices[0].WindowCenter[0] if isinstance(slices[0].WindowCenter, pydicom.multival.MultiValue) else slices[0].WindowCenter)
            width = float(slices[0].WindowWidth[0] if isinstance(slices[0].WindowWidth, pydicom.multival.MultiValue) else slices[0].WindowWidth)
            intensity_min = center - width / 2
            intensity_max = center + width / 2
        else:
            # Fallback to percentiles, but on the foreground pixels to avoid
            # the influence of background air, which can skew the result.
            foreground_pixels = image_stack[image_stack > image_stack.min()]
            if foreground_pixels.size > 0:
                intensity_min = np.percentile(foreground_pixels, 1)
                intensity_max = np.percentile(foreground_pixels, 99)
            else: # Handle edge case of a completely uniform image
                intensity_min, intensity_max = image_stack.min(), image_stack.max()


        dims = image_stack.shape

        return image_stack, affine, dims, intensity_min, intensity_max, text_data

    except Exception as e:
        print(f"Error loading DICOM data: {e}")
        return None, None, None, 0, 1, None


def load_nifti_data(file_path):
    """
    Loads NIfTI data, calculates intensity range, and returns data and metadata.

    Args:
        file_path (str): Path to the .nii or .nii.gz file.

    Returns:
        tuple: (data, affine, dims, intensity_min, intensity_max) or None on error.
    """
    try:
        nifti_file = nib.load(file_path)
        data = nifti_file.get_fdata()
        affine = nifti_file.affine

        if data.ndim < 3:
            raise ValueError(f"Data has fewer than 3 dimensions ({data.ndim}).")

        dims = data.shape
        # Use percentiles for robust min/max intensity windowing
        intensity_min = np.percentile(data, 1)
        intensity_max = np.percentile(data, 99)

        return data, affine, dims, intensity_min, intensity_max

    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None, None, None, 0, 1


def get_slice_data(data, dims, slices, affine, intensity_min=0, intensity_max=1000, rot_x_deg=0, rot_y_deg=0, view_type='axial'):
    """
    Generates a slice, correcting for aspect ratio, from the 3D volume.
    Args:
        data (np.ndarray): 3D volume data.
        dims (tuple): Dimensions of the data.
        slices (dict): Dictionary of current slice indices.
        affine (np.ndarray): Affine transformation matrix for the data.
        intensity_min (float): Minimum intensity for windowing.
        intensity_max (float): Maximum intensity for windowing.
        rot_x_deg (int): Oblique rotation around X-axis.
        rot_y_deg (int): Oblique rotation around Y-axis.
        view_type (str): 'axial', 'coronal', 'sagittal', or 'oblique'.
    Returns:
        np.ndarray: The 8-bit, aspect-ratio-corrected grayscale slice.
    """
    if data is None:
        return np.zeros((10, 10), dtype=np.uint8)

    # Determine the slice and the relevant pixel/slice spacing for it
    if view_type == 'axial':
        slice_data = np.rot90(data[:, :, slices['axial']])
        x_spacing, y_spacing = affine[0, 0], affine[1, 1]
    elif view_type == 'coronal':
        slice_data = np.rot90(data[:, slices['coronal'], :])
        x_spacing, y_spacing = affine[0, 0], affine[2, 2]
    elif view_type == 'sagittal':
        slice_data = np.rot90(data[slices['sagittal'], :, :])
        x_spacing, y_spacing = affine[1, 1], affine[2, 2]
    elif view_type == 'oblique':
        # Oblique slices are harder to correct perfectly without a full 3D resample.
        # For now, we'll treat them as having 1:1 aspect ratio, but a more
        # advanced implementation might resample the whole volume first.
        slice_data = _get_oblique_slice(data, rot_x_deg, rot_y_deg, slices['oblique'])
        x_spacing, y_spacing = 1, 1
    else:
        return np.zeros((10, 10), dtype=np.uint8)

    if slice_data.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    # --- Aspect Ratio Correction ---
    if x_spacing > 0 and y_spacing > 0:
        aspect_ratio = y_spacing / x_spacing
        new_height = int(slice_data.shape[0] * aspect_ratio)
        if new_height > 0:
            # Create a grid of coordinates for the target image
            y_coords = np.linspace(0, slice_data.shape[0] - 1, new_height)
            x_coords = np.arange(slice_data.shape[1])
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

            # Sample the original slice data at the new coordinates
            # order=1 is linear interpolation, order=0 is nearest-neighbor
            resampled_slice = map_coordinates(slice_data, [yy, xx], order=1, mode='constant', cval=slice_data.min())
            slice_data = resampled_slice

    # --- Intensity Windowing and Normalization ---
    if intensity_max > intensity_min:
        slice_data = np.clip(slice_data, intensity_min, intensity_max)
        slice_data = 255 * (slice_data - intensity_min) / (intensity_max - intensity_min)
    else:
        slice_data = np.zeros_like(slice_data)

    return slice_data.astype(np.uint8)


def _get_oblique_slice(data, rot_x_deg, rot_y_deg, slice_idx):
    """
    Internal function to generate an oblique slice using coordinate transformation.
    """
    center_voxel = np.array(data.shape) / 2.0
    slice_dim = int(np.linalg.norm(data.shape))

    # 1. Create rotation matrices
    theta_x = np.deg2rad(rot_x_deg)
    theta_y = np.deg2rad(rot_y_deg)
    rot_x_mat = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y_mat = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    transform_mat = rot_y_mat @ rot_x_mat

    # 2. Define the basis vectors and rotate them
    u_vec = transform_mat @ np.array([1, 0, 0])  # u is the x-axis of the new plane
    v_vec = transform_mat @ np.array([0, 1, 0])  # v is the y-axis of the new plane
    w_vec = transform_mat @ np.array([0, 0, 1])  # w is the normal of the new plane

    # 3. Create a 2D grid of coordinates for the output slice
    x_range = np.arange(-slice_dim / 2, slice_dim / 2)
    y_range = np.arange(-slice_dim / 2, slice_dim / 2)
    xx, yy = np.meshgrid(x_range, y_range)

    # 4. Map the 2D grid to 3D points in the volume
    slice_offset = slice_idx - (int(np.linalg.norm(data.shape)) / 2)
    points_3d = center_voxel[:, np.newaxis] \
                + xx.ravel() * u_vec[:, np.newaxis] \
                + yy.ravel() * v_vec[:, np.newaxis] \
                + slice_offset * w_vec[:, np.newaxis]

    # 5. Interpolate the data at these 3D points
    coords = [points_3d[0], points_3d[1], points_3d[2]]
    # Use order=1 for linear interpolation
    oblique_slice = map_coordinates(data, coords, order=1, cval=data.min(), mode='constant')
    return oblique_slice.reshape((slice_dim, slice_dim))