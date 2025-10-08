import numpy as np
import nibabel as nib

# Check for scipy availability
try:
    from scipy.ndimage import map_coordinates

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


def get_slice_data(data, dims, slices, intensity_min = 0, intensity_max = 1000, rot_x_deg=0, rot_y_deg=0, view_type='axial'):
    """
    Generates a slice (Axial, Coronal, Sagittal, or Oblique) from the 3D volume.

    Args:
        data (np.ndarray): 3D volume data.
        dims (tuple): Dimensions of the data.
        slices (dict): Dictionary of current slice indices.
        intensity_min (float): Minimum intensity for windowing.
        intensity_max (float): Maximum intensity for windowing.
        rot_x_deg (int): Oblique rotation around X-axis (Pitch).
        rot_y_deg (int): Oblique rotation around Y-axis (Yaw).
        view_type (str): 'axial', 'coronal', 'sagittal', or 'oblique'.

    Returns:
        np.ndarray: The 8-bit grayscale slice data (normalized and clipped).
    """
    if data is None:
        return np.zeros((10, 10), dtype=np.uint8)

    if view_type == 'axial':
        slice_data = np.rot90(data[:, :, slices['axial']])
    elif view_type == 'coronal':
        slice_data = np.rot90(data[:, slices['coronal'], :])
    elif view_type == 'sagittal':
        slice_data = np.rot90(data[slices['sagittal'], :, :])
    elif view_type == 'oblique':
        slice_data = _get_oblique_slice(data, rot_x_deg, rot_y_deg, slices['oblique'])
    else:
        return np.zeros((10, 10), dtype=np.uint8)

    if slice_data.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    # Intensity Windowing and Normalization (Converts to 0-255 range)
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
    if SCIPY_AVAILABLE:
        coords = [points_3d[0], points_3d[1], points_3d[2]]
        # Use order=1 for linear interpolation
        oblique_slice = map_coordinates(data, coords, order=1, cval=data.min(), mode='constant')
        return oblique_slice.reshape((slice_dim, slice_dim))
    else:
        # Fallback to nearest-neighbor interpolation
        points_3d = np.clip(points_3d, 0, np.array(data.shape)[:, np.newaxis] - 1).astype(int)
        oblique_slice = data[points_3d[0], points_3d[1], points_3d[2]]
        return oblique_slice.reshape((slice_dim, slice_dim))