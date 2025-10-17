# loader.py

import os
import datetime
import json
import numpy as np
import pydicom
import nibabel as nib
from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates


def load_dicom_data(folder_path):
    """
    Loads a DICOM series, handling orientation, spacing, intensity, and full metadata.
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
                ds = dcmread(f, stop_before_pixels=False)
                if hasattr(ds, 'pixel_array') and hasattr(ds, 'ImagePositionPatient'):
                    slices.append(ds)
            except Exception:
                continue

        if not slices:
            raise ValueError("No readable DICOM slices with pixel data and position info found.")

        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

        first_slice = slices[0]
        metadata = {}

        tags_to_extract = [
            # Patient/Study Info
            "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
            "StudyInstanceUID", "StudyDate", "StudyTime", "StudyDescription",
            "SeriesInstanceUID", "SeriesDate", "SeriesTime", "SeriesDescription",
            "Modality", "Manufacturer", "ManufacturersModelName", "ProtocolName", "BodyPartExamined",
            # Contrast/Intensity Info
            "RescaleIntercept", "RescaleSlope", "RescaleType",
            "WindowCenter", "WindowWidth", "VOILUTFunction"
        ]

        for tag in tags_to_extract:
            value = first_slice.get(tag, None)
            if value is not None:
                if isinstance(value, pydicom.multival.MultiValue):
                    metadata[tag] = [str(v) for v in value]
                else:
                    metadata[tag] = str(value)

        # Apply rescale slope and intercept to get the real-world values
        rescale_slope = float(metadata.get('RescaleSlope', 1))
        rescale_intercept = float(metadata.get('RescaleIntercept', 0))
        image_stack = np.stack([s.pixel_array * rescale_slope + rescale_intercept for s in slices])

        # Transpose and flip for consistent orientation
        image_stack = image_stack.transpose((2, 1, 0))
        image_stack = image_stack[:, ::-1, :]

        # Determine intensity window for display
        if 'WindowCenter' in metadata and 'WindowWidth' in metadata:
            center = float(
                metadata['WindowCenter'][0] if isinstance(metadata['WindowCenter'], list) else metadata['WindowCenter'])
            width = float(
                metadata['WindowWidth'][0] if isinstance(metadata['WindowWidth'], list) else metadata['WindowWidth'])
            intensity_min = center - width / 2
            intensity_max = center + width / 2
        else:
            foreground_pixels = image_stack[image_stack > image_stack.min()]
            if foreground_pixels.size > 0:
                intensity_min = np.percentile(foreground_pixels, 1)
                intensity_max = np.percentile(foreground_pixels, 99)
            else:
                intensity_min, intensity_max = image_stack.min(), image_stack.max()

        # Build affine matrix
        pixel_spacing = [float(x) for x in first_slice.PixelSpacing]
        if len(slices) > 1:
            pos1 = np.array(slices[0].ImagePositionPatient)
            pos2 = np.array(slices[1].ImagePositionPatient)
            slice_spacing = np.linalg.norm(pos2 - pos1)
        else:
            slice_spacing = float(getattr(first_slice, 'SliceThickness', 1.0))

        orientation = np.array(first_slice.ImageOrientationPatient).reshape(2, 3)
        row_vec, col_vec = orientation[0], orientation[1]
        slice_vec = np.cross(row_vec, col_vec)

        affine = np.identity(4)
        affine[:3, 0] = row_vec * pixel_spacing[0]
        affine[:3, 1] = col_vec * pixel_spacing[1]
        affine[:3, 2] = slice_vec * slice_spacing
        affine[:3, 3] = first_slice.ImagePositionPatient

        dims = image_stack.shape
        return image_stack, affine, dims, intensity_min, intensity_max, metadata

    except Exception as e:
        print(f"Error loading DICOM data: {e}")
        return None, None, None, 0, 1, None


def load_nifti_data(file_path):
    """
    Loads NIfTI data, creates default metadata, and calculates a display window.
    """
    try:
        nifti_file = nib.load(file_path)
        data = nifti_file.get_fdata()
        affine = nifti_file.affine

        data = data[::-1, :, :]  # Corrects left-right mirroring

        if data.ndim < 3:
            raise ValueError(f"Data has fewer than 3 dimensions ({data.ndim}).")

        dims = data.shape

        foreground_pixels = data[data > np.min(data)]
        if foreground_pixels.size > 0:
            intensity_min = np.percentile(foreground_pixels, 1)
            intensity_max = np.percentile(foreground_pixels, 99)
        else:
            intensity_min, intensity_max = np.min(data), np.max(data)

        metadata = {
            "PatientName": "Unknown", "PatientID": "Unknown",
            "StudyDescription": "NIfTI Study", "Modality": "Unknown",
            "StudyInstanceUID": "", "SeriesInstanceUID": "",
            "RescaleIntercept": "0",
            "RescaleSlope": "1",
            "WindowCenter": str((intensity_max + intensity_min) / 2),
            "WindowWidth": str(intensity_max - intensity_min)
        }

        descrip = nifti_file.header.get('descrip')
        if descrip:
            try:
                # Attempt to parse JSON from description field
                header_info = json.loads(descrip.tobytes().decode('utf-8', 'ignore').strip())
                metadata.update(header_info)
            except (json.JSONDecodeError, AttributeError):
                # Fallback to using the raw description if not JSON
                metadata["StudyDescription"] = descrip.tobytes().decode('utf-8', 'ignore').strip()

        return data, affine, dims, intensity_min, intensity_max, metadata

    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None, None, None, 0, 1, None


def project_point_to_oblique_plane(norm_coords, data_shape, rot_x_deg, rot_y_deg):
    """
    Projects a 3D point onto the oblique plane and returns normalized 2D coordinates
    and the perpendicular distance from the plane.
    
    Args:
        norm_coords: Dictionary with 'S', 'C', 'A' normalized coordinates (0-1)
        data_shape: Shape of the data volume
        rot_x_deg: Rotation around X axis in degrees
        rot_y_deg: Rotation around Y axis in degrees
    
    Returns:
        Tuple of (norm_x, norm_y, depth_offset) where:
            - norm_x, norm_y: projection on the oblique plane (0-1)
            - depth_offset: perpendicular distance from plane center
    """
    if norm_coords is None:
        return (0.5, 0.5, 0)
    
    # Convert normalized coordinates to voxel coordinates
    point_voxel = np.array([
        norm_coords['S'] * (data_shape[0] - 1),
        norm_coords['C'] * (data_shape[1] - 1),
        (1.0 - norm_coords['A']) * (data_shape[2] - 1)
    ])
    
    # Volume center (where oblique plane is centered)
    center_voxel = np.array(data_shape) / 2.0
    
    # Calculate rotation matrices
    theta_x = np.deg2rad(-rot_x_deg)
    theta_y = np.deg2rad(-rot_y_deg)
    rot_x_mat = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y_mat = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    transform_mat = rot_y_mat @ rot_x_mat
    
    # Get plane basis vectors
    u_vec = transform_mat @ np.array([1, 0, 0])
    v_vec = transform_mat @ np.array([0, 1, 0])
    w_vec = transform_mat @ np.array([0, 0, 1])  # Normal to the plane
    
    # Vector from plane center to the point
    point_rel = point_voxel - center_voxel
    
    # Project onto plane basis vectors (in-plane coordinates)
    u_coord = np.dot(point_rel, u_vec)
    v_coord = np.dot(point_rel, v_vec)
    
    # Project onto normal vector (depth/perpendicular distance from plane)
    depth_offset = np.dot(point_rel, w_vec)
    
    # Convert to normalized coordinates (0-1)
    slice_dim = int(np.linalg.norm(data_shape))
    norm_x = (u_coord + slice_dim / 2) / slice_dim
    norm_y = (v_coord + slice_dim / 2) / slice_dim
    
    return (norm_x, norm_y, depth_offset)


def get_slice_data(data, dims, slices, affine, intensity_min=0, intensity_max=1000, rot_x_deg=0, rot_y_deg=0, view_type='axial', norm_coords=None):
    """
    Get slice data with optional normalized coordinates for oblique slicing.

    Args:
        norm_coords: Dictionary with 'S', 'C', 'A' normalized coordinates (0-1) for oblique center
    """
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
        # Oblique slice: field of view stays fixed, but slice depth changes with crosshair
        # Calculate the perpendicular distance from volume center to use as slice offset
        slice_offset = 0
        if norm_coords is not None:
            _, _, depth_offset = project_point_to_oblique_plane(norm_coords, dims, rot_x_deg, rot_y_deg)
            # Use the depth offset to determine which parallel slice to show
            slice_dim = int(np.linalg.norm(dims))
            slice_offset = int(depth_offset + slice_dim / 2)
        else:
            slice_dim = int(np.linalg.norm(dims))
            slice_offset = slice_dim // 2
        
        # Keep the oblique plane centered on volume center
        slice_data = _get_oblique_slice(data, rot_x_deg, rot_y_deg, slice_offset, center_position=None)
        x_spacing, y_spacing = 1, 1
    else:
        return np.zeros((10, 10), dtype=np.uint8)

    if slice_data.size == 0:
        return np.zeros((10, 10), dtype=np.uint8)

    if x_spacing != 0 and y_spacing != 0:
        aspect_ratio = abs(y_spacing / x_spacing)
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


def _get_oblique_slice(data, rot_x_deg, rot_y_deg, slice_idx, center_position=None):
    """
    Extract an oblique slice from the volume.

    Args:
        data: 3D numpy array
        rot_x_deg: Rotation around X axis in degrees
        rot_y_deg: Rotation around Y axis in degrees
        slice_idx: Slice index (used for offset from center)
        center_position: Tuple of (x, y, z) normalized coordinates (0-1) for slice center.
                        If None, uses volume center.
    """
    if center_position is None:
        center_voxel = np.array(data.shape) / 2.0
    else:
        # Convert normalized coordinates to voxel coordinates
        # Invert the axial (Z) coordinate to match the other views
        center_voxel = np.array([
            center_position[0] * (data.shape[0] - 1),
             center_position[1] * (data.shape[1] - 1),
            (1.0 - center_position[2]) * (data.shape[2] - 1)  # Invert Z coordinate
        ])

    slice_dim = int(np.linalg.norm(data.shape))

# Negate the rotation angle to match the visual orientation
    theta_x = np.deg2rad(-rot_x_deg)
    theta_y = np.deg2rad(-rot_y_deg)
    rot_x_mat = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y_mat = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    transform_mat = rot_y_mat @ rot_x_mat

    u_vec = transform_mat @ np.array([1, 0, 0])
    v_vec = transform_mat @ np.array([0, 1, 0])
    w_vec = transform_mat @ np.array([0, 0, 1])

    x_range = np.arange(-slice_dim / 2, slice_dim / 2)
    y_range = np.arange(-slice_dim / 2, slice_dim / 2)
    xx, yy = np.meshgrid(x_range, y_range)

    # Calculate offset from center based on slice_idx
    slice_offset = slice_idx - (slice_dim / 2)

    points_3d = center_voxel[:, np.newaxis] \
                + xx.ravel() * u_vec[:, np.newaxis] \
                + yy.ravel() * v_vec[:, np.newaxis] \
                + slice_offset * w_vec[:, np.newaxis]

    coords = [points_3d[0], points_3d[1], points_3d[2]]
    oblique_slice = map_coordinates(data, coords, order=1, cval=data.min(), mode='constant')
    return oblique_slice.reshape((slice_dim, slice_dim))


def export_to_nifti(image_data, affine, output_path, metadata=None):
    """
    Exports 3D data to a NIfTI file, embedding metadata into the header.
    """
    try:
        # Reverse the flip from the loader to restore original orientation
        image_data = image_data[::-1, :, :]
        nifti_image = nib.Nifti1Image(image_data.astype(np.float32), affine)

        if metadata:
            header_info = {
                "PatientName": metadata.get("PatientName", "Unknown"),
                "StudyDescription": metadata.get("StudyDescription", "N/A"),
                "RescaleSlope": metadata.get("RescaleSlope", "1"),
                "RescaleIntercept": metadata.get("RescaleIntercept", "0"),
                "WindowCenter": metadata.get("WindowCenter"),
                "WindowWidth": metadata.get("WindowWidth")
            }

            header_info = {k: v for k, v in header_info.items() if v is not None}
            desc_str = json.dumps(header_info)
            nifti_image.header['descrip'] = desc_str[:79].encode('utf-8')

        nib.save(nifti_image, output_path)
        print(f"Successfully exported data to {output_path}")
        return True
    except Exception as e:
        print(f"Error exporting to NIfTI: {e}")
        return False


def export_to_dicom(image_data, affine, output_folder, metadata=None):
    """
    Exports 3D data to a DICOM series, populating full metadata including contrast.
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if metadata is None:
            metadata = {}

        image_data = image_data[:, ::-1, :]
        image_data = image_data.transpose((2, 1, 0))

        rescale_slope = float(metadata.get('RescaleSlope', 1))
        rescale_intercept = float(metadata.get('RescaleIntercept', 0))
        if rescale_slope == 0: rescale_slope = 1
        stored_pixel_data = (image_data - rescale_intercept) / rescale_slope

        study_instance_uid = metadata.get('StudyInstanceUID') or generate_uid()
        series_instance_uid = metadata.get('SeriesInstanceUID') or generate_uid()

        pixel_spacing = [abs(affine[0, 0]), abs(affine[1, 1])]
        slice_thickness = abs(affine[2, 2])

        row_vec = affine[:3, 0] / pixel_spacing[0]
        col_vec = affine[:3, 1] / pixel_spacing[1]
        image_orientation_patient = list(np.round(row_vec, 6)) + list(np.round(col_vec, 6))

        for i in range(stored_pixel_data.shape[0]):
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            ds = Dataset()
            ds.file_meta = file_meta

            ds.PatientName = metadata.get("PatientName", "Unknown")
            ds.PatientID = metadata.get("PatientID", "Unknown")
            ds.StudyDescription = metadata.get("StudyDescription", "Exported Study")
            ds.Modality = metadata.get("Modality", "CT")
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = series_instance_uid
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

            ds.RescaleIntercept = str(rescale_intercept)
            ds.RescaleSlope = str(rescale_slope)
            ds.RescaleType = metadata.get("RescaleType", "HU")
            if "WindowCenter" in metadata:
                ds.WindowCenter = metadata["WindowCenter"]
            if "WindowWidth" in metadata:
                ds.WindowWidth = metadata["WindowWidth"]
            if "VOILUTFunction" in metadata:
                ds.VOILUTFunction = metadata["VOILUTFunction"]

            position = affine @ np.array([0, 0, i, 1])
            ds.ImagePositionPatient = list(np.round(position[:3], 6))
            ds.ImageOrientationPatient = image_orientation_patient
            ds.PixelSpacing = pixel_spacing
            ds.SliceThickness = slice_thickness
            ds.InstanceNumber = i + 1

            slice_data = stored_pixel_data[i, :, :]
            ds.Rows, ds.Columns = slice_data.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 1  # Signed Integer
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelData = slice_data.astype(np.int16).tobytes()

            filename = os.path.join(output_folder, f"slice_{i + 1:04d}.dcm")
            ds.save_as(filename, write_like_original=False)

        print(f"Successfully exported DICOM series to {output_folder}")
        return True
    except Exception as e:
        print(f"Error exporting to DICOM: {e}")
        return False