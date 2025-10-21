import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def normalize_image(img_slice):
    """
    Normalize image slice for better contrast and brightness.
    Uses percentile-based normalization to handle outliers.
    """
    # Remove any NaN or infinite values
    img_slice = np.nan_to_num(img_slice, nan=0.0, posinf=0.0, neginf=0.0)

    # Use percentile-based normalization to handle outliers
    p2, p98 = np.percentile(img_slice, (2, 98))

    if p98 > p2:
        img_normalized = np.clip((img_slice - p2) / (p98 - p2), 0, 1)
    else:
        img_normalized = img_slice

    return img_normalized


def save_slices(nifti_path, output_dir='nifti_slices'):
    """
    Read a NIfTI file and save all slices as PNG images in three orientations.

    Parameters:
    -----------
    nifti_path : str
        Path to the NIfTI file (.nii or .nii.gz)
    output_dir : str
        Directory where PNG slices will be saved
    """
    # Load the NIfTI file
    print(f"Loading NIfTI file: {nifti_path}")
    nii_img = nib.load(nifti_path)
    data = nii_img.get_fdata()

    print(f"Image shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{np.min(data):.2f}, {np.max(data):.2f}]")

    # Create output directories
    base_dir = Path(output_dir)
    axial_dir = base_dir / 'axial'
    sagittal_dir = base_dir / 'sagittal'
    coronal_dir = base_dir / 'coronal'

    for dir_path in [axial_dir, sagittal_dir, coronal_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Configure matplotlib for high-quality output
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0

    # Extract and save axial slices (slice along z-axis)
    print(f"\nExtracting {data.shape[2]} axial slices...")
    for i in range(data.shape[2]):
        img_slice = data[:, :, i]
        img_normalized = normalize_image(img_slice)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_normalized.T, cmap='gray', origin='lower', interpolation='bilinear')
        ax.axis('off')
        plt.savefig(axial_dir / f'axial_slice_{i:04d}.png',
                    bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{data.shape[2]} axial slices")

    # Extract and save sagittal slices (slice along x-axis)
    print(f"\nExtracting {data.shape[0]} sagittal slices...")
    for i in range(data.shape[0]):
        img_slice = data[i, :, :]
        img_normalized = normalize_image(img_slice)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_normalized.T, cmap='gray', origin='lower', interpolation='bilinear')
        ax.axis('off')
        plt.savefig(sagittal_dir / f'sagittal_slice_{i:04d}.png',
                    bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{data.shape[0]} sagittal slices")

    # Extract and save coronal slices (slice along y-axis)
    print(f"\nExtracting {data.shape[1]} coronal slices...")
    for i in range(data.shape[1]):
        img_slice = data[:, i, :]
        img_normalized = normalize_image(img_slice)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_normalized.T, cmap='gray', origin='lower', interpolation='bilinear')
        ax.axis('off')
        plt.savefig(coronal_dir / f'coronal_slice_{i:04d}.png',
                    bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{data.shape[1]} coronal slices")

    print(f"\n✓ All slices saved successfully!")
    print(f"  Axial: {data.shape[2]} slices in {axial_dir}")
    print(f"  Sagittal: {data.shape[0]} slices in {sagittal_dir}")
    print(f"  Coronal: {data.shape[1]} slices in {coronal_dir}")


# Example usage
if __name__ == "__main__":
    nifti_file = r"F:\CUFE-MPR\CT_Set\s0029\ct.nii.gz"

    output_directory = "nifti_slices4"

    save_slices(nifti_file, output_directory)