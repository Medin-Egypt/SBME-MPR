"""
Segmentation Manager - Lazy loading and memory-efficient segmentation handling

This module provides memory-efficient segmentation loading by:
1. Using memory-mapped files (mmap) to access data without loading into RAM
2. Loading only the slices needed for current view
3. Caching recently accessed slices with automatic eviction
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from collections import OrderedDict
import os


class SegmentationManager:
    """
    Manages multiple segmentation files with lazy loading and caching.

    Uses nibabel's mmap feature to avoid loading entire volumes into memory.
    Caches individual slices with LRU eviction.
    """

    def __init__(self, max_cache_slices=100):
        """
        Initialize the segmentation manager.

        Parameters:
        -----------
        max_cache_slices : int
            Maximum number of slices to keep in cache before eviction
        """
        self.file_paths = []  # List of Path objects
        self.nifti_objs = []  # List of nibabel Nifti1Image objects (mmap)
        self.shapes = []  # List of shapes for quick access

        # LRU cache: key = (file_idx, axis, slice_idx), value = 2D numpy array
        self.slice_cache = OrderedDict()
        self.max_cache_slices = max_cache_slices

        # Merged volume for fast 2D rendering (created on demand)
        self.merged_volume = None
        self.merged_cache = OrderedDict()  # Cache for merged slices
        self.max_merged_cache = max_cache_slices

    def add_file(self, file_path):
        """
        Add a segmentation file to the manager.

        Parameters:
        -----------
        file_path : str or Path
            Path to the NIfTI segmentation file

        Returns:
        --------
        bool : True if successfully added, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Load with mmap_mode to avoid loading data into memory
            nii = nib.load(str(file_path), mmap=True)

            # Get data proxy shape without loading the data
            shape = nii.shape

            self.file_paths.append(file_path)
            self.nifti_objs.append(nii)
            self.shapes.append(shape)

            print(f"Added segmentation: {file_path.name} (shape: {shape})")
            return True

        except Exception as e:
            print(f"Failed to add segmentation {file_path}: {e}")
            return False

    def clear(self):
        """Clear all segmentation files and cache."""
        self.file_paths.clear()
        self.nifti_objs.clear()
        self.shapes.clear()
        self.slice_cache.clear()
        self.merged_volume = None
        self.merged_cache.clear()
        print("Cleared all segmentations from manager")

    def get_count(self):
        """Return the number of loaded segmentation files."""
        return len(self.file_paths)

    def get_file_paths(self):
        """Return list of file paths."""
        return self.file_paths.copy()

    def validate_shape(self, expected_shape):
        """
        Check if all loaded segmentations match the expected shape.

        Parameters:
        -----------
        expected_shape : tuple
            Expected (x, y, z) dimensions

        Returns:
        --------
        list : List of (filename, actual_shape) tuples for mismatched files
        """
        mismatches = []
        for i, shape in enumerate(self.shapes):
            if shape != expected_shape:
                mismatches.append((self.file_paths[i].name, shape))
        return mismatches

    def get_slice(self, file_idx, axis, slice_idx):
        """
        Get a 2D slice from a segmentation file.
        Uses caching and memory-mapped file access.

        Parameters:
        -----------
        file_idx : int
            Index of the segmentation file (0 to count-1)
        axis : str
            One of 'axial' (z), 'coronal' (y), or 'sagittal' (x)
        slice_idx : int
            Slice index along the specified axis

        Returns:
        --------
        np.ndarray : 2D slice, or None if invalid
        """
        if file_idx < 0 or file_idx >= len(self.nifti_objs):
            return None

        cache_key = (file_idx, axis, slice_idx)

        # Check cache first
        if cache_key in self.slice_cache:
            # Move to end (most recently used)
            self.slice_cache.move_to_end(cache_key)
            return self.slice_cache[cache_key]

        # Load slice from mmap file
        try:
            nii = self.nifti_objs[file_idx]
            shape = self.shapes[file_idx]

            # Get dataobj (mmap array) - doesn't load data yet
            data_proxy = nii.dataobj

            # Extract only the slice we need (this is where mmap shines)
            if axis == 'axial':
                if slice_idx < 0 or slice_idx >= shape[2]:
                    return None
                # Apply the same flip as main data loading
                slice_2d = np.array(data_proxy[:, :, slice_idx], dtype=np.float32)
                slice_2d = slice_2d[::-1, :, ...]  # Flip along first axis

            elif axis == 'coronal':
                if slice_idx < 0 or slice_idx >= shape[1]:
                    return None
                slice_2d = np.array(data_proxy[:, slice_idx, :], dtype=np.float32)
                slice_2d = slice_2d[::-1, :, ...]  # Flip along first axis

            elif axis == 'sagittal':
                if slice_idx < 0 or slice_idx >= shape[0]:
                    return None
                slice_2d = np.array(data_proxy[slice_idx, :, :], dtype=np.float32)
                # No flip needed, but need to reverse X axis
                slice_2d = slice_2d[::-1, :, ...]

            else:
                return None

            # Add to cache
            self.slice_cache[cache_key] = slice_2d

            # Evict oldest if cache is full
            if len(self.slice_cache) > self.max_cache_slices:
                self.slice_cache.popitem(last=False)

            return slice_2d

        except Exception as e:
            print(f"Error loading slice {axis}[{slice_idx}] from file {file_idx}: {e}")
            return None

    def get_all_slices(self, axis, slice_idx):
        """
        Get the same slice from all segmentation files.

        Parameters:
        -----------
        axis : str
            One of 'axial', 'coronal', or 'sagittal'
        slice_idx : int
            Slice index along the specified axis

        Returns:
        --------
        list : List of 2D numpy arrays (one per segmentation file)
        """
        slices = []
        for file_idx in range(len(self.nifti_objs)):
            slice_2d = self.get_slice(file_idx, axis, slice_idx)
            if slice_2d is not None:
                slices.append(slice_2d)
        return slices

    def clear_cache(self):
        """Clear the slice cache to free memory."""
        self.slice_cache.clear()
        print(f"Cleared slice cache")

    def get_cache_info(self):
        """Get information about cache usage."""
        return {
            'cached_slices': len(self.slice_cache),
            'max_slices': self.max_cache_slices,
            'memory_mb_approx': len(self.slice_cache) * 0.5  # Rough estimate
        }

    def build_merged_volume(self):
        """
        Build a merged binary volume from all segmentations for fast 2D rendering.
        This merges all segmentation masks into a single volume.
        Much faster than processing each segmentation separately.
        """
        if len(self.nifti_objs) == 0:
            print("No segmentations to merge")
            return

        if len(self.shapes) == 0:
            print("No shapes available")
            return

        # Get the shape from first segmentation
        shape = self.shapes[0]
        print(f"Building merged segmentation volume with shape {shape}...")

        # Create merged volume (binary: 0 or 1)
        self.merged_volume = np.zeros(shape, dtype=np.uint8)

        # Merge all segmentations
        for idx, nii in enumerate(self.nifti_objs):
            try:
                # Load entire volume from mmap (necessary for merge)
                data = np.array(nii.dataobj, dtype=np.float32)

                # Apply flip to match main data
                data = data[::-1, :, :]

                # Binarize and merge (any voxel with segmentation = 1)
                binary_mask = (data > 0.5).astype(np.uint8)
                self.merged_volume = np.maximum(self.merged_volume, binary_mask)

                print(f"  Merged {self.file_paths[idx].name}")

            except Exception as e:
                print(f"  Error merging {self.file_paths[idx].name}: {e}")

        print(f"Merged volume created: {np.count_nonzero(self.merged_volume)} non-zero voxels")

    def get_merged_slice(self, axis, slice_idx):
        """
        Get a 2D slice from the merged segmentation volume.
        Much faster than getting slices from all individual files.

        Parameters:
        -----------
        axis : str
            One of 'axial', 'coronal', or 'sagittal'
        slice_idx : int
            Slice index along the specified axis

        Returns:
        --------
        np.ndarray : 2D slice, or None if invalid
        """
        if self.merged_volume is None:
            return None

        cache_key = (axis, slice_idx)

        # Check cache first
        if cache_key in self.merged_cache:
            self.merged_cache.move_to_end(cache_key)
            return self.merged_cache[cache_key]

        # Extract slice from merged volume
        try:
            if axis == 'axial':
                if slice_idx < 0 or slice_idx >= self.merged_volume.shape[2]:
                    return None
                slice_2d = self.merged_volume[:, :, slice_idx].copy()

            elif axis == 'coronal':
                if slice_idx < 0 or slice_idx >= self.merged_volume.shape[1]:
                    return None
                slice_2d = self.merged_volume[:, slice_idx, :].copy()

            elif axis == 'sagittal':
                if slice_idx < 0 or slice_idx >= self.merged_volume.shape[0]:
                    return None
                slice_2d = self.merged_volume[slice_idx, :, :].copy()

            else:
                return None

            # Cache the slice
            self.merged_cache[cache_key] = slice_2d

            # Evict oldest if cache is full
            if len(self.merged_cache) > self.max_merged_cache:
                self.merged_cache.popitem(last=False)

            return slice_2d

        except Exception as e:
            print(f"Error getting merged slice {axis}[{slice_idx}]: {e}")
            return None
