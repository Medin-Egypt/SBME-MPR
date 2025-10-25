"""
Segmentation caching system for fast repeated loads.
Caches both merged volumes (2D) and generated meshes (3D).
"""

import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
import pyvista as pv


class SegmentationCache:
    """
    Manages disk-based cache for segmentation data.

    Cache Structure:
    - Merged volumes: Cached as numpy arrays
    - 3D meshes: Cached as PyVista mesh files
    """

    def __init__(self, cache_dir=None):
        """
        Initialize cache manager.

        Parameters:
        -----------
        cache_dir : str or Path, optional
            Directory for cache files. Defaults to .cache/segmentations in project root
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / ".cache" / "segmentations"
        else:
            cache_dir = Path(cache_dir)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Segmentation cache directory: {self.cache_dir}")

    def _generate_cache_key(self, file_paths):
        """
        Generate a unique cache key based on file paths and modification times.

        Parameters:
        -----------
        file_paths : list of Path
            List of segmentation file paths

        Returns:
        --------
        str : Hash-based cache key
        """
        # Create a string with all file paths and their modification times
        cache_string = ""
        for fp in sorted(file_paths):  # Sort for consistency
            try:
                mtime = os.path.getmtime(fp)
                cache_string += f"{fp}:{mtime};"
            except:
                cache_string += f"{fp}:0;"

        # Generate hash
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        return cache_hash

    def get_merged_volume_path(self, file_paths):
        """
        Get the cache path for merged volume.

        Parameters:
        -----------
        file_paths : list of Path
            List of segmentation file paths

        Returns:
        --------
        Path : Cache file path
        """
        cache_key = self._generate_cache_key(file_paths)
        return self.cache_dir / f"merged_{cache_key}.npz"

    def load_merged_volume(self, file_paths):
        """
        Load merged volume from cache if available.

        Parameters:
        -----------
        file_paths : list of Path
            List of segmentation file paths

        Returns:
        --------
        np.ndarray or None : Cached merged volume, or None if not cached
        """
        cache_path = self.get_merged_volume_path(file_paths)

        if cache_path.exists():
            try:
                data = np.load(cache_path)
                merged_volume = data['merged_volume']
                print(f"Loaded merged volume from cache: {cache_path.name}")
                return merged_volume
            except Exception as e:
                print(f"Failed to load cached merged volume: {e}")
                # Delete corrupted cache file
                try:
                    cache_path.unlink()
                except:
                    pass

        return None

    def save_merged_volume(self, file_paths, merged_volume):
        """
        Save merged volume to cache.

        Parameters:
        -----------
        file_paths : list of Path
            List of segmentation file paths
        merged_volume : np.ndarray
            Merged volume to cache
        """
        cache_path = self.get_merged_volume_path(file_paths)

        try:
            np.savez_compressed(cache_path, merged_volume=merged_volume)
            print(f"Saved merged volume to cache: {cache_path.name}")
        except Exception as e:
            print(f"Failed to save merged volume to cache: {e}")

    def get_mesh_cache_dir(self, file_paths):
        """
        Get the cache directory for meshes.

        Parameters:
        -----------
        file_paths : list of Path
            List of segmentation file paths

        Returns:
        --------
        Path : Cache directory for meshes
        """
        cache_key = self._generate_cache_key(file_paths)
        mesh_dir = self.cache_dir / f"meshes_{cache_key}"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        return mesh_dir

    def load_mesh(self, file_paths, filename):
        """
        Load a cached mesh.

        Parameters:
        -----------
        file_paths : list of Path
            List of segmentation file paths
        filename : str
            Stem name of the segmentation file

        Returns:
        --------
        pv.PolyData or None : Cached mesh, or None if not cached
        """
        mesh_dir = self.get_mesh_cache_dir(file_paths)
        cache_path = mesh_dir / f"{filename}.vtk"

        if cache_path.exists():
            try:
                mesh = pv.read(str(cache_path))
                print(f"  Loaded mesh from cache: {filename}")
                return mesh
            except Exception as e:
                print(f"  Failed to load cached mesh {filename}: {e}")
                # Delete corrupted cache file
                try:
                    cache_path.unlink()
                except:
                    pass

        return None

    def save_mesh(self, file_paths, filename, mesh):
        """
        Save a mesh to cache.

        Parameters:
        -----------
        file_paths : list of Path
            List of segmentation file paths
        filename : str
            Stem name of the segmentation file
        mesh : pv.PolyData
            Mesh to cache
        """
        mesh_dir = self.get_mesh_cache_dir(file_paths)
        cache_path = mesh_dir / f"{filename}.vtk"

        try:
            mesh.save(str(cache_path))
            # Don't print here to avoid spam
        except Exception as e:
            print(f"  Failed to save mesh {filename} to cache: {e}")

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print("Cache cleared")
        except Exception as e:
            print(f"Failed to clear cache: {e}")

    def get_cache_size(self):
        """
        Get the total size of cached data in MB.

        Returns:
        --------
        float : Cache size in MB
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except Exception as e:
            print(f"Failed to calculate cache size: {e}")

        return total_size / (1024 * 1024)  # Convert to MB
