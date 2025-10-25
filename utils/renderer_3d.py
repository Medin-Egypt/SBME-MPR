import pyvista as pv
from pyvistaqt import QtInteractor
import nibabel as nib
import numpy as np
from pathlib import Path
from skimage import measure
import json
import gc
from .segmentation_cache import SegmentationCache
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QCheckBox, QGroupBox, QScrollArea, QPushButton, QProgressDialog
)
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal


def load_colormap(colormap_file):
    """
    Load colormap from JSON file.
    Expected format:
    {
        "liver": "#ff0000",
        "kidney": "#00ff00",
        "spleen": "#0000ff"
    }
    Colors should be hex strings (#000000 to #ffffff)
    """
    with open(colormap_file, 'r') as f:
        colormap = json.load(f)

    # Convert hex colors to RGB [0, 1] range
    rgb_colormap = {}
    for key, hex_color in colormap.items():
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB [0, 1]
        try:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            rgb_colormap[key] = [r, g, b]
        except Exception as e:
            print(f"Warning: Skipping invalid color '{hex_color}' for key '{key}'. Error: {e}")

    return rgb_colormap


def nifti_to_surface(nifti_path, smoothing=True, smoothing_iterations=20, decimation_target=0.3):
    """
    Convert NIfTI segmentation mask to surface mesh using marching cubes.
    Optimized for memory efficiency with mesh decimation.

    Parameters:
    -----------
    nifti_path : str or Path
        Path to .nii or .nii.gz file
    smoothing : bool
        Whether to apply Laplacian smoothing
    smoothing_iterations : int
        Number of smoothing iterations (reduced from 50 to 20 for performance)
    decimation_target : float
        Target reduction ratio for mesh decimation (0.3 = reduce to 30% of original triangles)

    Returns:
    --------
    mesh : pv.PolyData
        Surface mesh
    affine : np.ndarray
        Affine transformation matrix from NIfTI
    """
    # Load NIfTI file
    nii = nib.load(str(nifti_path))
    # Use float32 instead of float64 to save memory
    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine.copy()

    # Binarize if needed (assumes non-zero values are the segmentation)
    binary_data = (data > 0).astype(np.uint8)

    # Clear original data to free memory
    del data

    # Apply marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        binary_data,
        level=0.5,
        spacing=nii.header.get_zooms()[:3]
    )

    # Clear binary data to free memory
    del binary_data

    # Convert to float32 for memory efficiency
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int32)

    # Create PyVista mesh
    # Ensure faces are correctly formatted for PyVista
    # Each face should be [n_points, p0, p1, p2, ...]
    n_points = 3  # Assuming marching cubes produces triangles
    faces_pv = np.empty((faces.shape[0], n_points + 1), dtype=np.int32)
    faces_pv[:, 0] = n_points
    faces_pv[:, 1:] = faces

    # Clear intermediate face array
    del faces

    mesh = pv.PolyData(verts.astype(np.float64), faces_pv)

    # Clear intermediate arrays
    del verts, faces_pv

    # Apply affine transformation to align with physical coordinates
    # This transforms voxel coordinates (i, j, k) to physical space (x, y, z)
    # Marching cubes output is already in scaled voxel space (due to spacing)
    # We need to apply the affine transformation
    verts_transformed = nib.affines.apply_affine(affine, mesh.points).astype(np.float32)
    mesh.points = verts_transformed
    del verts_transformed

    # Decimate mesh to reduce triangle count and memory usage
    # Only decimate if mesh has more than 10000 faces
    # Use n_cells (new API) instead of deprecated n_faces
    if mesh.n_cells > 10000:
        try:
            original_cells = mesh.n_cells
            mesh = mesh.decimate(decimation_target, volume_preservation=True)
            print(f"  Decimated mesh from {original_cells} to {mesh.n_cells} faces ({decimation_target*100}% target)")
        except Exception as e:
            print(f"  Warning: Decimation failed. {e}")

    # Optional smoothing (reduced iterations for performance)
    if smoothing:
        try:
            mesh = mesh.smooth(n_iter=smoothing_iterations, relaxation_factor=0.1)
        except Exception as e:
            print(f"  Warning: Smoothing failed. {e}")

    return mesh, affine


def categorize_structures(filenames):
    """
    Categorize anatomical structures into systems based on filename patterns.

    Returns:
    --------
    dict : {system_name: [filenames]}
    """
    systems = {
        'Skeletal - Spine': [],
        'Skeletal - Ribs': [],
        'Skeletal - Limbs': [],
        'Skeletal - Other': [],
        'Muscular': [],
        'Cardiovascular - Arteries': [],
        'Cardiovascular - Veins': [],
        'Cardiovascular - Heart': [],
        'Other': []
    }

    for fname in filenames:
        name_lower = fname.stem.lower()

        if 'vertebrae' in name_lower or 'sacrum' in name_lower:
            systems['Skeletal - Spine'].append(fname)
        elif 'rib' in name_lower or 'sternum' in name_lower or 'costal' in name_lower:
            systems['Skeletal - Ribs'].append(fname)
        elif any(bone in name_lower for bone in ['humerus', 'scapula', 'clavicula', 'femur', 'hip', 'skull']):
            systems['Skeletal - Limbs'].append(fname)
        elif any(muscle in name_lower for muscle in ['gluteus', 'autochthon', 'iliopsoas', 'muscle']):
            systems['Muscular'].append(fname)
        elif 'artery' in name_lower or 'aorta' in name_lower or 'trunk' in name_lower:
            systems['Cardiovascular - Arteries'].append(fname)
        elif 'vein' in name_lower or 'vena' in name_lower:
            systems['Cardiovascular - Veins'].append(fname)
        elif 'heart' in name_lower:
            systems['Cardiovascular - Heart'].append(fname)
        else:
            systems['Other'].append(fname)

    # Remove empty systems
    systems = {k: v for k, v in systems.items() if v}

    return systems


class MeshLoadWorker(QThread):
    """
    Worker thread for loading meshes in the background.
    Prevents UI freezing during mesh generation.
    Can also handle merged volume building.
    """
    # Signals
    progress = pyqtSignal(str, int, int)  # (message, current, total)
    mesh_loaded = pyqtSignal(str, object, list, str)  # (filename, mesh, color, system_name)
    merged_volume_ready = pyqtSignal()  # Emitted when merged volume is built
    finished = pyqtSignal()
    error = pyqtSignal(str, str)  # (filename, error_message)

    def __init__(self, files_to_load, colormap, system_opacities, seg_manager=None, cache=None):
        super().__init__()
        self.files_to_load = files_to_load  # List of (system_name, filepath) tuples
        self.colormap = colormap
        self.system_opacities = system_opacities
        self.seg_manager = seg_manager  # Optional: for building merged volume
        self.cache = cache  # Optional: SegmentationCache instance
        self._cancelled = False

    def cancel(self):
        """Cancel the loading process."""
        self._cancelled = True

    def run(self):
        """
        Load files once and use for both merged volume and 3D meshes.
        Uses cache when available for massive speed improvement!
        """
        total = len(self.files_to_load)

        # Get list of all file paths for cache key
        all_file_paths = [fp for _, fp in self.files_to_load] if self.cache else []

        # Try to load merged volume from cache
        merged_from_cache = False
        if self.seg_manager is not None and self.cache is not None and total > 0:
            cached_merged = self.cache.load_merged_volume(all_file_paths)
            if cached_merged is not None:
                self.seg_manager.merged_volume = cached_merged
                merged_from_cache = True
                print("✓ Using cached merged volume")
                self.merged_volume_ready.emit()

                if self._cancelled:
                    self.finished.emit()
                    return

        # Initialize merged volume if not cached
        if self.seg_manager is not None and not merged_from_cache and total > 0:
            # Get shape from first file
            first_file = self.files_to_load[0][1]
            try:
                nii = nib.load(str(first_file), mmap=True)
                shape = nii.shape
                print(f"Initializing merged volume with shape {shape}")
                self.seg_manager.merged_volume = np.zeros(shape, dtype=np.uint8)
            except Exception as e:
                print(f"Error initializing merged volume: {e}")
                self.seg_manager = None  # Disable merging on error

        # Process each file: load once, use for both 2D merge and 3D mesh
        for idx, (system_name, nifti_file) in enumerate(self.files_to_load):
            if self._cancelled:
                break

            filename = nifti_file.stem
            self.progress.emit(f"Loading {filename}...", idx, total)

            # Try to load mesh from cache first
            cached_mesh = None
            if self.cache is not None:
                cached_mesh = self.cache.load_mesh(all_file_paths, filename)

            if cached_mesh is not None:
                # Use cached mesh
                try:
                    # Find color
                    color = [0.5, 0.5, 0.5]  # Default grey
                    for key, rgb in self.colormap.items():
                        if key.lower() in filename.lower():
                            color = rgb
                            break

                    # Emit cached mesh
                    self.mesh_loaded.emit(filename, cached_mesh, color, system_name)
                except Exception as e:
                    print(f"  Error loading cached mesh {filename}: {e}")
                continue  # Skip to next file

            # Not cached, need to process
            try:
                # Load NIfTI file ONCE
                nii = nib.load(str(nifti_file))
                data = nii.get_fdata(dtype=np.float32)
                affine = nii.affine

                # Step 1: Add to merged volume for 2D (if seg_manager provided and not cached)
                if self.seg_manager is not None and not merged_from_cache:
                    try:
                        # Apply flip to match main data
                        data_flipped = data[::-1, :, :]
                        # Binarize and merge
                        binary_mask = (data_flipped > 0.5).astype(np.uint8)
                        self.seg_manager.merged_volume = np.maximum(
                            self.seg_manager.merged_volume,
                            binary_mask
                        )
                    except Exception as e:
                        print(f"  Error merging {filename} to 2D volume: {e}")

                # Step 2: Generate 3D mesh from the same data
                try:
                    # Binarize
                    binary_data = (data > 0).astype(np.uint8)

                    # Apply marching cubes
                    verts, faces, normals, values = measure.marching_cubes(
                        binary_data,
                        level=0.5,
                        spacing=nii.header.get_zooms()[:3]
                    )

                    # Clear binary data to free memory
                    del binary_data, data

                    # Create PyVista mesh
                    verts = verts.astype(np.float32)
                    faces = faces.astype(np.int32)

                    n_points = 3
                    faces_pv = np.empty((faces.shape[0], n_points + 1), dtype=np.int32)
                    faces_pv[:, 0] = n_points
                    faces_pv[:, 1:] = faces
                    del faces

                    mesh = pv.PolyData(verts.astype(np.float64), faces_pv)
                    del verts, faces_pv

                    # Apply affine transformation
                    verts_transformed = nib.affines.apply_affine(affine, mesh.points).astype(np.float32)
                    mesh.points = verts_transformed
                    del verts_transformed

                    # Decimate mesh if needed
                    if mesh.n_cells > 10000:
                        try:
                            original_cells = mesh.n_cells
                            mesh = mesh.decimate(0.3, volume_preservation=True)
                            print(f"  Decimated {filename}: {original_cells} -> {mesh.n_cells} faces")
                        except Exception as e:
                            print(f"  Warning: Decimation failed for {filename}. {e}")

                    # Smooth mesh
                    try:
                        mesh = mesh.smooth(n_iter=20, relaxation_factor=0.1)
                    except Exception as e:
                        print(f"  Warning: Smoothing failed for {filename}. {e}")

                    # Find color
                    color = [0.5, 0.5, 0.5]  # Default grey
                    for key, rgb in self.colormap.items():
                        if key.lower() in filename.lower():
                            color = rgb
                            break

                    # Save mesh to cache
                    if self.cache is not None:
                        self.cache.save_mesh(all_file_paths, filename, mesh)

                    # Emit mesh loaded signal
                    self.mesh_loaded.emit(filename, mesh, color, system_name)

                except Exception as e:
                    print(f"  Error generating 3D mesh for {filename}: {e}")
                    self.error.emit(filename, str(e))

            except Exception as e:
                self.error.emit(filename, str(e))
                import traceback
                traceback.print_exc()

        # Save merged volume to cache and emit signal
        if self.seg_manager is not None and not self._cancelled:
            if not merged_from_cache:
                print(f"Merged volume complete: {np.count_nonzero(self.seg_manager.merged_volume)} non-zero voxels")
                # Save to cache
                if self.cache is not None:
                    self.cache.save_merged_volume(all_file_paths, self.seg_manager.merged_volume)
            self.merged_volume_ready.emit()

        self.finished.emit()


class SegmentationViewer3D(QWidget):
    """
    3D visualization widget for medical image segmentations.

    Memory Optimizations:
    - Meshes are NOT cached; VTK actors manage their own geometry data
    - Mesh decimation reduces triangle count to 30% for meshes > 10k faces
    - Float32 data types used instead of float64 where possible
    - Garbage collection forced after batch loading
    - Reduced smoothing iterations (20 instead of 50)
    - Progress dialogs shown during loading with cancellation support
    """

    # Signals
    loading_finished = pyqtSignal()  # Emitted when all loading finishes
    merged_volume_ready = pyqtSignal()  # Emitted when merged volume is built

    def __init__(self, nifti_files, parent=None, volume_data=None, affine=None, dims=None, intensity_min=0, intensity_max=255):
        super().__init__(parent)
        try:
            self.colormap = load_colormap("utils/colormap.json")
        except Exception as e:
            print(f"Error loading colormap: {e}. Using empty map.")
            self.colormap = {}

        # Use provided file list
        self.nifti_files = [Path(f) for f in nifti_files]
        print(f"Received {len(self.nifti_files)} segmentation files")

        # Categorize structures
        self.systems = categorize_structures(self.nifti_files)
        print(f"Organized into {len(self.systems)} systems")

        # Storage for actors (meshes are not cached to save memory)
        self.actors = {}  # {filename_stem: actor}

        # System visibility and opacity states
        self.system_visible = {system: True for system in self.systems.keys()}
        self.system_opacity = {system: 1.0 for system in self.systems.keys()}

        # Background loading
        self.load_worker = None
        self.load_progress_dialog = None

        # Caching
        self.cache = SegmentationCache()

        # UI components
        self.system_checkboxes = {}
        self.system_sliders = {}
        self.slice_sliders = {}  # Sliders for controlling plane positions

        # Volume data for plane rendering
        self.volume_data = volume_data
        self.affine = affine
        self.dims = dims
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

        # Plane mode properties
        self.planes_mode = False
        self.plane_actors = {}  # {'axial': actor, 'sagittal': actor, 'coronal': actor}
        self.current_slices = {'axial': 0, 'sagittal': 0, 'coronal': 0}
        if dims is not None:
            self.current_slices = {
                'axial': dims[2] // 2,
                'sagittal': dims[0] // 2,
                'coronal': dims[1] // 2
            }

        # Create UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the Qt layout with PyVista plotter and controls"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create PyVista plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background('black')
        main_layout.addWidget(self.plotter.interactor, stretch=3)

        # Create controls panel
        controls_widget = self.create_controls_panel()
        main_layout.addWidget(controls_widget, stretch=1)

    def create_controls_panel(self):
        """Create scrollable controls panel for system toggles and opacity"""
        controls_container = QWidget()
        # This objectName is the hook for the QSS file
        controls_container.setObjectName("controls_container_3d")
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("Anatomical Systems" if self.systems else "3D Controls")
        # This objectName is the hook for the QSS file
        title_label.setObjectName("anatomical_systems_title")
        controls_layout.addWidget(title_label)
        self.controls_title_label = title_label

        # Slice controls group (for planes mode)
        self.slice_controls_group = self.create_slice_controls()
        controls_layout.addWidget(self.slice_controls_group)
        self.slice_controls_group.hide()  # Initially hidden

        # Scroll area for systems
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Removed inline QSS

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # Create controls for each system (if any exist)
        if self.systems:
            for system_name in self.systems.keys():
                system_group = self.create_system_control(system_name)
                scroll_layout.addWidget(system_group)
        else:
            # Show message if no segmentations
            no_seg_label = QLabel("No segmentations loaded.\n\nUse Planes mode to view slices.")
            # Removed inline QSS
            no_seg_label.setAlignment(Qt.AlignCenter)
            scroll_layout.addWidget(no_seg_label)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        controls_layout.addWidget(scroll)
        self.systems_scroll_area = scroll

        # Reset all button
        reset_btn = QPushButton("Reset View")
        # Removed inline QSS
        reset_btn.clicked.connect(self.reset_camera)
        controls_layout.addWidget(reset_btn)

        return controls_container

    def create_slice_controls(self):
        """Create slice control sliders for planes mode"""
        group = QGroupBox("Slice Controls")
        # Removed inline QSS

        layout = QVBoxLayout()

        # Create slider for each plane type
        for plane_type, label_text in [('axial', 'Axial'), ('coronal', 'Coronal'), ('sagittal', 'Sagittal')]:
            plane_layout = QVBoxLayout()

            # Label
            label = QLabel(label_text)
            # Removed inline QSS
            plane_layout.addWidget(label)

            # Slider with value label
            slider_layout = QHBoxLayout()

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)

            # Set maximum based on actual data dimensions
            if self.dims is not None:
                if plane_type == 'axial':
                    max_val = self.dims[2] - 1
                    slider.setValue(min(self.current_slices['axial'], max_val))
                elif plane_type == 'coronal':
                    max_val = self.dims[1] - 1
                    slider.setValue(min(self.current_slices['coronal'], max_val))
                elif plane_type == 'sagittal':
                    max_val = self.dims[0] - 1
                    slider.setValue(min(self.current_slices['sagittal'], max_val))

                slider.setMaximum(max_val)

            # Removed inline QSS

            value_label = QLabel(f"{slider.value()}/{slider.maximum()}")
            # Removed inline QSS
            # Set a minimum width to prevent layout jumping
            value_label.setMinimumWidth(60)


            # Connect slider to update function
            slider.valueChanged.connect(
                lambda value, pt=plane_type, vl=value_label, s=slider: self.on_slice_slider_changed(pt, value, vl, s)
            )

            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)

            plane_layout.addLayout(slider_layout)
            layout.addLayout(plane_layout)

            self.slice_sliders[plane_type] = (slider, value_label)

        group.setLayout(layout)
        return group

    def on_slice_slider_changed(self, plane_type, value, value_label, slider):
        """Handle slice slider value changes"""
        value_label.setText(f"{value}/{slider.maximum()}")
        self.update_plane_position(plane_type, value)

    def update_slider_ranges(self):
        """Update slider ranges when new data is loaded"""
        if self.dims is None:
            return

        for plane_type, (slider, value_label) in self.slice_sliders.items():
            # Set maximum based on actual data dimensions
            if plane_type == 'axial':
                max_val = self.dims[2] - 1
                current_val = min(self.current_slices['axial'], max_val)
            elif plane_type == 'coronal':
                max_val = self.dims[1] - 1
                current_val = min(self.current_slices['coronal'], max_val)
            elif plane_type == 'sagittal':
                max_val = self.dims[0] - 1
                current_val = min(self.current_slices['sagittal'], max_val)

            slider.setMaximum(max_val)
            slider.setValue(current_val)
            value_label.setText(f"{current_val}/{max_val}")

    def create_system_control(self, system_name):
        """Create checkbox and opacity slider for a system"""
        group = QGroupBox(system_name)
        # Removed inline QSS

        layout = QVBoxLayout()

        # Visibility checkbox
        checkbox = QCheckBox("Visible")
        checkbox.setChecked(True)
        # Removed inline QSS
        checkbox.stateChanged.connect(lambda state: self.toggle_system(system_name, state == Qt.Checked))
        self.system_checkboxes[system_name] = checkbox
        layout.addWidget(checkbox)

        # Opacity slider
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        # Removed inline QSS
        opacity_layout.addWidget(opacity_label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(100)
        # Removed inline QSS
        slider.valueChanged.connect(lambda value: self.set_system_opacity(system_name, value / 100.0))
        self.system_sliders[system_name] = slider
        opacity_layout.addWidget(slider)

        opacity_value_label = QLabel("100%")
        # Removed inline QSS
        # Set a minimum width to prevent layout jumping
        opacity_value_label.setMinimumWidth(40)
        slider.valueChanged.connect(lambda value: opacity_value_label.setText(f"{value}%"))
        opacity_layout.addWidget(opacity_value_label)

        layout.addLayout(opacity_layout)

        group.setLayout(layout)
        return group

    def toggle_system(self, system_name, visible):
        """
        Toggle visibility of an entire system.
        Loads meshes on demand if they aren't already loaded.
        Uses synchronous loading (simpler, avoids threading complexity for small sets).
        """
        self.system_visible[system_name] = visible
        files_to_load = self.systems.get(system_name, [])

        # If hiding, just hide all actors
        if not visible:
            for nifti_file in files_to_load:
                filename = nifti_file.stem
                if filename in self.actors:
                    self.actors[filename].SetVisibility(False)
            self.plotter.render()
            return

        # If showing, count how many files need to be loaded
        files_needing_load = [f for f in files_to_load if f.stem not in self.actors]

        # Create progress dialog if we have files to load
        progress = None
        if files_needing_load:
            progress = QProgressDialog(
                f"Loading {system_name}...",
                "Cancel",
                0,
                len(files_needing_load),
                self
            )
            progress.setWindowTitle("Loading 3D Segmentations")
            progress.setWindowModality(Qt.ApplicationModal)  # Changed to ApplicationModal
            progress.setMinimumDuration(0)  # Show immediately
            progress.setValue(0)

        # Iterate over all files in this system
        load_count = 0
        for nifti_file in files_to_load:
            filename = nifti_file.stem

            # --- Show or Load Actor ---
            if filename in self.actors:
                # Actor already exists, just show it
                self.actors[filename].SetVisibility(True)
            else:
                # Actor doesn't exist, need to load it
                if progress:
                    progress.setLabelText(f"Loading {system_name}...\n{filename}")
                    progress.setValue(load_count)
                    QCoreApplication.processEvents()  # Update UI

                    # Check if user cancelled
                    if progress.wasCanceled():
                        print(f"Loading cancelled by user")
                        if progress:
                            progress.close()
                        self.plotter.render()
                        return

                print(f"Loading {filename}...")
                try:
                    # Generate mesh (not cached to save memory)
                    mesh, _ = nifti_to_surface(nifti_file)

                    # Find color
                    color = [0.5, 0.5, 0.5]  # Default grey
                    for key, rgb in self.colormap.items():
                        if key.lower() in filename.lower():
                            color = rgb
                            break

                    # Add to plotter and store actor
                    actor = self.plotter.add_mesh(
                        mesh,
                        color=color,
                        opacity=self.system_opacity[system_name],
                        show_edges=False,
                        lighting=True,
                        name=filename
                    )
                    self.actors[filename] = actor

                    # Don't cache mesh - let VTK/PyVista manage the memory
                    # This saves significant memory as we don't store duplicate data
                    del mesh

                    load_count += 1

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()

        # Close progress dialog
        if progress:
            progress.setValue(len(files_needing_load))
            progress.close()

            # Force garbage collection to free memory from deleted mesh objects
            gc.collect()
            print(f"Loaded {load_count} meshes for {system_name}. Memory freed via garbage collection.")

        self.plotter.render()

    def set_system_opacity(self, system_name, opacity):
        """Set opacity for an entire system."""
        self.system_opacity[system_name] = opacity

        # Iterate over all files in this system
        for nifti_file in self.systems.get(system_name, []):
            filename = nifti_file.stem

            # Only apply opacity if the actor exists (has been loaded)
            if filename in self.actors:
                self.actors[filename].GetProperty().SetOpacity(opacity)

        self.plotter.render()

    def reset_camera(self):
        """Reset camera to default view"""
        self.plotter.reset_camera()
        self.plotter.render()

    def initialize(self):
        """Initialize the 3D view UI and load default visible systems in background thread."""
        # Setup camera and display
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.show()

        # Initially load all visible systems (if any exist)
        if self.systems:
            print("Initializing 3D viewer... Loading default visible systems in background.")

            # Collect files to load
            files_to_load = []
            for system_name, is_visible in self.system_visible.items():
                if is_visible:
                    for nifti_file in self.systems[system_name]:
                        files_to_load.append((system_name, nifti_file))

            if files_to_load:
                # Create progress dialog
                self.load_progress_dialog = QProgressDialog(
                    "Loading 3D segmentations...",
                    "Cancel",
                    0,
                    len(files_to_load),
                    self
                )
                self.load_progress_dialog.setWindowTitle("Initializing 3D Viewer")
                self.load_progress_dialog.setWindowModality(Qt.WindowModal)
                self.load_progress_dialog.setMinimumDuration(0)
                self.load_progress_dialog.setValue(0)

                # Create and start worker thread
                self.load_worker = MeshLoadWorker(files_to_load, self.colormap, self.system_opacity)

                # Connect signals
                self.load_worker.progress.connect(self._on_load_progress)
                self.load_worker.mesh_loaded.connect(self._on_mesh_loaded)
                self.load_worker.finished.connect(self._on_load_finished)
                self.load_worker.error.connect(self._on_load_error)
                self.load_progress_dialog.canceled.connect(self._on_load_cancelled)

                # Start loading
                self.load_worker.start()
            else:
                # No files to load, emit signal immediately
                print("Initializing 3D viewer... No visible systems to load.")
                self.loading_finished.emit()

        else:
            # No segmentations at all, emit signal immediately
            print("Initializing 3D viewer... No segmentations to load.")
            self.loading_finished.emit()

    def _on_load_progress(self, message, current, total):
        """Update progress dialog."""
        if self.load_progress_dialog:
            self.load_progress_dialog.setLabelText(message)
            self.load_progress_dialog.setValue(current)

    def _on_mesh_loaded(self, filename, mesh, color, system_name):
        """Handle mesh loaded in background thread - add to plotter."""
        try:
            # Add to plotter (must be done in main thread)
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=self.system_opacity[system_name],
                show_edges=False,
                lighting=True,
                name=filename
            )
            self.actors[filename] = actor
            print(f"Added {filename} to 3D scene")

            # Render to show progress
            self.plotter.render()

        except Exception as e:
            print(f"Error adding mesh {filename} to plotter: {e}")

    def _on_load_error(self, filename, error_msg):
        """Handle load error."""
        print(f"Error loading {filename}: {error_msg}")

    def _on_load_finished(self):
        """Handle loading finished."""
        if self.load_progress_dialog:
            self.load_progress_dialog.setValue(self.load_progress_dialog.maximum())
            self.load_progress_dialog.close()
            self.load_progress_dialog = None

        # Force garbage collection
        gc.collect()
        print(f"3D loading complete. Loaded {len(self.actors)} meshes. Memory freed via garbage collection.")

        # Clean up worker
        if self.load_worker:
            self.load_worker.deleteLater()
            self.load_worker = None

        # Emit signal to notify that loading is complete
        self.loading_finished.emit()

    def _on_load_cancelled(self):
        """Handle user cancelling the load."""
        if self.load_worker:
            print("Cancelling 3D loading...")
            self.load_worker.cancel()
            self.load_worker.wait()  # Wait for thread to finish
            self.load_worker.deleteLater()
            self.load_worker = None

        if self.load_progress_dialog:
            self.load_progress_dialog.close()
            self.load_progress_dialog = None

        gc.collect()
        print("3D loading cancelled by user.")

        # Emit signal even when cancelled
        self.loading_finished.emit()

    def initialize_with_merge(self, seg_manager):
        """
        Initialize the 3D view with unified loading:
        1. Build merged volume for 2D views
        2. Load 3D meshes for 3D views
        Shows a single progress dialog for both operations.
        """
        # Setup camera and display
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.show()

        # Collect files to load
        files_to_load = []
        for system_name, is_visible in self.system_visible.items():
            if is_visible:
                for nifti_file in self.systems[system_name]:
                    files_to_load.append((system_name, nifti_file))

        if files_to_load:
            print("Starting unified segmentation loading (2D merge + 3D meshes)...")

            # Create progress dialog
            self.load_progress_dialog = QProgressDialog(
                "Loading segmentations...",
                "Cancel",
                0,
                len(files_to_load),  # Each file loaded once for both operations
                self
            )
            self.load_progress_dialog.setWindowTitle("Loading Segmentations")
            self.load_progress_dialog.setWindowModality(Qt.WindowModal)
            self.load_progress_dialog.setMinimumDuration(0)
            self.load_progress_dialog.setValue(0)

            # Create and start worker thread with seg_manager and cache
            self.load_worker = MeshLoadWorker(
                files_to_load,
                self.colormap,
                self.system_opacity,
                seg_manager=seg_manager,  # Pass seg_manager for merging
                cache=self.cache  # Pass cache for fast repeated loads
            )

            # Connect signals
            self.load_worker.progress.connect(self._on_load_progress)
            self.load_worker.mesh_loaded.connect(self._on_mesh_loaded)
            self.load_worker.merged_volume_ready.connect(self._on_merged_volume_ready)
            self.load_worker.finished.connect(self._on_load_finished)
            self.load_worker.error.connect(self._on_load_error)
            self.load_progress_dialog.canceled.connect(self._on_load_cancelled)

            # Start loading
            self.load_worker.start()
        else:
            # No files to load, emit signals immediately
            print("No visible systems to load.")
            self.loading_finished.emit()

    def _on_merged_volume_ready(self):
        """Handle merged volume being ready."""
        print("Merged volume ready for 2D views")
        # Forward the signal
        self.merged_volume_ready.emit()

    def create_plane_mesh(self, plane_type, slice_idx):
        """
        Create a textured plane mesh for the given plane type and slice index.

        Parameters:
        -----------
        plane_type : str
            One of 'axial', 'sagittal', 'coronal'
        slice_idx : int
            The slice index for this plane

        Returns:
        --------
        mesh : pv.PolyData
            The plane mesh with texture
        """
        if self.volume_data is None or self.affine is None or self.dims is None:
            return None

        # Clamp slice_idx to valid range
        if plane_type == 'axial':
            slice_idx = max(0, min(slice_idx, self.dims[2] - 1))
            slice_data = self.volume_data[:, :, slice_idx]
            slice_data = np.rot90(slice_data)
        elif plane_type == 'coronal':
            slice_idx = max(0, min(slice_idx, self.dims[1] - 1))
            slice_data = self.volume_data[:, slice_idx, :]
            slice_data = np.rot90(slice_data)
        elif plane_type == 'sagittal':
            slice_idx = max(0, min(slice_idx, self.dims[0] - 1))
            slice_data = self.volume_data[slice_idx, :, :]
            slice_data = np.rot90(slice_data)
        else:
            return None

        # Normalize intensity to 0-255 range
        slice_data = np.clip(slice_data, self.intensity_min, self.intensity_max)
        slice_data = ((slice_data - self.intensity_min) / (self.intensity_max - self.intensity_min) * 255).astype(np.uint8)

        # Get texture dimensions
        height, width = slice_data.shape

        # Get voxel spacing
        x_spacing = abs(self.affine[0, 0])
        y_spacing = abs(self.affine[1, 1])
        z_spacing = abs(self.affine[2, 2])

        # Create plane geometry
        if plane_type == 'axial':
            x_size = self.dims[0] * x_spacing
            y_size = self.dims[1] * y_spacing
            z_pos = slice_idx * z_spacing + self.affine[2, 3]

            plane = pv.Plane(
                center=(x_size/2 + self.affine[0, 3], y_size/2 + self.affine[1, 3], z_pos),
                direction=(0, 0, 1),
                i_size=x_size,
                j_size=y_size,
                i_resolution=width - 1,
                j_resolution=height - 1
            )

        elif plane_type == 'coronal':
            x_size = self.dims[0] * x_spacing
            z_size = self.dims[2] * z_spacing
            y_pos = slice_idx * y_spacing + self.affine[1, 3]

            plane = pv.Plane(
                center=(x_size/2 + self.affine[0, 3], y_pos, z_size/2 + self.affine[2, 3]),
                direction=(0, 1, 0),
                i_size=z_size,
                j_size=x_size,
                i_resolution=width - 1,
                j_resolution=height - 1
            )

        elif plane_type == 'sagittal':
            y_size = self.dims[1] * y_spacing
            z_size = self.dims[2] * z_spacing
            x_pos = slice_idx * x_spacing + self.affine[0, 3]

            plane = pv.Plane(
                center=(x_pos, y_size/2 + self.affine[1, 3], z_size/2 + self.affine[2, 3]),
                direction=(1, 0, 0),
                i_size=z_size,
                j_size=y_size,
                i_resolution=width - 1,
                j_resolution=height - 1
            )

        # Add texture coordinates
        plane.texture_map_to_plane(inplace=True)

        return plane, slice_data

    def toggle_planes_mode(self, enabled):
        """
        Toggle the display of intersecting planes.

        Parameters:
        -----------
        enabled : bool
            Whether to show or hide the planes
        """
        self.planes_mode = enabled

        if enabled:
            # Hide all segmentation actors
            for actor in self.actors.values():
                actor.SetVisibility(False)
            # Show planes
            self.create_planes()
            # Update UI to show slice controls
            self.controls_title_label.setText("Plane Controls")
            self.slice_controls_group.show()
            self.systems_scroll_area.hide()
        else:
            # Show segmentation actors that were visible
            for nifti_file in self.nifti_files:
                filename = nifti_file.stem
                if filename in self.actors:
                    # Check if the system this file belongs to is visible
                    for system_name, files in self.systems.items():
                        if nifti_file in files and self.system_visible.get(system_name, True):
                            self.actors[filename].SetVisibility(True)
                            break
            # Hide planes
            self.remove_planes()
            # Update UI to show system controls
            self.controls_title_label.setText("Anatomical Systems")
            self.slice_controls_group.hide()
            self.systems_scroll_area.show()

        self.plotter.render()

    def create_planes(self):
        """Create and display all three intersecting planes."""
        if self.volume_data is None:
            print("No volume data available for plane rendering")
            return

        # Remove existing planes first
        self.remove_planes()

        # Create each plane
        for plane_type in ['axial', 'sagittal', 'coronal']:
            slice_idx = self.current_slices[plane_type]
            result = self.create_plane_mesh(plane_type, slice_idx)

            if result is not None:
                plane_mesh, texture_data = result

                # Create texture from slice data
                texture = pv.Texture(texture_data)

                # Add to plotter
                actor = self.plotter.add_mesh(
                    plane_mesh,
                    texture=texture,
                    name=f"plane_{plane_type}",
                    opacity=1,
                    show_edges=False
                )

                self.plane_actors[plane_type] = actor

        print(f"Created {len(self.plane_actors)} planes")

    def remove_planes(self):
        """Remove all plane actors from the scene."""
        for plane_type in ['axial', 'sagittal', 'coronal']:
            plane_name = f"plane_{plane_type}"
            if plane_name in self.plotter.actors:
                self.plotter.remove_actor(plane_name)

        self.plane_actors.clear()

    def update_plane_position(self, plane_type, slice_idx):
        """
        Update the position of a specific plane.

        Parameters:
        -----------
        plane_type : str
            One of 'axial', 'sagittal', 'coronal'
        slice_idx : int
            The new slice index
        """
        if not self.planes_mode:
            return

        # Don't update if parent widget is not visible to prevent rendering conflicts
        if not self.isVisible():
            return

        self.current_slices[plane_type] = slice_idx

        # Create new plane geometry and texture
        result = self.create_plane_mesh(plane_type, slice_idx)
        if result is None:
            return

        plane_mesh, texture_data = result
        plane_name = f"plane_{plane_type}"

        # Check if actor already exists
        if plane_name in self.plotter.actors:
            # Update existing mesh in place to avoid flickering
            actor = self.plotter.actors[plane_name]

            # Get the mapper and update the input data
            mapper = actor.GetMapper()
            mapper.SetInputData(plane_mesh)

            # Update the texture
            texture = pv.Texture(texture_data)
            actor.SetTexture(texture)

            self.plotter.render()
        else:
            # Create new actor if it doesn't exist
            texture = pv.Texture(texture_data)
            actor = self.plotter.add_mesh(
                plane_mesh,
                texture=texture,
                name=plane_name,
                opacity=0.8,
                show_edges=False
            )
            self.plane_actors[plane_type] = actor
            self.plotter.render()

    def update_slices(self, slices_dict):
        """
        Update all plane positions from a dictionary of slice indices.

        Parameters:
        -----------
        slices_dict : dict
            Dictionary with keys 'axial', 'sagittal', 'coronal' and integer values
        """
        for plane_type in ['axial', 'sagittal', 'coronal']:
            if plane_type in slices_dict:
                self.update_plane_position(plane_type, slices_dict[plane_type])