import pyvista as pv
from pyvistaqt import QtInteractor
import nibabel as nib
import numpy as np
from pathlib import Path
from skimage import measure
import json
import gc
import collections
from .segmentation_cache import SegmentationCache
from .centerline_extractor import extract_centerline, compute_camera_positions, estimate_vessel_radius
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QCheckBox, QGroupBox, QScrollArea, QPushButton, QProgressDialog, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal, QTimer


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


def nifti_to_surface(nifti_path, smoothing=True, smoothing_iterations=50, decimation_target=0.1):
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
            mesh = mesh.smooth(n_iter=smoothing_iterations, relaxation_factor=0.15)
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
    systems = collections.defaultdict(list)

    # --- Keyword Definitions ---
    skeletal_limbs_keywords = ['humerus', 'scapula', 'clavicula', 'femur', 'hip']
    skeletal_skull_keywords = ['skull', 'jawbone']
    muscular_keywords = ['gluteus', 'autochthon', 'iliopsoas', 'muscle']

    # --- MERGED Brain Part list ---
    brain_parts_list = [
        "frontal_lobe", "parietal_lobe", "temporal_lobe", "occipital_lobe",
        "limbic_lobe", "insular_lobe", "cerebellum", "cerebrum",
        "Brainstem", "Vermis"
    ]
    # --- End Definitions ---

    for fname in filenames:
        # Ensure fname is a Path object if it's not already
        if not isinstance(fname, Path):
            fname = Path(fname)

        name_lower = fname.stem.lower()

        # --- Broad Categories (Checked First) ---
        if 'vertebrae' in name_lower or 'sacrum' in name_lower:
            systems['Skeletal - Spine'].append(fname)
        elif 'rib' in name_lower or 'sternum' in name_lower or 'costal' in name_lower:
            systems['Skeletal - Ribs'].append(fname)
        elif any(bone in name_lower for bone in skeletal_limbs_keywords):
            systems['Skeletal - Limbs'].append(fname)
        elif any(bone in name_lower for bone in skeletal_skull_keywords):
            systems['Skeletal - Skull'].append(fname)
        elif any(muscle in name_lower for muscle in muscular_keywords):
            systems['Muscular'].append(fname)
        elif 'artery' in name_lower or 'aorta' in name_lower or 'trunk' in name_lower:
            systems['Cardiovascular - Arteries'].append(fname)
        elif 'vein' in name_lower or 'vena' in name_lower:
            systems['Cardiovascular - Veins'].append(fname)
        elif 'heart' in name_lower:
            systems['Cardiovascular - Heart'].append(fname)

        # --- Dental & Cranial Categories ---
        # Check for 'pulp' first, as it's more specific
        elif 'pulp' in name_lower:
            systems['Dental - Pulp'].append(fname)
        # 'fdi' is a good indicator of a tooth segmentation
        elif 'fdi' in name_lower:
            systems['Dental - Teeth'].append(fname)
        elif 'alveolar_canal' in name_lower:
            systems['Dental - Nerves'].append(fname)
        elif 'maxillary_sinus' in name_lower:
            systems['Cranial - Sinuses'].append(fname)
        elif 'pharynx' in name_lower:
            systems['Digestive/Respiratory'].append(fname)

        # --- Specific Brain Categories (Checked Last) ---
        else:
            categorized = False
            # Loop through the specific brain parts
            for part_name in brain_parts_list:
                if part_name.lower() in name_lower:
                    systems[f'Nervous System - {part_name}'].append(fname)
                    categorized = True
                    break # Found its category

            if not categorized:
                systems['Other'].append(fname)

    # Convert back to a regular dict, removing empty systems
    systems = {k: v for k, v in systems.items() if v}

    return systems


class BloodFlowWorker(QThread):
    """
    Worker thread for computing vessel centerlines in the background.
    Prevents UI freezing during blood flow initialization.
    """
    # Signals
    progress = pyqtSignal(str, int, int)  # (message, current, total)
    centerline_computed = pyqtSignal(str, object, float)  # (mesh_name, centerline_points, radius)
    finished = pyqtSignal(int)  # (vessel_count)
    error = pyqtSignal(str, str)  # (mesh_name, error_message)

    def __init__(self, vessel_data):
        """
        Parameters:
        -----------
        vessel_data : list of tuples
            List of (mesh_name, actor) for vessels to process
        """
        super().__init__()
        self.vessel_data = vessel_data
        self._cancelled = False

    def cancel(self):
        """Cancel the computation."""
        self._cancelled = True

    def run(self):
        """Compute centerlines for all vessels."""
        total = len(self.vessel_data)
        vessel_count = 0

        for idx, (mesh_name, actor) in enumerate(self.vessel_data):
            if self._cancelled:
                break

            self.progress.emit(f"Computing centerline for {mesh_name}...", idx, total)

            try:
                # Get mesh from actor
                mapper = actor.GetMapper()
                mesh = mapper.GetInput()
                pv_mesh = pv.wrap(mesh)

                # Extract centerline
                centerline_points = extract_centerline(pv_mesh, resolution=50, smooth=True)

                if len(centerline_points) > 10:
                    # Estimate vessel radius
                    radius = estimate_vessel_radius(pv_mesh, centerline_points, sample_points=10)

                    # Emit result
                    self.centerline_computed.emit(mesh_name, centerline_points, radius)
                    vessel_count += 1
                else:
                    self.error.emit(mesh_name, "Centerline too short")

            except Exception as e:
                self.error.emit(mesh_name, str(e))

        self.finished.emit(vessel_count)


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
                        mesh = mesh.smooth(n_iter=50, relaxation_factor=0.15)
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

        # Centerline walkthrough properties
        self.walkthrough_active = False
        self.walkthrough_timer = QTimer()
        self.walkthrough_timer.timeout.connect(self._update_walkthrough)
        self.camera_path = []  # List of camera positions along centerline
        self.current_path_index = 0
        self.walkthrough_speed = 50  # milliseconds per step
        self.centerline_mesh_name = None  # Name of mesh being walked through
        self.centerline_actor = None  # Actor for visualizing centerline
        self.centerline_points = None  # Stored centerline points for recomputing camera path
        self.look_ahead_distance = 10  # Look ahead distance in number of points
        self.focus_mode_active = False
        self.focused_mesh_name = None
        self.hidden_actors_before_focus = {}

        # Blood flow visualization properties
        self.blood_flow_active = False
        self.heart_rate = 60  # BPM (beats per minute)
        self.blood_flow_timer = QTimer()
        self.blood_flow_timer.timeout.connect(self._update_blood_flow)
        self.flow_phase = 0.0  # Current phase of the cardiac cycle (0.0 to 1.0)
        self.vessel_centerlines = {}  # Store centerlines for each vessel {mesh_name: points}
        self.vessel_radii = {}  # Store estimated radii for each vessel {mesh_name: radius}
        self.vessel_flow_rings = {}  # Store ring actors for each vessel {mesh_name: [ring_actors]}
        self.vessel_ring_positions = {}  # Store ring positions {mesh_name: [position_indices]}
        self.num_rings_per_vessel = 5  # Number of rings traveling down each vessel (increased for better visual)
        self.blood_flow_worker = None  # Worker thread for computing centerlines

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

        # Flythrough controls group
        self.walkthrough_controls_group = self.create_walkthrough_controls()
        controls_layout.addWidget(self.walkthrough_controls_group)

        # Blood flow controls group
        self.blood_flow_controls_group = self.create_blood_flow_controls()
        controls_layout.addWidget(self.blood_flow_controls_group)

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

    def create_walkthrough_controls(self):
        """Create centerline walkthrough controls"""
        group = QGroupBox("Flythrough")

        layout = QVBoxLayout()

        # Mesh selection
        mesh_layout = QVBoxLayout()
        mesh_label = QLabel("Select Mesh:")
        mesh_layout.addWidget(mesh_label)

        self.mesh_selector = QComboBox()
        self.mesh_selector.addItem("-- Select a mesh --")
        mesh_layout.addWidget(self.mesh_selector)
        layout.addLayout(mesh_layout)

        # Play/Pause button (computes centerline automatically on first play)
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self._toggle_walkthrough)
        self.play_pause_btn.setEnabled(False)
        layout.addWidget(self.play_pause_btn)

        # Stop button
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop_walkthrough)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        # Speed control
        speed_layout = QVBoxLayout()
        speed_label = QLabel("Speed:")
        speed_layout.addWidget(speed_label)

        speed_slider_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        self.speed_value_label = QLabel("50 ms")
        self.speed_value_label.setMinimumWidth(60)

        speed_slider_layout.addWidget(self.speed_slider)
        speed_slider_layout.addWidget(self.speed_value_label)
        speed_layout.addLayout(speed_slider_layout)
        layout.addLayout(speed_layout)

        # Progress slider
        progress_layout = QVBoxLayout()
        progress_label = QLabel("Progress:")
        progress_layout.addWidget(progress_label)

        progress_slider_layout = QHBoxLayout()
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.setEnabled(False)
        self.progress_slider.valueChanged.connect(self._on_progress_changed)

        self.progress_value_label = QLabel("0%")
        self.progress_value_label.setMinimumWidth(60)

        progress_slider_layout.addWidget(self.progress_slider)
        progress_slider_layout.addWidget(self.progress_value_label)
        progress_layout.addLayout(progress_slider_layout)
        layout.addLayout(progress_layout)

        # Connect mesh selector change
        self.mesh_selector.currentIndexChanged.connect(self._on_mesh_selected)

        group.setLayout(layout)
        return group

    def create_blood_flow_controls(self):
        """Create blood flow visualization controls"""
        group = QGroupBox("Blood Flow Visualization")

        layout = QVBoxLayout()

        # Heart rate input
        hr_layout = QHBoxLayout()
        hr_label = QLabel("Heart Rate (BPM):")
        hr_layout.addWidget(hr_label)

        self.heart_rate_spinbox = QSpinBox()
        self.heart_rate_spinbox.setMinimum(40)
        self.heart_rate_spinbox.setMaximum(200)
        self.heart_rate_spinbox.setValue(60)
        self.heart_rate_spinbox.setSuffix(" BPM")
        self.heart_rate_spinbox.valueChanged.connect(self._on_heart_rate_changed)
        hr_layout.addWidget(self.heart_rate_spinbox)

        layout.addLayout(hr_layout)

        # Start/Stop button
        self.blood_flow_btn = QPushButton("▶ Start Blood Flow")
        self.blood_flow_btn.clicked.connect(self._toggle_blood_flow)
        layout.addWidget(self.blood_flow_btn)

        # Info label
        info_label = QLabel("Visualizes blood flow with\nanimated rings in vessels")
        info_label.setStyleSheet("color: #888; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group.setLayout(layout)
        return group

    def toggle_focus_navigation(self, enabled):
        """
        Toggle focus navigation mode.
        When enabled, clicking on a mesh will hide all other meshes.
        
        Parameters:
        -----------
        enabled : bool
            Whether to enable or disable focus navigation mode
        """
        if enabled:
            print("Focus navigation enabled - RIGHT-CLICK on any organ to focus")
            try:
                # Disable any existing picking first
                try:
                    self.plotter.disable_picking()
                except:
                    pass
                
                # Enable picking in the plotter with error handling
                self.plotter.enable_mesh_picking(
                    callback=self._on_mesh_picked,
                    show_message=False,
                    use_picker='cell',  # Use cell picker for better accuracy
                    show=False  # Don't show the picked mesh separately
                )
                self.focus_mode_active = True
            except Exception as e:
                print(f"Error enabling picking: {e}")
                self.focus_mode_active = False
        else:
            print("Focus navigation disabled")
            try:
                # Disable picking
                self.plotter.disable_picking()
            except Exception as e:
                print(f"Error disabling picking: {e}")
            
            self.focus_mode_active = False
            
            # Restore all hidden meshes if we were in focus mode
            if self.focused_mesh_name is not None:
                self._unfocus_all_meshes()
    
    def _on_mesh_picked(self, mesh):
        """
        Callback when a mesh is picked in focus navigation mode.
        
        Parameters:
        -----------
        mesh : pv.PolyData or vtk object
            The picked mesh
        """
        if not self.focus_mode_active:
            return
        
        try:
            # Handle None or invalid mesh
            if mesh is None:
                return
            
            # Convert to PyVista mesh if needed
            if not isinstance(mesh, pv.PolyData):
                try:
                    mesh = pv.wrap(mesh)
                except:
                    return
            
            # Find which actor was picked by comparing mesh data
            picked_mesh_name = None
            picked_n_points = mesh.GetNumberOfPoints()
            picked_n_cells = mesh.GetNumberOfCells()
            
            for mesh_name, actor in self.actors.items():
                if not actor.GetVisibility():
                    continue
                    
                try:
                    mapper = actor.GetMapper()
                    actor_mesh = mapper.GetInput()
                    
                    # Compare both points and cells for better accuracy
                    if (actor_mesh.GetNumberOfPoints() == picked_n_points and 
                        actor_mesh.GetNumberOfCells() == picked_n_cells):
                        picked_mesh_name = mesh_name
                        break
                except:
                    continue
            
            if picked_mesh_name is None:
                print("Could not identify picked mesh")
                return
            
            # If already focused on this mesh, unfocus
            if self.focused_mesh_name == picked_mesh_name:
                print(f"Unfocusing from: {picked_mesh_name}")
                self._unfocus_all_meshes()
                return
            
            # Focus on the picked mesh
            self._focus_on_mesh(picked_mesh_name)
            
        except Exception as e:
            print(f"Error in mesh picking: {e}")
            return
    
    def _focus_on_mesh(self, mesh_name):
        """
        Hide all meshes except the focused one and zoom to fit it.
        
        Parameters:
        -----------
        mesh_name : str
            Name of the mesh to focus on
        """
        print(f"Focusing on: {mesh_name}")
        
        # Store current visibility states before hiding
        self.hidden_actors_before_focus = {}
        
        # Hide all other meshes
        for name, actor in self.actors.items():
            # Store previous visibility
            was_visible = actor.GetVisibility()
            self.hidden_actors_before_focus[name] = was_visible
            
            if name != mesh_name:
                actor.SetVisibility(False)
            else:
                actor.SetVisibility(True)
        
        self.focused_mesh_name = mesh_name
        
        # Zoom to fit the focused mesh
        if mesh_name in self.actors:
            try:
                # Get the focused actor
                focused_actor = self.actors[mesh_name]
                
                # Get the bounds of the mesh
                mapper = focused_actor.GetMapper()
                mesh = mapper.GetInput()
                
                # Reset camera to focus on this mesh
                self.plotter.reset_camera_clipping_range()
                
                # Use PyVista's built-in camera positioning for the mesh
                # Get bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
                bounds = mesh.GetBounds()
                
                # Calculate center of the mesh
                center = [
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ]
                
                # Calculate size of the mesh
                size = max(
                    bounds[1] - bounds[0],  # x size
                    bounds[3] - bounds[2],  # y size
                    bounds[5] - bounds[4]   # z size
                )
                
                # Set camera position
                camera = self.plotter.camera
                
                # Position camera at a distance proportional to mesh size
                distance = size * 2.5  # Adjust multiplier for desired zoom level
                
                # Set camera to look at the mesh from a nice angle
                # Position camera at 45 degrees in both horizontal and vertical
                import numpy as np
                angle_h = np.radians(45)
                angle_v = np.radians(30)
                
                camera.position = [
                    center[0] + distance * np.cos(angle_h) * np.cos(angle_v),
                    center[1] + distance * np.sin(angle_h) * np.cos(angle_v),
                    center[2] + distance * np.sin(angle_v)
                ]
                
                camera.focal_point = center
                camera.up = [0, 0, 1]  # Z-axis is up
                
                # Reset the camera clipping range to avoid clipping the mesh
                self.plotter.reset_camera_clipping_range()
                
                print(f"Zoomed to fit: {mesh_name}")
                
            except Exception as e:
                print(f"Error zooming to mesh: {e}")
        
        self.plotter.render()
    
    def _unfocus_all_meshes(self):
        """
        Restore visibility of all meshes to their pre-focus state and reset camera.
        """
        print("Unfocusing - restoring all meshes")
        
        # Restore previous visibility states
        for name, was_visible in self.hidden_actors_before_focus.items():
            if name in self.actors:
                self.actors[name].SetVisibility(was_visible)
        
        self.focused_mesh_name = None
        self.hidden_actors_before_focus = {}
        
        # Reset camera to show all visible meshes
        try:
            self.plotter.reset_camera()
            print("Camera reset to show all meshes")
        except Exception as e:
            print(f"Error resetting camera: {e}")
        
        self.plotter.render()

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
                        smooth_shading=True,
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

        # Update mesh selector if any meshes were loaded
        if load_count > 0:
            self._update_mesh_selector()

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

    def set_background_color(self, is_dark_mode):
        """Set the plotter background color based on theme"""
        if is_dark_mode:
            self.plotter.set_background('black')
        else:
            self.plotter.set_background('white')
        self.plotter.render()

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
                smooth_shading=True,
                name=filename
            )
            self.actors[filename] = actor
            print(f"Added {filename} to 3D scene")

            # Update mesh selector for walkthrough
            self._update_mesh_selector()

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
            # Hide walkthrough and blood flow controls in planes mode
            self.walkthrough_controls_group.hide()
            self.blood_flow_controls_group.hide()
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
            # Show walkthrough and blood flow controls in surface mode
            self.walkthrough_controls_group.show()
            self.blood_flow_controls_group.show()

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
                    lighting=False,
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
                lighting=False,
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

    # ========== Centerline Flythrough Methods ==========

    def _is_flythrough_compatible(self, mesh_name):
        """
        Check if a mesh is suitable for fly-through navigation.
        Only tubular/vessel-like structures make sense for centerline fly-through.

        Parameters:
        -----------
        mesh_name : str
            The name of the mesh (filename stem)

        Returns:
        --------
        bool
            True if the mesh is suitable for fly-through, False otherwise
        """
        name_lower = mesh_name.lower()

        # Tubular structures that make sense for fly-through
        flythrough_keywords = [
            'aorta',
            'artery',
            'vein',
            'vena',
            'vessel',
            'trunk',  # brachiocephalic trunk
            'coronary',
            'pulmonary_artery',
            'pulmonary_vein',
            'portal',
            'splenic'
        ]

        # Check if mesh name contains any flythrough-compatible keywords
        return any(keyword in name_lower for keyword in flythrough_keywords)

    def _update_mesh_selector(self):
        """Update the mesh selector dropdown with loaded meshes (filtered for fly-through compatibility)"""
        self.mesh_selector.clear()
        self.mesh_selector.addItem("-- Select a mesh --")

        # Only add meshes that are compatible with fly-through
        compatible_meshes = []
        for mesh_name in sorted(self.actors.keys()):
            if self._is_flythrough_compatible(mesh_name):
                compatible_meshes.append(mesh_name)

        # Add compatible meshes to selector
        for mesh_name in compatible_meshes:
            # Remove .nii or .nii.gz extension from display name
            display_name = mesh_name.replace('.nii.gz', '').replace('.nii', '')
            self.mesh_selector.addItem(display_name, mesh_name)  # Store original as data

        if len(compatible_meshes) == 0:
            # Show message if no compatible meshes loaded
            self.mesh_selector.addItem("(No vessel meshes loaded)")
            self.mesh_selector.setEnabled(False)
        else:
            self.mesh_selector.setEnabled(True)

    def _on_mesh_selected(self, index):
        """Handle mesh selection change"""
        if index > 0:  # 0 is the "Select a mesh" placeholder
            # Get the original mesh name from item data
            self.centerline_mesh_name = self.mesh_selector.currentData()
            if not self.centerline_mesh_name:
                # Fallback if data not set
                self.centerline_mesh_name = self.mesh_selector.currentText()
            self.play_pause_btn.setEnabled(True)
            print(f"Selected mesh: {self.centerline_mesh_name}")
            # Reset centerline when new mesh is selected
            self.centerline_points = None
            self.camera_path = []
            self.play_pause_btn.setText("▶ Play")
        else:
            self.centerline_mesh_name = None
            self.play_pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def _compute_centerline(self):
        """Compute the centerline for the selected mesh"""
        if not self.centerline_mesh_name or self.centerline_mesh_name not in self.actors:
            print("No valid mesh selected")
            return False

        # Get the mesh from the actor
        actor = self.actors[self.centerline_mesh_name]
        mapper = actor.GetMapper()
        mesh = mapper.GetInput()

        # Convert VTK data to PyVista mesh
        pv_mesh = pv.wrap(mesh)

        print(f"Computing centerline for {self.centerline_mesh_name}...")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setText("Computing...")
        QCoreApplication.processEvents()

        try:
            # Extract centerline
            centerline_points = extract_centerline(pv_mesh, resolution=80, smooth=True)

            if len(centerline_points) == 0:
                print("Failed to extract centerline - no points found")
                self.play_pause_btn.setText("▶ Play")
                self.play_pause_btn.setEnabled(True)
                return False

            # Store centerline points for recomputing camera path
            self.centerline_points = centerline_points

            # Visualize centerline (currently hidden)
            self._visualize_centerline(centerline_points)

            # Set initial look-ahead distance based on path length
            # Look ahead by ~5% of the total path length for smooth camera movement
            initial_lookahead = max(5, len(centerline_points) // 20)
            self.look_ahead_distance = initial_lookahead

            # Compute camera positions along centerline
            self._recompute_camera_path()

            print(f"Computed camera path with {len(self.camera_path)} positions")

            # Enable playback controls
            self.play_pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.progress_slider.setEnabled(True)
            self.progress_slider.setMaximum(max(1, len(self.camera_path) - 1))

            # Reset progress
            self.current_path_index = 0
            self._update_progress_display()

            return True

        except Exception as e:
            print(f"Error computing centerline: {e}")
            import traceback
            traceback.print_exc()
            self.play_pause_btn.setText("▶ Play")
            self.play_pause_btn.setEnabled(True)
            return False

    def _visualize_centerline(self, centerline_points):
        """Visualize the centerline as a tube in the 3D view"""
        # Remove existing centerline if any
        if self.centerline_actor is not None:
            self.plotter.remove_actor("centerline_path")
            self.centerline_actor = None

        # Don't visualize centerline - keep it hidden so user can see the structure
        # The centerline is computed and used for camera path only
        print("Centerline computed (hidden from view)")

    def _toggle_walkthrough(self):
        """Toggle play/pause for the walkthrough"""
        # If no camera path exists yet, compute centerline first
        if not self.camera_path:
            if not self.centerline_mesh_name:
                print("No mesh selected")
                return

            # Compute centerline automatically on first play
            print("Computing centerline automatically...")
            success = self._compute_centerline()

            if not success:
                print("Failed to compute centerline")
                return

        if self.walkthrough_active:
            # Pause
            self.walkthrough_timer.stop()
            self.walkthrough_active = False
            self.play_pause_btn.setText("▶ Play")
            print("Flythrough paused")
        else:
            # Play
            self.walkthrough_timer.start(self.walkthrough_speed)
            self.walkthrough_active = True
            self.play_pause_btn.setText("⏸ Pause")
            print("Flythrough started")

    def _stop_walkthrough(self):
        """Stop the walkthrough and reset to start"""
        self.walkthrough_timer.stop()
        self.walkthrough_active = False
        self.play_pause_btn.setText("▶ Play")
        self.current_path_index = 0
        self._update_progress_display()

        # Reset camera
        self.plotter.reset_camera()
        self.plotter.render()
        print("Flythrough stopped")

    def _on_speed_changed(self, value):
        """Handle speed slider changes"""
        self.walkthrough_speed = value
        self.speed_value_label.setText(f"{value} ms")

        # Update timer if active
        if self.walkthrough_active:
            self.walkthrough_timer.setInterval(self.walkthrough_speed)

    def _recompute_camera_path(self):
        """Recompute camera path from stored centerline points"""
        if self.centerline_points is None:
            return

        self.camera_path = compute_camera_positions(
            self.centerline_points,
            look_ahead_distance=self.look_ahead_distance
        )

        # Update progress slider maximum
        if len(self.camera_path) > 0:
            self.progress_slider.setMaximum(max(1, len(self.camera_path) - 1))

        # If currently viewing, update camera immediately
        if self.current_path_index < len(self.camera_path):
            self._update_camera_position()

    def _on_progress_changed(self, value):
        """Handle manual progress slider changes"""
        if not self.walkthrough_active:  # Only allow manual seeking when paused
            self.current_path_index = value
            self._update_camera_position()
            self._update_progress_display()

    def _update_walkthrough(self):
        """Update camera position during walkthrough animation"""
        if not self.camera_path:
            return

        # Update camera position
        self._update_camera_position()

        # Advance to next position
        self.current_path_index += 1

        # Check if we've reached the end
        if self.current_path_index >= len(self.camera_path):
            # Loop back to start
            self.current_path_index = 0

        # Update progress display
        self._update_progress_display()

    def _update_camera_position(self):
        """Update the camera to the current path position"""
        if not self.camera_path or self.current_path_index >= len(self.camera_path):
            return

        camera_config = self.camera_path[self.current_path_index]

        # Update PyVista camera
        camera = self.plotter.camera
        camera.position = camera_config['position']
        camera.focal_point = camera_config['focal_point']
        camera.up = camera_config['up']

        self.plotter.render()

    def _update_progress_display(self):
        """Update the progress slider and label"""
        if not self.camera_path:
            return

        # Block signals to avoid triggering _on_progress_changed
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(self.current_path_index)
        self.progress_slider.blockSignals(False)

        # Update percentage label
        if len(self.camera_path) > 0:
            percentage = int((self.current_path_index / len(self.camera_path)) * 100)
            self.progress_value_label.setText(f"{percentage}%")

    # ========== Blood Flow Visualization Methods ==========

    def _is_artery(self, mesh_name):
        """Check if a mesh is an artery"""
        name_lower = mesh_name.lower()
        artery_keywords = ['aorta', 'artery', 'arterial', 'trunk', 'coronary']
        return any(keyword in name_lower for keyword in artery_keywords)

    def _is_vein(self, mesh_name):
        """Check if a mesh is a vein"""
        name_lower = mesh_name.lower()
        vein_keywords = ['vein', 'vena', 'venous', 'portal', 'splenic']
        return any(keyword in name_lower for keyword in vein_keywords)

    def _on_heart_rate_changed(self, value):
        """Handle heart rate changes"""
        self.heart_rate = value
        # No need to update timer here, it updates continuously

    def _toggle_blood_flow(self):
        """Toggle blood flow visualization on/off"""
        if self.blood_flow_active:
            # Stop blood flow
            self._stop_blood_flow()
        else:
            # Start blood flow
            self._start_blood_flow()

    def _start_blood_flow(self):
        """Start the blood flow visualization with moving rings (using background thread)"""
        print(f"Starting blood flow visualization at {self.heart_rate} BPM...")
        self.blood_flow_btn.setEnabled(False)
        self.blood_flow_btn.setText("Computing...")

        # Collect vessel data for processing
        vessel_data = []
        for mesh_name, actor in self.actors.items():
            if actor.GetVisibility() and (self._is_artery(mesh_name) or self._is_vein(mesh_name)):
                vessel_data.append((mesh_name, actor))

        if len(vessel_data) == 0:
            print("No visible vessels found for blood flow visualization")
            self.blood_flow_btn.setText("▶ Start Blood Flow")
            self.blood_flow_btn.setEnabled(True)
            return

        # Create progress dialog
        self.blood_flow_progress_dialog = QProgressDialog(
            "Computing vessel centerlines...",
            "Cancel",
            0,
            len(vessel_data),
            self
        )
        self.blood_flow_progress_dialog.setWindowTitle("Blood Flow Initialization")
        self.blood_flow_progress_dialog.setWindowModality(Qt.WindowModal)
        self.blood_flow_progress_dialog.setMinimumDuration(0)
        self.blood_flow_progress_dialog.setValue(0)

        # Create worker thread
        self.blood_flow_worker = BloodFlowWorker(vessel_data)

        # Connect signals
        self.blood_flow_worker.progress.connect(self._on_blood_flow_progress)
        self.blood_flow_worker.centerline_computed.connect(self._on_centerline_computed)
        self.blood_flow_worker.finished.connect(self._on_blood_flow_finished)
        self.blood_flow_worker.error.connect(self._on_blood_flow_error)
        self.blood_flow_progress_dialog.canceled.connect(self._on_blood_flow_cancelled)

        # Start worker
        self.blood_flow_worker.start()

    def _on_blood_flow_progress(self, message, current, total):
        """Handle blood flow computation progress updates"""
        if self.blood_flow_progress_dialog:
            self.blood_flow_progress_dialog.setLabelText(message)
            self.blood_flow_progress_dialog.setValue(current)

    def _on_centerline_computed(self, mesh_name, centerline_points, radius):
        """Handle computed centerline result"""
        # Store centerline and radius
        self.vessel_centerlines[mesh_name] = centerline_points
        self.vessel_radii[mesh_name] = radius

        # Initialize ring positions evenly spaced along centerline
        spacing = len(centerline_points) // self.num_rings_per_vessel
        if spacing < 1:
            spacing = 1
        initial_positions = [i * spacing for i in range(self.num_rings_per_vessel)]
        self.vessel_ring_positions[mesh_name] = initial_positions

        print(f"  ✓ {mesh_name}: centerline computed (radius: {radius:.1f})")

    def _on_blood_flow_error(self, mesh_name, error_msg):
        """Handle blood flow computation error"""
        print(f"  ✗ {mesh_name}: {error_msg}")

    def _on_blood_flow_finished(self, vessel_count):
        """Handle blood flow computation completion"""
        if self.blood_flow_progress_dialog:
            self.blood_flow_progress_dialog.close()
            self.blood_flow_progress_dialog = None

        # Clean up worker
        if self.blood_flow_worker:
            self.blood_flow_worker.deleteLater()
            self.blood_flow_worker = None

        if vessel_count == 0:
            print("No vessels processed for blood flow visualization")
            self.blood_flow_btn.setText("▶ Start Blood Flow")
            self.blood_flow_btn.setEnabled(True)
            return

        # Create rings for all vessels
        print(f"Creating flow rings for {vessel_count} vessels...")
        for mesh_name in self.vessel_centerlines.keys():
            if mesh_name in self.actors:
                self._create_flow_rings(mesh_name, self.actors[mesh_name])

        print(f"Blood flow visualization ready for {vessel_count} vessels")

        # Reset phase
        self.flow_phase = 0.0

        # Start timer - update at 60 FPS for smooth animation
        self.blood_flow_timer.start(16)  # ~60 FPS

        # Update button
        self.blood_flow_active = True
        self.blood_flow_btn.setText("⏹ Stop Blood Flow")
        self.blood_flow_btn.setEnabled(True)

    def _on_blood_flow_cancelled(self):
        """Handle user cancelling blood flow computation"""
        if self.blood_flow_worker:
            print("Cancelling blood flow computation...")
            self.blood_flow_worker.cancel()
            self.blood_flow_worker.wait()
            self.blood_flow_worker.deleteLater()
            self.blood_flow_worker = None

        if self.blood_flow_progress_dialog:
            self.blood_flow_progress_dialog.close()
            self.blood_flow_progress_dialog = None

        self.blood_flow_btn.setText("▶ Start Blood Flow")
        self.blood_flow_btn.setEnabled(True)
        print("Blood flow computation cancelled")

    def _stop_blood_flow(self):
        """Stop the blood flow visualization and remove ring actors"""
        self.blood_flow_timer.stop()
        self.blood_flow_active = False
        self.blood_flow_btn.setText("▶ Start Blood Flow")

        # Remove all flow ring actors
        for mesh_name, ring_actors in self.vessel_flow_rings.items():
            for ring_actor in ring_actors:
                if ring_actor is not None:
                    try:
                        self.plotter.remove_actor(ring_actor)
                    except:
                        pass

        # Clear data structures
        self.vessel_centerlines.clear()
        self.vessel_radii.clear()
        self.vessel_flow_rings.clear()
        self.vessel_ring_positions.clear()

        self.plotter.render()
        print("Blood flow visualization stopped")

    def _create_ring_mesh_for_vessel(self, vessel_radius):
        """
        Create a ring mesh for a specific vessel radius.

        Parameters:
        -----------
        vessel_radius : float
            Radius of the vessel

        Returns:
        --------
        mesh : pv.PolyData
            Ring mesh with appropriate size
        """
        # Create circle points in XY plane with vessel-specific radius
        theta = np.linspace(0, 2*np.pi, 16, endpoint=False)

        circle_x = vessel_radius * np.cos(theta)
        circle_y = vessel_radius * np.sin(theta)
        circle_z = np.zeros_like(theta)

        circle_points = np.column_stack([circle_x, circle_y, circle_z])

        # Create polyline
        lines = np.full((len(theta), 3), 2, dtype=np.int32)
        lines[:, 1] = np.arange(len(theta))
        lines[:, 2] = np.roll(np.arange(len(theta)), -1)

        poly = pv.PolyData(circle_points, lines=lines)

        # Apply tube filter with radius proportional to vessel size
        tube_radius = max(0.3, vessel_radius * 0.15)  # 15% of vessel radius
        tube = poly.tube(radius=tube_radius, n_sides=6)

        return tube

    def _create_flow_rings(self, mesh_name, vessel_actor):
        """
        Create flow ring actors for a vessel with proper sizing.

        Parameters:
        -----------
        mesh_name : str
            Name of the vessel mesh
        vessel_actor : vtkActor
            The vessel actor to match color
        """
        if mesh_name not in self.vessel_centerlines:
            return False

        centerline = self.vessel_centerlines[mesh_name]
        positions = self.vessel_ring_positions[mesh_name]

        # Get vessel-specific radius
        vessel_radius = self.vessel_radii.get(mesh_name, 5.0)

        # Create ring mesh for this vessel
        ring_mesh = self._create_ring_mesh_for_vessel(vessel_radius)

        # Determine ring color based on vessel type
        if self._is_artery(mesh_name):
            ring_color = [1.0, 0.3, 0.1]  # Bright orange-red
            ring_opacity = 0.8
        else:
            ring_color = [0.2, 0.5, 1.0]  # Bright blue
            ring_opacity = 0.7

        # Create ring actors using vessel-specific mesh
        ring_actors = []
        for pos_idx in positions:
            if pos_idx < len(centerline):
                # Add mesh with transform
                actor = self.plotter.add_mesh(
                    ring_mesh,
                    color=ring_color,
                    opacity=ring_opacity,
                    lighting=True,
                    smooth_shading=True
                )

                # Position will be updated via transform in update loop
                if actor:
                    ring_actors.append(actor)

        self.vessel_flow_rings[mesh_name] = ring_actors
        print(f"    Created {len(ring_actors)} rings for {mesh_name} (radius: {vessel_radius:.1f})")
        return len(ring_actors) > 0

    def _update_ring_transform(self, actor, centerline, position_idx):
        """
        Update ring position using VTK transforms (much faster than recreating geometry).

        Parameters:
        -----------
        actor : vtkActor
            The ring actor to update
        centerline : np.ndarray
            Centerline points
        position_idx : int
            Position index along centerline
        """
        import vtk

        if position_idx >= len(centerline) - 1:
            return

        # Get position and direction
        center = centerline[position_idx]

        # Calculate direction vector (tangent)
        if position_idx < len(centerline) - 1:
            direction = centerline[position_idx + 1] - centerline[position_idx]
        else:
            direction = centerline[position_idx] - centerline[position_idx - 1]

        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return
        direction = direction / direction_norm

        # Create transform
        transform = vtk.vtkTransform()

        # Calculate rotation to align Z-axis with direction
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            rotation_angle_deg = np.degrees(rotation_angle)

            # Apply rotation and translation
            transform.Translate(center)
            transform.RotateWXYZ(rotation_angle_deg, rotation_axis[0], rotation_axis[1], rotation_axis[2])
        else:
            transform.Translate(center)

        # Set actor transform
        actor.SetUserTransform(transform)

    def _update_blood_flow(self):
        """Update blood flow animation by moving rings along vessels using transforms"""
        # Update cardiac phase for smooth timing
        cycle_duration = 60.0 / self.heart_rate
        phase_increment = 1.0 / (cycle_duration * 60)
        self.flow_phase += phase_increment
        if self.flow_phase >= 1.0:
            self.flow_phase = 0.0

        # Update each vessel's ring positions
        for mesh_name in list(self.vessel_centerlines.keys()):
            if mesh_name not in self.vessel_ring_positions:
                continue

            centerline = self.vessel_centerlines[mesh_name]
            positions = self.vessel_ring_positions[mesh_name]
            ring_actors = self.vessel_flow_rings.get(mesh_name, [])

            # Calculate movement speed based on vessel type and heart rate
            if self._is_artery(mesh_name):
                # Arteries: pulsatile flow
                base_speed = self.heart_rate / 60.0 * 2.0

                # Pulsatile component
                pulse = np.sin(self.flow_phase * 2 * np.pi)
                speed = base_speed * (1.0 + 0.5 * pulse)  # Vary speed ±50%
            else:
                # Veins: steady, slower flow
                speed = self.heart_rate / 60.0 * 0.8

            # Update ring positions using transforms (fast!)
            for i in range(len(positions)):
                positions[i] += speed

                # Wrap around when reaching end
                if positions[i] >= len(centerline) - 1:
                    positions[i] = 0

                # Update ring transform (no geometry recreation needed)
                if i < len(ring_actors) and ring_actors[i] is not None:
                    pos_idx = int(positions[i])
                    if pos_idx < len(centerline) - 1:
                        self._update_ring_transform(ring_actors[i], centerline, pos_idx)

        # Render updated scene
        self.plotter.render()