import pyvista as pv
from pyvistaqt import QtInteractor
import nibabel as nib
import numpy as np
from pathlib import Path
from skimage import measure
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QCheckBox, QGroupBox, QScrollArea, QPushButton
)
from PyQt5.QtCore import Qt


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


def nifti_to_surface(nifti_path, smoothing=True, smoothing_iterations=50):
    """
    Convert NIfTI segmentation mask to surface mesh using marching cubes.

    Parameters:
    -----------
    nifti_path : str or Path
        Path to .nii or .nii.gz file
    smoothing : bool
        Whether to apply Laplacian smoothing
    smoothing_iterations : int
        Number of smoothing iterations

    Returns:
    --------
    mesh : pv.PolyData
        Surface mesh
    affine : np.ndarray
        Affine transformation matrix from NIfTI
    """
    # Load NIfTI file
    nii = nib.load(str(nifti_path))
    data = nii.get_fdata()
    affine = nii.affine

    # Binarize if needed (assumes non-zero values are the segmentation)
    binary_data = (data > 0).astype(np.uint8)

    # Apply marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        binary_data,
        level=0.5,
        spacing=nii.header.get_zooms()[:3]
    )

    # Create PyVista mesh
    # Ensure faces are correctly formatted for PyVista
    # Each face should be [n_points, p0, p1, p2, ...]
    n_points = 3  # Assuming marching cubes produces triangles
    faces_pv = np.empty((faces.shape[0], n_points + 1), dtype=np.int_)
    faces_pv[:, 0] = n_points
    faces_pv[:, 1:] = faces

    mesh = pv.PolyData(verts, faces_pv)

    # Apply affine transformation to align with physical coordinates
    # This transforms voxel coordinates (i, j, k) to physical space (x, y, z)
    # Marching cubes output is already in scaled voxel space (due to spacing)
    # We need to apply the affine transformation
    verts_transformed = nib.affines.apply_affine(affine, verts)
    mesh.points = verts_transformed

    # Optional smoothing
    if smoothing:
        try:
            mesh = mesh.smooth(n_iter=smoothing_iterations, relaxation_factor=0.1)
        except Exception as e:
            print(f"Warning: Smoothing failed. {e}")

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


class SegmentationViewer3D(QWidget):
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

        # Storage for meshes and actors (for lazy loading)
        self.meshes = {}  # {filename_stem: pv.PolyData}
        self.actors = {}  # {filename_stem: actor}

        # System visibility and opacity states
        self.system_visible = {system: True for system in self.systems.keys()}
        self.system_opacity = {system: 1.0 for system in self.systems.keys()}

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
        """
        self.system_visible[system_name] = visible

        # Iterate over all files in this system
        for nifti_file in self.systems.get(system_name, []):
            filename = nifti_file.stem

            if visible:
                # --- Show or Load Actor ---
                if filename in self.actors:
                    # Actor already exists, just show it
                    self.actors[filename].SetVisibility(True)
                else:
                    # Actor doesn't exist, need to load it
                    print(f"Loading {filename}...")
                    try:
                        # Check if mesh is cached
                        if filename in self.meshes:
                            mesh = self.meshes[filename]
                        else:
                            # Generate and cache mesh
                            mesh, _ = nifti_to_surface(nifti_file)
                            self.meshes[filename] = mesh

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

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

            else:
                # --- Hide Actor ---
                if filename in self.actors:
                    self.actors[filename].SetVisibility(False)

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
        """Initialize the 3D view UI. Does not load meshes."""
        # Setup camera and display
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.show()

        # Initially load all visible systems (if any exist)
        # This will trigger the on-demand loading for default-visible systems
        if self.systems:
            print("Initializing 3D viewer... Loading default visible systems.")
            for system_name, is_visible in self.system_visible.items():
                if is_visible:
                    self.toggle_system(system_name, True)
        else:
            print("Initializing 3D viewer... No segmentations to load.")

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