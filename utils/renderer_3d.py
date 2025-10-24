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
    def __init__(self, nifti_files, parent=None):
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
        controls_container.setStyleSheet("background-color: #1a1a1a;")  # Dark background
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("Anatomical Systems")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        controls_layout.addWidget(title_label)

        # Scroll area for systems
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
            }
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        # Create controls for each system
        for system_name in self.systems.keys():
            system_group = self.create_system_control(system_name)
            scroll_layout.addWidget(system_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        controls_layout.addWidget(scroll)

        # Reset all button
        reset_btn = QPushButton("Reset View")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a5568;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6578;
            }
        """)
        reset_btn.clicked.connect(self.reset_camera)
        controls_layout.addWidget(reset_btn)

        return controls_container

    def create_system_control(self, system_name):
        """Create checkbox and opacity slider for a system"""
        group = QGroupBox(system_name)
        group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid #4a5568;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout()

        # Visibility checkbox
        checkbox = QCheckBox("Visible")
        checkbox.setChecked(True)
        checkbox.setStyleSheet("color: white;")
        checkbox.stateChanged.connect(lambda state: self.toggle_system(system_name, state == Qt.Checked))
        self.system_checkboxes[system_name] = checkbox
        layout.addWidget(checkbox)

        # Opacity slider
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        opacity_label.setStyleSheet("color: white;")
        opacity_layout.addWidget(opacity_label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(100)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a5568;
                height: 8px;
                background: #1a1a1a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4299e1;
                border: 1px solid #3182ce;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        slider.valueChanged.connect(lambda value: self.set_system_opacity(system_name, value / 100.0))
        self.system_sliders[system_name] = slider
        opacity_layout.addWidget(slider)

        opacity_value_label = QLabel("100%")
        opacity_value_label.setStyleSheet("color: white; min-width: 40px;")
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

        # Initially load all visible systems
        # This will trigger the on-demand loading for default-visible systems
        print("Initializing 3D viewer... Loading default visible systems.")
        for system_name, is_visible in self.system_visible.items():
            if is_visible:
                self.toggle_system(system_name, True)