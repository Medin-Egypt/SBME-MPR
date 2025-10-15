import tensorflow as tf
import os
import sys
import nibabel as nib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QSizePolicy, QButtonGroup, QFileDialog, QMessageBox,
    QDialog, QFormLayout, QSpinBox, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSize, QEvent, QTimer, QPoint
from PyQt5.QtGui import QPixmap, QIcon, QImage, QColor
import utils.loader as loader
import utils.detect_orientation as od
from utils.SliceViewLabel import  SliceViewLabel

class SliceCropDialog(QDialog):
    """A dialog to get a range of slices from the user."""
    def __init__(self, max_slice_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Slices")
        
        # Remove default title bar
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        
        # Main container
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Custom title bar
        title_bar = QWidget()
        title_bar.setObjectName("dialog_title_bar")
        title_bar.setFixedHeight(35)
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_bar_layout.setSpacing(0)
        
        # Title label
        title_label = QLabel("Crop Slices")
        title_label.setObjectName("dialog_title_label")
        title_bar_layout.addWidget(title_label)
        title_bar_layout.addStretch()
        
        # Close button
        close_btn = QPushButton()
        close_btn.setObjectName("dialog_close_btn")
        close_btn.setIcon(QIcon("Icons/cross.png"))
        close_btn.setIconSize(QSize(16, 16))
        close_btn.setFixedSize(40, 35)
        close_btn.clicked.connect(self.reject)
        title_bar_layout.addWidget(close_btn)
        
        # Enable dragging
        self.drag_position = None
        title_bar.mousePressEvent = self.title_bar_mouse_press
        title_bar.mouseMoveEvent = self.title_bar_mouse_move
        
        main_layout.addWidget(title_bar)
        
        # Content area
        content_widget = QWidget()
        content_widget.setObjectName("dialog_content")
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        form_layout = QFormLayout()

        # Information label
        info_label = QLabel(f"This file has {max_slice_count} slices.")
        info_label.setObjectName("dialog_info_label")
        layout.addWidget(info_label)

        # Spinboxes for start and end slices
        self.start_slice = QSpinBox()
        self.start_slice.setRange(1, max_slice_count)
        self.start_slice.setValue(1)

        self.end_slice = QSpinBox()
        self.end_slice.setRange(1, max_slice_count)
        self.end_slice.setValue(max_slice_count)

        form_layout.addRow("Crop slices from:", self.start_slice)
        form_layout.addRow("to:", self.end_slice)

        layout.addLayout(form_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)
        main_layout.addWidget(content_widget)
        
        self.setLayout(main_layout)
        self.setMinimumWidth(400)

    def title_bar_mouse_press(self, event):
        """Handle mouse press on title bar for dragging"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def title_bar_mouse_move(self, event):
        """Handle mouse move on title bar for dragging"""
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def get_values(self):
        """Returns the selected start and end slices."""
        return self.start_slice.value(), self.end_slice.value()

class MPRViewer(QMainWindow):
    def __init__(self, file_path=None):
        super().__init__()
        self.metadata = None
        self.setWindowTitle("MPR VIEWER")
        self.setGeometry(100, 100, 1200, 800)

        self.setMinimumSize(800, 600)
        self.setMaximumSize(16777215, 16777215)

        # Remove default title bar
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.view_colors = {
            'axial': QColor(100, 220, 100),  # Green
            'coronal': QColor(100, 150, 255),  # Blue
            'sagittal': QColor(255, 100, 100),  # Red
            'oblique': QColor(255, 255, 100),  # Yellow
        }

        self.data = None
        self.affine = None
        self.dims = None

        self.pixel_dims = {'axial': (0, 0), 'coronal': (0, 0), 'sagittal': (0, 0)}

        self.intensity_min = 0
        self.intensity_max = 255
        self.file_loaded = False

        # Store original contrast values
        self.original_intensity_min = 0
        self.original_intensity_max = 255

        # Crop bounds (normalized 0-1 coordinates)
        self.crop_bounds = None
        self.original_data = None
        self.segmentation_files = []  # List of loaded segmentation file paths
        self.segmentation_data_list = []  # List of numpy arrays for each segmentation
        self.original_segmentation_data_list = []
        self.segmentation_visible = False  # Whether to show segmentation overlays

        self.norm_coords = {'S': 0.5, 'C': 0.5, 'A': 0.5}

        self.slices = {
            'axial': 0, 'coronal': 0, 'sagittal': 0, 'oblique': 0
        }

        self.rot_x_deg = 0
        self.rot_y_deg = 0

        self.view_labels = {}
        self.view_panels = {}
        self.maximized_view = None

        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False

        # Track window dragging
        self.drag_position = None
        self.is_maximized = False

        # NEW properties for coordinated scaling/zooming
        self.global_zoom_factor = 1.0
        self.default_scale_factor = 1.0
        # Oblique axis properties
        self.oblique_axis_visible = False
        self.oblique_axis_angle = 0  # Default angle in degrees
        self.oblique_axis_dragging = False
        self.oblique_axis_handle_size = 10  # Size of draggable handle
        # Create main container widget
        container = QWidget()
        self.setCentralWidget(container)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Create custom title bar
        title_bar = self.create_title_bar()
        container_layout.addWidget(title_bar)

        # Create content widget
        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        sidebar = self.create_sidebar()
        self.viewing_area_widget = self.create_viewing_area()

        main_layout.addWidget(sidebar)
        # Fix: Duplicate addition of sidebar and viewing_area_widget removed
        main_layout.addWidget(self.viewing_area_widget)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        container_layout.addWidget(content_widget)

        self.add_image_to_button("mode_btn_0", "Icons/windows.png", "3 Main Views")
        self.add_image_to_button("mode_btn_1", "Icons/heart.png", "Segmentation View")
        self.add_image_to_button("mode_btn_2", "Icons/diagram.png", "Oblique View")
        self.add_image_to_button("tool_btn_0_0", "Icons/tab.png", "Slide/Crosshair Mode")
        self.add_image_to_button("tool_btn_0_1", "Icons/brightness.png", "Contrast Mode")
        self.add_image_to_button("tool_btn_0_2", "Icons/loupe.png", "Zoom/Pan Mode")
        self.add_image_to_button("tool_btn_1_0", "Icons/expand.png", "Crop Mode")
        self.add_image_to_button("tool_btn_1_1", "Icons/rotating-arrow-to-the-right.png", "Rotate Mode")
        self.add_image_to_button("tool_btn_1_2", "Icons/video.png", "Cine Mode")
        self.add_image_to_button("export_btn_0", "Icons/NII.png", "NIFTI Export")
        self.add_image_to_button("export_btn_1", "Icons/DIC.png", "DICOM Export")

        main_views_btn = self.findChild(QPushButton, "mode_btn_0")
        if main_views_btn:
            main_views_btn.setChecked(True)

        default_tool = self.findChild(QPushButton, "tool_btn_0_0")
        if default_tool:
            default_tool.setChecked(True)

        content_widget.installEventFilter(self)

        cine_btn = self.findChild(QPushButton, "tool_btn_1_2")
        if cine_btn:
            cine_btn.clicked.connect(self.handle_cine_button_toggle)

        self.show_main_views_initially()

    def _calculate_pixel_dims(self):
        """
        Calculates the aspect-ratio-corrected pixel dimensions for axial,
        coronal, and sagittal views based on voxel spacing.
        This should be called once after a file is loaded.
        """
        if self.dims is None or self.affine is None:
            self.pixel_dims = {'axial': (0, 0), 'coronal': (0, 0), 'sagittal': (0, 0)}
            return

        # Voxel spacing from the affine matrix diagonal
        # Assuming affine[0,0]=x, affine[1,1]=y, affine[2,2]=z spacing
        x_spacing = self.affine[0, 0]
        y_spacing = self.affine[1, 1]
        z_spacing = self.affine[2, 2]

        # Raw voxel counts from data shape (Sagittal, Coronal, Axial)
        sag_vox, cor_vox, ax_vox = self.dims[0], self.dims[1], self.dims[2]

        # Calculate Axial view dimensions (Sagittal x Coronal plane)
        ax_w = sag_vox
        ax_h = int(cor_vox * (y_spacing / x_spacing)) if x_spacing > 0 else cor_vox
        self.pixel_dims['axial'] = (ax_w, ax_h)

        # Calculate Coronal view dimensions (Sagittal x Axial plane)
        cor_w = sag_vox
        cor_h = int(ax_vox * (z_spacing / x_spacing)) if x_spacing > 0 else ax_vox
        self.pixel_dims['coronal'] = (cor_w, cor_h)

        # Calculate Sagittal view dimensions (Coronal x Axial plane)
        sag_w = cor_vox
        sag_h = int(ax_vox * (z_spacing / y_spacing)) if y_spacing > 0 else ax_vox
        self.pixel_dims['sagittal'] = (sag_w, sag_h)

    def create_title_bar(self):
        """Create custom title bar with window controls"""
        title_bar = QWidget()
        title_bar.setObjectName("custom_title_bar")
        title_bar.setFixedHeight(35)

        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title label
        title_label = QLabel("SBME29 MPR")
        title_label.setObjectName("title_label")
        layout.addWidget(title_label)

        layout.addStretch()

        # Minimize button
        minimize_btn = QPushButton()
        minimize_btn.setObjectName("minimize_btn")
        minimize_btn.setIcon(QIcon("Icons/window-minimize.png"))
        minimize_btn.setIconSize(QSize(16, 16))
        minimize_btn.clicked.connect(self.showMinimized)
        layout.addWidget(minimize_btn)

        # Maximize/Restore button
        self.maximize_btn = QPushButton()
        self.maximize_btn.setObjectName("maximize_btn")
        self.maximize_btn.setIcon(QIcon("Icons/window-maximize.png"))
        self.maximize_btn.setIconSize(QSize(16, 16))
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        layout.addWidget(self.maximize_btn)

        # Close button
        close_btn = QPushButton()
        close_btn.setObjectName("close_btn")
        close_btn.setIcon(QIcon("Icons/cross.png"))
        close_btn.setIconSize(QSize(16, 16))
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # Enable dragging on title bar
        title_bar.mousePressEvent = self.title_bar_mouse_press
        title_bar.mouseMoveEvent = self.title_bar_mouse_move
        title_bar.mouseDoubleClickEvent = self.title_bar_double_click

        return title_bar

    def title_bar_mouse_press(self, event):
        """Handle mouse press on title bar for dragging"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def title_bar_mouse_move(self, event):
        """Handle mouse move on title bar for dragging"""
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            if self.is_maximized:
                # Restore window when dragging from maximized state
                self.toggle_maximize()
                # Adjust drag position for restored window size
                self.drag_position = QPoint(self.width() // 2, 10)
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def title_bar_double_click(self, event):
        """Handle double click on title bar to maximize/restore"""
        if event.button() == Qt.LeftButton:
            self.toggle_maximize()

    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.is_maximized:
            self.showNormal()
            self.is_maximized = False
            self.maximize_btn.setIcon(QIcon("Icons/window-maximize.png"))
        else:
            self.showMaximized()
            self.is_maximized = True
            self.maximize_btn.setIcon(QIcon("Icons/browsers.png"))




    def show_slice_crop_dialog(self):
        """Shows a dialog to get a slice range and applies the crop."""
        if not self.file_loaded or self.original_data is None:
            QMessageBox.warning(self, "No File", "Please load a file first.")
            return

        # Use original_data to always know the full slice range
        total_slices = self.original_data.shape[2]
        dialog = SliceCropDialog(total_slices, self)

        if dialog.exec_() == QDialog.Accepted:
            start, end = dialog.get_values()

            if start >= end:
                QMessageBox.critical(self, "Input Error", "The 'from' slice must be smaller than the 'to' slice.")
                return

            # Convert 1-based UI values to 0-based numpy indices
            start_idx = start - 1
            end_idx = end - 1

            self.apply_slice_crop(start_idx, end_idx)

    def apply_slice_crop(self, start_idx, end_idx):
        """Crops the data volume to the specified slice range (axial)."""
        if self.original_data is None:
            return

        # Crop the original data along the axial (3rd) axis
        self.data = self.original_data[:, :, start_idx : end_idx + 1].copy()
        self.dims = self.data.shape

        # ADD THIS: Crop all segmentation data as well
        if self.segmentation_data_list:
            for i in range(len(self.segmentation_data_list)):
                self.segmentation_data_list[i] = self.segmentation_data_list[i][:, :, start_idx : end_idx + 1].copy()

        # Reset views to the new, smaller volume
        self.reset_crosshair_and_slices()
        self.update_all_views()

        QMessageBox.information(self, "Crop Applied",
                                f"Volume cropped to show slices {start_idx + 1} to {end_idx + 1}.\n"
                                f"New dimensions: {self.dims}")

    # --- Coordinated Zoom Logic ---

    def change_global_zoom(self, delta):
        """Updates the global zoom factor and applies it to all views."""
        if not self.file_loaded:
            return

        zoom_step = 1.15

        if delta > 0:
            new_zoom = self.global_zoom_factor * zoom_step
            self.global_zoom_factor = min(new_zoom, 10.0)
        else:
            new_zoom = self.global_zoom_factor / zoom_step
            self.global_zoom_factor = max(new_zoom, 1.0)

        # Apply the new zoom to all views
        for label in self.view_labels.values():
            if isinstance(label, SliceViewLabel):
                label.zoom_factor = self.global_zoom_factor
                if self.global_zoom_factor == 1.0:
                    label.pan_offset_x = 0
                    label.pan_offset_y = 0
                label._apply_zoom_and_pan()

    # --- Reset logic methods ---

    def on_reset_clicked(self):
        """Handler for the Reset button. Resets the currently active tool."""
        if not self.file_loaded:
            return

        checked_btn = self.tools_group_buttons.checkedButton()
        if not checked_btn:
            return

        btn_name = checked_btn.objectName()

        if btn_name == "tool_btn_0_0":
            self.reset_crosshair_and_slices()
        elif btn_name == "tool_btn_0_1":
            self.reset_contrast()
        elif btn_name == "tool_btn_0_2":
            self.reset_all_zooms()
        elif btn_name == "tool_btn_1_0":
            self.reset_crop()
        elif btn_name == "tool_btn_1_1":
            self.reset_rotation()

        self.update_all_views()

    def reset_all_zooms(self):
        """Resets the global zoom factor and all view-specific pan offsets."""
        self.global_zoom_factor = 1.0
        for label in self.view_labels.values():
            if isinstance(label, SliceViewLabel):
                label.zoom_factor = 1.0
                label.pan_offset_x = 0
                label.pan_offset_y = 0
        self.update_all_views()  # Trigger full update to reapply correct base scale

    def reset_contrast(self):
        """Resets the window/level to the initial values from file load."""
        self.intensity_min = self.original_intensity_min
        self.intensity_max = self.original_intensity_max

    def reset_rotation(self):
        """Resets the oblique rotation angles to default."""
        self.rot_x_deg = 0
        self.rot_y_deg = 0
        self.oblique_axis_angle = 0  # Reset to default angle

    def reset_crosshair_and_slices(self):
        """Resets crosshairs to the center and slices to the middle."""
        self.norm_coords = {'S': 0.5, 'C': 0.5, 'A': 0.5}
        if self.dims:
            self.slices['axial'] = self.dims[2] // 2
            self.slices['coronal'] = self.dims[1] // 2
            self.slices['sagittal'] = self.dims[0] // 2
            self.slices['oblique'] = self.dims[2] // 2

    def reset_crop(self):
        """Resets the crop to show the full volume."""
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self.dims = self.data.shape
            self.crop_bounds = None
            
            # ADD THIS: Reset segmentation data to original as well
            if hasattr(self, 'original_segmentation_data_list') and self.original_segmentation_data_list:
                self.segmentation_data_list = [seg.copy() for seg in self.original_segmentation_data_list]
            
            self.reset_crosshair_and_slices()

    def set_slice_from_crosshair(self, source_view, norm_x, norm_y):
        """Updates slice indices based on the normalized crosshair position from a source view."""
        if not self.file_loaded or self.dims is None:
            return

        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        if source_view == 'axial':
            self.norm_coords['S'] = norm_x
            self.norm_coords['C'] = norm_y
            self.slices['coronal'] = int(norm_y * (self.dims[1] - 1))
            self.slices['sagittal'] = int(norm_x * (self.dims[0] - 1))
        elif source_view == 'coronal':
            self.norm_coords['S'] = norm_x
            self.norm_coords['A'] = norm_y
            # Note: Axial dimension is typically inverted in normalized space
            self.slices['axial'] = int((1 - norm_y) * (self.dims[2] - 1))
            self.slices['sagittal'] = int(norm_x * (self.dims[0] - 1))
        elif source_view == 'sagittal':
            self.norm_coords['C'] = norm_x
            self.norm_coords['A'] = norm_y
            # Note: Axial dimension is typically inverted in normalized space
            self.slices['axial'] = int((1 - norm_y) * (self.dims[2] - 1))
            self.slices['coronal'] = int(norm_x * (self.dims[1] - 1))


        self.update_all_views()


    def update_oblique_from_crosshair(self):
        """Update the oblique view slice based on the current crosshair position."""
        if not self.file_loaded or self.dims is None:
            return
        
        if not self.oblique_view_enabled:
            return
        
        # Use the axial coordinate to determine the oblique slice
        # This makes the oblique view follow the crosshair position
        self.slices['oblique'] = self.slices['axial']
        
        # Update only the oblique view
        self.update_view('oblique', 'oblique')

    
    # --- NEW METHOD ---
    def set_slice_from_scroll(self, view_type, new_slice_index):
        """
        Updates a slice index from scrolling or cine mode, recalculates the
        corresponding normalized coordinate for the crosshair, and updates all views.
        """
        if not self.file_loaded or self.dims is None:
            return

        self.slices[view_type] = new_slice_index

        # If scrolling in oblique view, only update that view without syncing crosshairs
        if view_type == 'oblique':
            self.update_view('oblique', 'oblique')
            return

        # Update the corresponding normalized coordinate based on which view was scrolled
        if view_type == 'axial':
            # Scrolling through axial slices changes the Axial ('A') coordinate.
            if self.dims[2] > 1:
                # Based on `set_slice_from_crosshair`, the axial dimension is inverted.
                self.norm_coords['A'] = 1.0 - (new_slice_index / (self.dims[2] - 1))
        elif view_type == 'coronal':
            # Scrolling through coronal slices changes the Coronal ('C') coordinate.
            if self.dims[1] > 1:
                self.norm_coords['C'] = new_slice_index / (self.dims[1] - 1)
        elif view_type == 'sagittal':
            # Scrolling through sagittal slices changes the Sagittal ('S') coordinate.
            if self.dims[0] > 1:
                self.norm_coords['S'] = new_slice_index / (self.dims[0] - 1)

        # Update all views to reflect the new slice and the new crosshair positions
        self.update_all_views()

    def calculate_and_set_uniform_default_scale(self):
        """CHANGE 1: Calculates the minimum non-distorting scale factor across all visible views."""
        if not self.file_loaded or not self.dims:
            self.default_scale_factor = 1.0
            return

        min_scale = float('inf')

        # Get list of currently visible main views
        views_to_check = []
        if self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled:
            views_to_check.extend(['frontal', 'sagittal', 'axial'])
        if self.oblique_view_enabled:
            views_to_check.append('oblique')

        for view_name in views_to_check:
            label = self.view_labels.get(view_name)
            if not label or not label.isVisible() or label.width() < 10 or label.height() < 10:
                continue

            view_type = 'coronal' if view_name == 'frontal' else view_name

            # --- SIMPLIFIED LOGIC ---
            # Get pre-calculated, aspect-ratio-corrected dimensions
            if view_type in self.pixel_dims:
                img_w, img_h = self.pixel_dims[view_type]
            else:  # Fallback for oblique or other views
                img_w, img_h = self.pixel_dims['axial']

                # Calculate the scaling factor needed for this view
            if img_w > 0 and img_h > 0:
                scale_w = label.width() / img_w
                scale_h = label.height() / img_h
                current_scale = min(scale_w, scale_h)
                min_scale = min(min_scale, current_scale)

        self.default_scale_factor = min_scale

    def update_all_views(self):
        views_to_update = []
        if self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled:
            views_to_update.extend([
                ('frontal', 'coronal'), ('sagittal', 'sagittal'), ('axial', 'axial')
            ])
        if self.oblique_view_enabled:
            views_to_update.append(('oblique', 'oblique'))
        if self.segmentation_view_enabled:
            views_to_update.append(('segmentation', 'segmentation'))

        # Calculate uniform base scale before updating views
        self.calculate_and_set_uniform_default_scale()

        for ui_title, view_type in views_to_update:
            self.update_view(ui_title, view_type, sync_crosshair=True)

    def open_nifti_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "",
                                                   "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if file_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max, self.metadata = loader.load_nifti_data(
                    file_path)
                self.file_loaded = True

                self._calculate_pixel_dims()

                # Store original contrast values on load
                self.original_intensity_min = self.intensity_min
                self.original_intensity_max = self.intensity_max

                # Store original data for crop reset
                self.original_data = self.data.copy()
                self.crop_bounds = None

                self.reset_crosshair_and_slices()
                self.reset_all_zooms()  # Resets global zoom factor
                self.reset_rotation()

                self.show_main_views_initially()
                self.update_all_views()  # Triggers uniform default scale calculation
                QMessageBox.information(self, "Success", f"NIfTI file loaded successfully!\nDimensions: {self.dims}")
            except Exception as e:

                QMessageBox.critical(self, "Error", f"Failed to load NIfTI file:\n{str(e)}")

    def open_dicom_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", "", QFileDialog.ShowDirsOnly)
        if folder_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max, self.metadata = loader.load_dicom_data(
                    folder_path)
                self.file_loaded = True

                self._calculate_pixel_dims()

                # Store original contrast values on load
                self.original_intensity_min = self.intensity_min
                self.original_intensity_max = self.intensity_max

                # Store original data for crop reset
                self.original_data = self.data.copy()
                self.crop_bounds = None

                self.reset_crosshair_and_slices()
                self.reset_all_zooms()  # Resets global zoom factor
                self.reset_rotation()

                middle_slice_data = self.data[:, :, self.slices['axial']]
                orientation, confidence = od.predict_dicom_image(middle_slice_data)
                orientation_info = f"\n\nDetected Orientation: {orientation} ({confidence:.2f}% confidence)"

                self.show_main_views_initially()
                self.update_all_views()  # Triggers uniform default scale calculation

                QMessageBox.information(
                    self, "Success",
                    f"DICOM folder loaded successfully!\nDimensions: {self.dims}{orientation_info}\n\n"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM folder:\n{str(e)}")
                import traceback
                print(traceback.print_exc())
    
    def load_segmentation_files(self):
        """Opens a file dialog to select multiple NIfTI segmentation files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Segmentation Files", 
            "",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        if not self.file_loaded:
            QMessageBox.warning(self, "No Data", "Please load a main file first.")
            return
        
        # Clear existing segmentations
        self.segmentation_files = []
        self.segmentation_data_list = []
        self.original_segmentation_data_list = []  # Store original segmentation data for crop reset

        # Load each segmentation file
        for file_path in file_paths:
            try:
                nifti_file = nib.load(file_path)
                seg_data = nifti_file.get_fdata()
                
                # Apply the same flip as the main data
                seg_data = seg_data[::-1, :, :]
                
                # Check if dimensions match
                if seg_data.shape != self.data.shape:
                    QMessageBox.warning(
                        self, 
                        "Dimension Mismatch", 
                        f"Segmentation file {os.path.basename(file_path)} has different dimensions.\n"
                        f"Expected: {self.data.shape}, Got: {seg_data.shape}"
                    )
                    continue
                
                self.segmentation_files.append(file_path)
                self.segmentation_data_list.append(seg_data)
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Failed to load segmentation file {os.path.basename(file_path)}:\n{str(e)}"
                )
        
        if self.segmentation_data_list:
            self.segmentation_visible = True
            self.update_all_views()
            QMessageBox.information(
                self, 
                "Success", 
                f"Loaded {len(self.segmentation_data_list)} segmentation file(s)."
            )

    def delete_segmentation_files(self):
        """Deletes all loaded segmentation files."""
        if not self.segmentation_data_list:
            QMessageBox.information(self, "No Segmentation", "No segmentation files are currently loaded.")
            return
        
        reply = QMessageBox.question(
            self,
            "Delete Segmentation",
            f"Are you sure you want to delete all {len(self.segmentation_data_list)} segmentation file(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear all segmentation data
            self.segmentation_files = []
            self.segmentation_data_list = []
            self.original_segmentation_data_list = []
            self.segmentation_visible = False
            
            # Update all views to remove overlays
            self.update_all_views()
            
            QMessageBox.information(
                self,
                "Success",
                "All segmentation files have been deleted."
            )

    def handle_cine_button_toggle(self, checked):
        if not checked:
            for label in self.view_labels.values():
                if isinstance(label, SliceViewLabel):
                    label.stop_cine()

    def handle_rotate_mode_toggle(self, checked):
        """Show/hide oblique axis based on rotate mode and current view"""
        if self.oblique_view_enabled:
            # Just update the view to show/hide the axis based on rotate mode
            self.update_view('frontal', 'coronal', sync_crosshair=True)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            if self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled:
                if not hasattr(self, '_resize_timer'):
                    self._resize_timer = QTimer()
                    self._resize_timer.setSingleShot(True)
                    self._resize_timer.timeout.connect(self.update_visible_views)
                self._resize_timer.stop()
                self._resize_timer.start(50)
        return super().eventFilter(obj, event)

    def numpy_to_qpixmap(self, array_2d: np.ndarray) -> QPixmap:
        if array_2d.dtype != np.uint8:
            array_2d = array_2d.astype(np.uint8)
        h, w = array_2d.shape
        q_img = QImage(array_2d.tobytes(), w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def add_segmentation_overlay(self, base_pixmap, view_type):
        """Adds red outline overlay from segmentation data to the pixmap."""
        from PyQt5.QtGui import QPainter, QPen
        
        # Convert pixmap to QImage for painting
        image = base_pixmap.toImage()
        painter = QPainter(image)
        pen = QPen(QColor(255, 0, 0), 2)  # Red color, 2px width
        painter.setPen(pen)
        
        # Get the current slice for this view
        if view_type == 'axial':
            slice_idx = self.slices['axial']
        elif view_type == 'coronal':
            slice_idx = self.slices['coronal']
        elif view_type == 'sagittal':
            slice_idx = self.slices['sagittal']
        elif view_type == 'oblique':
            # For oblique, we'll skip segmentation overlay for now
            painter.end()
            return QPixmap.fromImage(image)
        else:
            painter.end()
            return base_pixmap
        
        # Process each loaded segmentation
        for seg_data in self.segmentation_data_list:
            # Extract the slice from segmentation data
            if view_type == 'axial':
                seg_slice = seg_data[:, :, slice_idx]
                seg_slice = np.flipud(np.rot90(seg_slice))
            elif view_type == 'coronal':
                seg_slice = seg_data[:, slice_idx, :]
                seg_slice = np.rot90(seg_slice)
            elif view_type == 'sagittal':
                seg_slice = seg_data[slice_idx, :, :]
                seg_slice = np.rot90(seg_slice)
            
            # Find edges/contours in the segmentation
            from scipy import ndimage
            
            # Create binary mask
            mask = seg_slice > 0.5
            
            if not mask.any():
                continue
            
            # Find edges using morphological operations
            eroded = ndimage.binary_erosion(mask)
            edges = mask & ~eroded
            
            # Scale factor to match pixmap size
            scale_y = base_pixmap.height() / edges.shape[0]
            scale_x = base_pixmap.width() / edges.shape[1]
            
            # Draw the edges
            edge_coords = np.argwhere(edges)
            for y, x in edge_coords:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                painter.drawPoint(scaled_x, scaled_y)
        
        painter.end()
        return QPixmap.fromImage(image)

    def update_view(self, ui_title: str, view_type: str, sync_crosshair=False):
        if ui_title not in self.view_labels:
            return
        label = self.view_labels[ui_title]
        # Only check visibility, let the label's apply_zoom handle the rest of the scaling
        if not label.isVisible():
            return
        if not self.file_loaded or self.data is None:
            return

        if view_type == 'segmentation':
            self.update_segmentation_view()
            return

        slice_data = loader.get_slice_data(
            self.data, self.dims, self.slices, self.affine,
            self.intensity_min, self.intensity_max,
            rot_x_deg=self.rot_x_deg, rot_y_deg=self.rot_y_deg,
            view_type=view_type,
            norm_coords=self.norm_coords  # Pass normalized coordinates
        )

        # Determine the target size for the unscaled image pixmap
        # This size is based on the original data shape to allow for correct aspect ratio calculation.
        # DIMS are (Sagittal, Coronal, Axial)
        if view_type == 'axial' or view_type == 'oblique':
            pixmap_w, pixmap_h = self.dims[0], self.dims[1]  # Sagittal (x), Coronal (y)
        elif view_type == 'coronal':
            pixmap_w, pixmap_h = self.dims[0], self.dims[2]  # Sagittal (x), Axial (y)
        elif view_type == 'sagittal':
            pixmap_w, pixmap_h = self.dims[1], self.dims[2]  # Coronal (x), Axial (y)
        else:
            return  # Should not happen

        # Scale the numpy data to the target size before converting to QPixmap
        # This is a simplification. A proper MPR would handle image size based on voxel size.
        # For now, we will create the pixmap from the raw slice and let SliceViewLabel handle scaling.
        pixmap = self.numpy_to_qpixmap(slice_data)

        if self.segmentation_visible and self.segmentation_data_list and view_type != 'segmentation':
            pixmap = self.add_segmentation_overlay(pixmap, view_type)

        if isinstance(label, SliceViewLabel):
            # Ensure the label's zoom factor is synchronized
            label.zoom_factor = self.global_zoom_factor
            label.set_image_pixmap(pixmap)

            if sync_crosshair:
                if view_type == 'axial':
                    norm_x, norm_y = self.norm_coords['S'], self.norm_coords['C']
                elif view_type == 'coronal':
                    norm_x, norm_y = self.norm_coords['S'], self.norm_coords['A']
                elif view_type == 'sagittal':
                    norm_x, norm_y = self.norm_coords['C'], self.norm_coords['A']
                else:
                    norm_x, norm_y = 0.5, 0.5

                label.set_normalized_crosshair(norm_x, norm_y)
            
            if view_type == 'coronal' and self.oblique_view_enabled:
                label.oblique_axis_angle = self.oblique_axis_angle
                label.oblique_axis_visible = self.oblique_axis_visible
            else:
                label.oblique_axis_visible = False
        else:
            scaled = pixmap.scaled(
                QSize(label.size().width() - 2, label.size().height() - 2),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled)

    def draw_oblique_axis(self, label, view_type):
        """Draw the oblique axis on the frontal view"""
        if view_type != 'coronal' or not self.oblique_axis_visible:
            return
        
        if not isinstance(label, SliceViewLabel):
            return
        
        label.oblique_axis_angle = self.oblique_axis_angle
        label.oblique_axis_visible = True
        label.update()

    def update_segmentation_view(self):
        if 'segmentation' in self.view_labels:
            label = self.view_labels['segmentation']
            label.setText("Segmentation View\n\n[Add your segmentation data here]")

    def update_visible_views(self):
        # When resizing, recalculate the uniform scale and then update all visible views
        self.calculate_and_set_uniform_default_scale()

        visible_views = [name for name, panel in self.view_panels.items() if panel.isVisible()]
        for view_name in visible_views:
            view_type = view_name
            if view_name == 'frontal':
                view_type = 'coronal'
            self.update_view(view_name, view_type, sync_crosshair=True)

    def maximize_view(self, view_name):
        if not (self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled):
            return

        self.maximized_view = view_name

        for name, panel in self.view_panels.items():
            if name != view_name:
                panel.hide()
            else:
                self.viewing_grid.removeWidget(panel)
                self.viewing_grid.addWidget(panel, 0, 0, 2, 2)
                panel.show()

        self.viewing_grid.invalidate()
        QApplication.processEvents()
        self.update_visible_views()

    def restore_views(self):
        if self.maximized_view is None:
            return

        max_panel = self.view_panels[self.maximized_view]
        self.viewing_grid.removeWidget(max_panel)
        self.maximized_view = None

        # Panels list to restore the grid layout structure
        panels = [
            ("Frontal", 'coronal', 0, 0), ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0), ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1)
        ]

        for title, view_type, row, col in panels:
            panel = self.view_panels[title.lower()]
            # Ensure the widget is not already in the layout before adding it
            if self.viewing_grid.indexOf(panel) == -1:
                self.viewing_grid.addWidget(panel, row, col)

        # Restore visibility based on current mode
        if self.main_views_enabled:
            self.toggle_main_views(True)
        elif self.oblique_view_enabled:
            self.toggle_oblique_view(True)
        elif self.segmentation_view_enabled:
            self.toggle_segmentation_view(True)

        self.viewing_grid.invalidate()
        QApplication.processEvents()
        self.update_visible_views()

    def show_main_views_initially(self):
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False

        # Ensure correct widgets are added to the grid if they were removed
        self.restore_views()

        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial']:
                panel.show()
                # Update is handled by update_visible_views in the end
            else:
                panel.hide()

    def toggle_main_views(self, checked):
        if not checked:
            return

        self.restore_views()
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False
        self.segmentation_visible = True if self.segmentation_data_list else False

        self.findChild(QPushButton, "mode_btn_2").setChecked(False)
        self.findChild(QPushButton, "mode_btn_1").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial']:
                panel.show()
            else:
                panel.hide()

        self.update_visible_views()

    def toggle_oblique_view(self, checked):
        if not checked and self.oblique_view_enabled:
            self.toggle_main_views(True)
            self.findChild(QPushButton, "mode_btn_0").setChecked(True)
            return

        self.restore_views()
        self.oblique_view_enabled = True
        self.main_views_enabled = False
        self.segmentation_view_enabled = False
        self.segmentation_visible = True if self.segmentation_data_list else False

        # Always show oblique axis in oblique view mode
        self.oblique_axis_visible = True

        self.findChild(QPushButton, "mode_btn_0").setChecked(False)
        self.findChild(QPushButton, "mode_btn_1").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial', 'oblique']:
                panel.show()
                if view_name == 'oblique':
                    # Ensure oblique is placed in 1,1
                    self.viewing_grid.removeWidget(panel)
                    self.viewing_grid.addWidget(panel, 1, 1)
            else:
                panel.hide()

        self.update_visible_views()

    def toggle_segmentation_view(self, checked):
        if not checked and self.segmentation_view_enabled:
            self.toggle_main_views(True)
            self.findChild(QPushButton, "mode_btn_0").setChecked(True)
            return

        self.restore_views()
        self.segmentation_view_enabled = True
        self.main_views_enabled = False
        self.oblique_view_enabled = False
        self.segmentation_visible = False  # Hide overlays in segmentation view mode

        self.findChild(QPushButton, "mode_btn_0").setChecked(False)
        self.findChild(QPushButton, "mode_btn_2").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial', 'segmentation']:
                panel.show()
                if view_name == 'segmentation':
                    # Ensure segmentation is placed in 1,1
                    self.viewing_grid.removeWidget(panel)
                    self.viewing_grid.addWidget(panel, 1, 1)
            else:
                panel.hide()

        self.update_visible_views()

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFrameStyle(QFrame.Box)
        sidebar.setFixedWidth(200)

        layout = QVBoxLayout(sidebar)  # This line MUST be here at the beginning
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)

        file_group = QGroupBox("Load File:")
        file_layout = QVBoxLayout()

        open_nifti_btn = QPushButton("Open NIfTI File")
        open_nifti_btn.setObjectName("open_btn_nifti")
        open_nifti_btn.setMinimumHeight(35)
        open_nifti_btn.clicked.connect(self.open_nifti_file)
        file_layout.addWidget(open_nifti_btn)

        open_dicom_btn = QPushButton("Open DICOM Folder")
        open_dicom_btn.setObjectName("open_btn_dicom")
        open_dicom_btn.setMinimumHeight(35)
        open_dicom_btn.clicked.connect(self.open_dicom_folder)
        file_layout.addWidget(open_dicom_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        mode_group = QGroupBox("Mode:")
        mode_layout = QVBoxLayout()
        mode_buttons_widget = QWidget()
        mode_buttons_layout = QGridLayout(mode_buttons_widget)
        self.mode_group_buttons = QButtonGroup(self)
        self.mode_group_buttons.setExclusive(False)
        for i in range(3):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"mode_btn_{i}")
            btn.setCheckable(True)
            mode_buttons_layout.addWidget(btn, 0, i)
            self.mode_group_buttons.addButton(btn, i)
            if i == 0:
                btn.clicked.connect(self.toggle_main_views)
            elif i == 1:
                btn.clicked.connect(self.toggle_segmentation_view)
            elif i == 2:
                btn.clicked.connect(self.toggle_oblique_view)

        mode_layout.addWidget(mode_buttons_widget)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Add Segmentation section
        seg_group = QGroupBox("Segmentation:")
        seg_layout = QVBoxLayout()

        load_seg_btn = QPushButton("Load Segmentation")
        load_seg_btn.setObjectName("load_seg_btn")
        load_seg_btn.setMinimumHeight(35)
        load_seg_btn.clicked.connect(self.load_segmentation_files)
        seg_layout.addWidget(load_seg_btn)

        delete_seg_btn = QPushButton("Delete Segmentation")
        delete_seg_btn.setObjectName("delete_seg_btn")
        delete_seg_btn.setMinimumHeight(35)
        delete_seg_btn.clicked.connect(self.delete_segmentation_files)
        seg_layout.addWidget(delete_seg_btn)

        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)

        tools_group = QGroupBox("Tools:")
        tools_main_layout = QVBoxLayout()
        tools_grid_widget = QWidget()
        tools_layout = QGridLayout(tools_grid_widget)

        self.tools_group_buttons = QButtonGroup(self)
        self.tools_group_buttons.setExclusive(True)
        for r in range(2):
            for c in range(3):
                btn = QPushButton()
                btn.setFixedSize(40, 40)
                object_name = f"tool_btn_{r}_{c}"
                btn.setObjectName(object_name)
                tools_layout.addWidget(btn, r, c)

                # If it's the crop button, connect its new function
                if object_name == "tool_btn_1_0":
                    btn.setCheckable(False)  # Make it a normal button, not a mode
                    btn.clicked.connect(self.show_slice_crop_dialog)
                else:
                    # Other buttons remain checkable modes
                    btn.setCheckable(True)
                    self.tools_group_buttons.addButton(btn, r * 3 + c)

        rotate_btn = self.findChild(QPushButton, "tool_btn_1_1")
        if rotate_btn:
            rotate_btn.clicked.connect(self.handle_rotate_mode_toggle)
        
        tools_main_layout.addWidget(tools_grid_widget)

        # Add the Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("reset_btn")
        reset_btn.setMinimumHeight(35)
        reset_btn.clicked.connect(self.on_reset_clicked)
        tools_main_layout.addWidget(reset_btn)

        tools_group.setLayout(tools_main_layout)
        layout.addWidget(tools_group)

        export_group = QGroupBox("Export:")
        export_layout = QGridLayout()
        self.export_group_buttons = QButtonGroup(self)
        self.export_group_buttons.setExclusive(True)
        for i in range(2):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"export_btn_{i}")
            btn.setCheckable(True)
            export_layout.addWidget(btn, 0, i)
            self.export_group_buttons.addButton(btn, i)
            btn.clicked.connect(lambda checked, b=btn: self.toggle_export_button(b))
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()
        return sidebar

    def toggle_export_button(self, button):
        if button.isChecked():
            button.setChecked(False)

            if not self.file_loaded:
                QMessageBox.warning(self, "No Data", "Please load a file first.")
                return

            if self.crop_bounds is not None:
                reply = QMessageBox.question(
                    self, "Apply Crop",
                    "Do you want to apply the current crop before exporting?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )

                if reply == QMessageBox.Cancel:
                    return
                elif reply == QMessageBox.Yes:
                    self.apply_crop()

            if button.objectName() == "export_btn_1":  # DICOM export
                output_dir = QFileDialog.getExistingDirectory(
                    self, "Select folder to save DICOM file", ""
                )
                if output_dir:
                    success = loader.export_to_dicom(self.data, self.affine, output_dir, self.metadata)
                    if success:
                        QMessageBox.information(self, "Export Successful", f"DICOM file saved to:\n{output_dir}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to save DICOM file.")


            elif button.objectName() == "export_btn_0":  # NIFTI export
                output_file = QFileDialog.getSaveFileName(
                    self, "Save NIfTI file", "", "NIfTI Files (*.nii.gz *.nii)"
                )
                if output_file[0]:
                    success = loader.export_to_nifti(self.data, self.affine, output_file[0], self.metadata)
                    if success:
                        QMessageBox.information(self, "Export Successful",
                                                f"NIfTI file saved to:\n{output_file[0]}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to save NIfTI file.")

    def create_viewing_area(self):
        widget = QWidget()
        self.viewing_grid = QGridLayout(widget)
        self.viewing_grid.setSpacing(10)
        self.viewing_grid.setContentsMargins(0, 0, 0, 0)

        panels = [
            ("Frontal", 'coronal', 0, 0), ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0), ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1)
        ]

        for title, view_type, row, col in panels:
            panel = QFrame()
            panel.setObjectName(f"viewing_panel_{title.lower()}")
            panel.setFrameStyle(QFrame.Box)
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(5, 5, 5, 5)
            panel_layout.setSpacing(5)

            title_bar_widget = QWidget()
            title_bar_layout = QHBoxLayout(title_bar_widget)
            title_bar_layout.setContentsMargins(0, 0, 0, 0)
            title_bar_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            color_key = view_type
            if title.lower() == 'frontal':
                color_key = 'coronal'

            if color_key in self.view_colors:
                color_indicator = QLabel()
                color_indicator.setFixedSize(12, 12)
                color = self.view_colors[color_key]
                color_indicator.setStyleSheet(f"""
                    background-color: {color.name()};
                    border-radius: 6px;
                    border: 1px solid #E2E8F0;
                """)
                title_bar_layout.addWidget(color_indicator)

            title_lbl = QLabel(title)
            title_lbl.setObjectName(f"view_title_{title.lower()}")
            title_bar_layout.addWidget(title_lbl)

            panel_layout.addWidget(title_bar_widget)

            if view_type != 'segmentation':
                data_view_type = 'coronal' if title.lower() == 'frontal' else view_type
                view_area = SliceViewLabel(self, data_view_type, title)
            else:
                view_area = QLabel()
                view_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                view_area.setScaledContents(False)
                view_area.setObjectName(f"view_{title.lower()}")
                view_area.setAlignment(Qt.AlignCenter)
                view_area.setText("")

            panel_layout.addWidget(view_area, stretch=1)

            self.view_panels[title.lower()] = panel
            self.view_labels[title.lower()] = view_area
            self.viewing_grid.addWidget(panel, row, col)

        self.viewing_grid.setRowStretch(0, 1)
        self.viewing_grid.setRowStretch(1, 1)
        self.viewing_grid.setColumnStretch(0, 1)
        self.viewing_grid.setColumnStretch(1, 1)

        return widget

    def add_image_to_button(self, name, img, tip=None):
        button = self.findChild(QPushButton, name)
        if button:
            try:
                icon = QIcon(QPixmap(img))
                if not icon.isNull():
                    button.setIcon(icon)
                    button.setIconSize(QSize(32, 32))
            except Exception:
                button.setText(name)
            if tip:
                button.setToolTip(tip)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        with open("style.qss", "r") as f:
            style_sheet = f.read()
            app.setStyleSheet(style_sheet)
    except FileNotFoundError:
        print("Warning: style.qss file not found.")

    viewer = MPRViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()