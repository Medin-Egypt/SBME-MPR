import tensorflow as tf
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QSizePolicy, QButtonGroup, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QSize, QEvent, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QImage
import utils.loader as loader
import utils.detect_orientation as od
import time


# --- Custom QLabel for Mouse Wheel Interaction ---
class SliceViewLabel(QLabel):
    """
    Custom QLabel that handles mouse wheel events to scroll through volume slices,
    contrast adjustment, zoom functionality, and cine mode.
    """
    def __init__(self, parent_viewer, view_type, ui_title):
        super().__init__()
        self.parent_viewer = parent_viewer
        self.view_type = view_type      # e.g. 'axial', 'coronal'
        self.ui_title = ui_title        # e.g. 'Axial', 'Frontal'
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 1px solid #666666;
                color: #555555;
                font-size: 18px;
            }
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        self.setObjectName(f"view_{self.ui_title.lower()}")
        self.setText("")

        # State for contrast mode
        self._dragging = False
        self._last_pos = None
        
        # State for zoom mode
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self._panning = False
        self._pan_start = None
        
        # Store the original pixmap for quality preservation
        self._original_pixmap = None
        
        # Prevent rapid zoom events
        self._last_zoom_time = 0
        self._zoom_cooldown = 100  # milliseconds
        
        # Cine mode state
        self.cine_timer = QTimer()
        self.cine_timer.timeout.connect(self._cine_next_slice)
        self.cine_active = False
        self.cine_fps = 10  # Frames per second
        
    def _cine_next_slice(self):
        """Advance to the previous slice in cine mode (going inward/zooming in)."""
        if not self.cine_active:
            return
            
        current_slice = self.parent_viewer.slices[self.view_type]
        
        # Determine max slice based on view type
        if self.view_type in ('axial', 'oblique'):
            max_dim_index = 2
        elif self.view_type == 'coronal':
            max_dim_index = 1
        elif self.view_type == 'sagittal':
            max_dim_index = 0
        else:
            return
        
        max_slice = self.parent_viewer.dims[max_dim_index]
        # Go backwards (subtract 1) to zoom in instead of out
        new_slice = (current_slice - 1) % max_slice
        self.parent_viewer.slices[self.view_type] = new_slice
        self.parent_viewer.update_view(self.ui_title.lower(), self.view_type)
    
    def start_cine(self):
        """Start cine mode playback."""
        if not self.cine_active:
            self.cine_active = True
            self.cine_timer.start(1000 // self.cine_fps)  # Convert FPS to milliseconds
            
    def stop_cine(self):
        """Stop cine mode playback."""
        if self.cine_active:
            self.cine_active = False
            self.cine_timer.stop()
        
    def wheelEvent(self, event):
        slide_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_0")
        zoom_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_2")
        
        if zoom_btn and zoom_btn.isChecked():
            # Zoom mode with cooldown to prevent rapid events
            current_time = time.time() * 1000  # Convert to milliseconds
            if current_time - self._last_zoom_time < self._zoom_cooldown:
                event.accept()
                return
            
            self._last_zoom_time = current_time
            
            delta = event.angleDelta().y()
            
            # Only process if there's a significant delta
            if abs(delta) < 15:
                event.accept()
                return
            
            zoom_step = 1.15  # 15% zoom per scroll
            
            if delta > 0:
                # Zoom in
                new_zoom = self.zoom_factor * zoom_step
                self.zoom_factor = min(new_zoom, 10.0)  # Max 10x zoom
            else:
                # Zoom out
                new_zoom = self.zoom_factor / zoom_step
                self.zoom_factor = max(new_zoom, 1.0)  # Min 1x zoom
                
                # Reset pan when zooming out to 1.0
                if self.zoom_factor == 1.0:
                    self.pan_offset_x = 0
                    self.pan_offset_y = 0
            
            # Re-render with new zoom level for quality preservation
            self.parent_viewer.update_view(self.ui_title.lower(), self.view_type)
            event.accept()
            
        elif slide_btn and slide_btn.isChecked():
            # Slide mode
            delta = event.angleDelta().y()
            step = 1 if abs(delta) > 0 else 0
            direction = step * (-1 if delta > 0 else 1)
            if direction == 0:
                return

            current_slice = self.parent_viewer.slices[self.view_type]
            if self.view_type in ('axial', 'oblique'):
                max_dim_index = 2
            elif self.view_type == 'coronal':
                max_dim_index = 1
            elif self.view_type == 'sagittal':
                max_dim_index = 0
            else:
                return

            max_slice = self.parent_viewer.dims[max_dim_index]
            new_slice = (current_slice + direction) % max_slice
            self.parent_viewer.slices[self.view_type] = new_slice
            self.parent_viewer.update_view(self.ui_title.lower(), self.view_type)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        contrast_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_1")
        zoom_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_2")
        cine_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_2")
        
        if cine_btn and cine_btn.isChecked() and event.button() == Qt.LeftButton:
            # Toggle cine mode playback
            if self.cine_active:
                self.stop_cine()
            else:
                self.start_cine()
        elif zoom_btn and zoom_btn.isChecked() and event.button() == Qt.LeftButton:
            # Pan mode when zoomed
            if self.zoom_factor > 1.0:
                self._panning = True
                self._pan_start = event.pos()
        elif contrast_btn and contrast_btn.isChecked() and event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        zoom_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_2")
        
        if self._panning and self._pan_start and zoom_btn and zoom_btn.isChecked():
            # Handle panning
            dx = event.x() - self._pan_start.x()
            dy = event.y() - self._pan_start.y()
            
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            
            self._pan_start = event.pos()
            
            # Update display with new pan offset
            self._apply_zoom_and_pan()
            
        elif self._dragging and self._last_pos:
            # Handle contrast adjustment
            dx = event.x() - self._last_pos.x()
            dy = event.y() - self._last_pos.y()

            window_change = dx * 2
            level_change = -dy * 2

            window = self.parent_viewer.intensity_max - self.parent_viewer.intensity_min
            level = (self.parent_viewer.intensity_max + self.parent_viewer.intensity_min) / 2

            new_window = max(1, window + window_change)
            new_level = level + level_change

            self.parent_viewer.intensity_min = int(new_level - new_window / 2)
            self.parent_viewer.intensity_max = int(new_level + new_window / 2)

            self._last_pos = event.pos()
            self.parent_viewer.update_view(self.ui_title.lower(), self.view_type)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.LeftButton:
            self._panning = False
        elif self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
        else:
            super().mouseReleaseEvent(event)
    
    def _apply_zoom_and_pan(self):
        """Apply zoom and pan transformation to the stored original pixmap."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        
        label_size = self.size()

        # Ensure we have valid dimensions
        if label_size.width() < 10 or label_size.height() < 10:
            return
        
        # Calculate the size after zoom
        zoomed_width = int(label_size.width() * self.zoom_factor)
        zoomed_height = int(label_size.height() * self.zoom_factor)
        
        # Clamp to reasonable values
        zoomed_width = max(10, min(zoomed_width, 50000))
        zoomed_height = max(10, min(zoomed_height, 50000))
        
        # Scale the original pixmap to zoomed size with high quality
        zoomed_pixmap = self._original_pixmap.scaled(
            QSize(zoomed_width, zoomed_height),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        if self.zoom_factor > 1.0:
            # Calculate maximum pan offsets to keep image within reasonable bounds
            max_offset_x = (zoomed_pixmap.width() - label_size.width()) // 2
            max_offset_y = (zoomed_pixmap.height() - label_size.height()) // 2
            
            # Clamp pan offsets
            self.pan_offset_x = max(-max_offset_x, min(max_offset_x, self.pan_offset_x))
            self.pan_offset_y = max(-max_offset_y, min(max_offset_y, self.pan_offset_y))
            
            # Calculate crop rectangle
            center_x = zoomed_pixmap.width() // 2
            center_y = zoomed_pixmap.height() // 2
            
            crop_x = center_x - label_size.width() // 2 - self.pan_offset_x
            crop_y = center_y - label_size.height() // 2 - self.pan_offset_y
            
            # Ensure crop coordinates are within bounds
            crop_x = max(0, min(crop_x, zoomed_pixmap.width() - label_size.width()))
            crop_y = max(0, min(crop_y, zoomed_pixmap.height() - label_size.height()))
            
            # Crop the zoomed pixmap
            cropped = zoomed_pixmap.copy(
                crop_x, crop_y,
                min(label_size.width(), zoomed_pixmap.width()),
                min(label_size.height(), zoomed_pixmap.height())
            )
            
            self.setPixmap(cropped)
        else:
            # No zoom, display normally
            self.setPixmap(zoomed_pixmap)
    
    def set_image_pixmap(self, pixmap):
        """Store the original high-quality pixmap and apply zoom/pan."""
        self._original_pixmap = pixmap
        self._apply_zoom_and_pan()
    
    def reset_zoom(self):
        """Reset zoom and pan to default values."""
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        if self._original_pixmap:
            self._apply_zoom_and_pan()


class MPRViewer(QMainWindow):
    def __init__(self, file_path=None):
        super().__init__()
        self.setWindowTitle("MPR VIEWER")
        self.setGeometry(100, 100, 1200, 800)
        
        # Prevent minimum size constraints that cause geometry issues
        self.setMinimumSize(800, 600)
        self.setMaximumSize(16777215, 16777215)  # Qt's maximum

        # Initialize data variables to None
        self.data = None
        self.affine = None
        self.dims = None
        self.intensity_min = 0
        self.intensity_max = 255
        self.file_loaded = False

        # Slice indices
        self.slices = {
            'axial': 0,
            'coronal': 0,
            'sagittal': 0,
            'oblique': 0
        }

        self.rot_x_deg = 0
        self.rot_y_deg = 0


        self.view_labels = {}
        self.view_panels = {}

        # View states
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False

        # Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        sidebar = self.create_sidebar()
        viewing_area = self.create_viewing_area()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(viewing_area)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        # Style
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: white;
                border: 2px solid #cccccc;
                border-radius: 3px;
                padding: 2px;
            }
            QPushButton:checked {
                background-color: #a0c8ff;
                border: 2px solid #3366ff;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border: 2px solid #999999;
            }
            QPushButton:pressed { background-color: #d0d0d0; }
        """)

        # Add icons
        self.add_image_to_button("mode_btn_0", "Icons/windows.png", "3 Main Views")
        self.add_image_to_button("mode_btn_1", "Icons/heart.png", "Segmentation View")
        self.add_image_to_button("mode_btn_2", "Icons/diagram.png", "Oblique View")
        self.add_image_to_button("tool_btn_0_0", "Icons/tab.png", "Slide Mode (Scroll Slices)")
        self.add_image_to_button("tool_btn_0_1", "Icons/brightness.png", "Contrast Mode")
        self.add_image_to_button("tool_btn_0_2", "Icons/loupe.png", "Zoom Mode")
        self.add_image_to_button("tool_btn_1_0", "Icons/expand.png", "Crop Mode")
        self.add_image_to_button("tool_btn_1_1", "Icons/rotating-arrow-to-the-right.png", "Rotate Mode")
        self.add_image_to_button("tool_btn_1_2", "Icons/video.png", "Cine Mode (Click view to start/stop)")
        self.add_image_to_button("export_btn_0", "Icons/NII.png", "NIFTI Export")
        self.add_image_to_button("export_btn_1", "Icons/DIC.png", "DICOM Export")

        # Default view
        main_views_btn = self.findChild(QPushButton, "mode_btn_0")
        if main_views_btn:
            main_views_btn.setChecked(True)
        
        # Don't show views initially until file is loaded
        # self.show_main_views_initially()

        # Redraw on resize
        self.centralWidget().installEventFilter(self)
        
        # Connect cine button to stop all cine modes when unchecked
        cine_btn = self.findChild(QPushButton, "tool_btn_1_2")
        if cine_btn:
            cine_btn.clicked.connect(self.handle_cine_button_toggle)
    
    def open_nifti_file(self):
        """Open a NIfTI file dialog and load the selected file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open NIfTI File",
            "",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        
        if file_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max = loader.load_nifti_data(file_path)
                self.file_loaded = True
                
                # Reset slices to middle
                self.slices = {
                    'axial': self.dims[2] // 2,
                    'coronal': self.dims[1] // 2,
                    'sagittal': self.dims[0] // 2,
                    'oblique': self.dims[2] // 2
                }
                
                # Update all visible views
                self.show_main_views_initially()
                QMessageBox.information(self, "Success", f"NIfTI file loaded successfully!\nDimensions: {self.dims}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load NIfTI file:\n{str(e)}")

    def open_dicom_folder(self):
        """Open a folder dialog and load DICOM files from the selected folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select DICOM Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max, organ_data = loader.load_dicom_data(
                    folder_path)
                self.file_loaded = True

                # Reset slices to middle
                self.slices = {
                    'axial': self.dims[2] // 2,
                    'coronal': self.dims[1] // 2,
                    'sagittal': self.dims[0] // 2,
                    'oblique': self.dims[2] // 2
                }

                # --- Integration Point ---
                # 1. Get the middle axial slice data directly from the loaded volume
                middle_slice_data = self.data[:, :, self.slices['axial']]

                # 2. Call the utility function to get the orientation
                orientation, confidence = od.predict_dicom_image(middle_slice_data)

                # 3. Prepare the result string
                orientation_info = f"\n\nDetected Orientation: {orientation} ({confidence:.2f}% confidence)"
                # --- End Integration ---

                # Update all visible views
                self.show_main_views_initially()
                QMessageBox.information(
                    self,
                    "Success",
                    f"DICOM folder loaded successfully!\nDimensions: {self.dims}{orientation_info}\n\n{"\n".join(organ_data)}"
                    # <-- Updated message
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM folder:\n{str(e)}")

    def handle_cine_button_toggle(self, checked):
        """Handle cine button toggle - stop all cine playback when unchecked."""
        if not checked:
            # Stop cine mode on all views
            for label in self.view_labels.values():
                if isinstance(label, SliceViewLabel):
                    label.stop_cine()

    def eventFilter(self, obj, event):
        if obj == self.centralWidget() and event.type() == QEvent.Resize:
            # Delay the update to avoid geometry conflicts
            if self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled:
                # Use a timer to defer the update after resize is complete
                if not hasattr(self, '_resize_timer'):
                    self._resize_timer = QTimer()
                    self._resize_timer.setSingleShot(True)
                    self._resize_timer.timeout.connect(self.update_visible_views)
                self._resize_timer.stop()
                self._resize_timer.start(50)  # 50ms delay
        return super().eventFilter(obj, event)

    def numpy_to_qpixmap(self, array_2d: np.ndarray) -> QPixmap:
        if array_2d.dtype != np.uint8:
            array_2d = array_2d.astype(np.uint8)
        h, w = array_2d.shape
        q_img = QImage(array_2d.tobytes(), w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def update_view(self, ui_title: str, view_type: str):
        """Update a specific view with current slice data."""
        if ui_title not in self.view_labels:
            return
            
        label = self.view_labels[ui_title]
        
        # Skip update if label is not visible or has invalid size
        if not label.isVisible() or label.width() < 10 or label.height() < 10:
            return
        
        # If no file is loaded, keep views black
        if not self.file_loaded or self.data is None:
            return
            
        if view_type == 'segmentation':
            self.update_segmentation_view()
            return

        # Get the slice data
        slice_data = loader.get_slice_data(
            self.data, self.dims, self.slices, self.affine,
            self.intensity_min, self.intensity_max,
            rot_x_deg=self.rot_x_deg, rot_y_deg=self.rot_y_deg,
            view_type=view_type
        )
        
        # Convert to pixmap
        pixmap = self.numpy_to_qpixmap(slice_data)
        
        # Get label size
        label_size = label.size()
        
        # For zoom mode, we want to scale to a larger size first to maintain quality
        if isinstance(label, SliceViewLabel) and label.zoom_factor > 1.0:
            # Scale to actual zoomed size for better quality
            target_width = int(label_size.width() * label.zoom_factor)
            target_height = int(label_size.height() * label.zoom_factor)
            scaled = pixmap.scaled(
                QSize(target_width, target_height),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        else:
            # Normal scaling
            scaled = pixmap.scaled(
                QSize(label_size.width() - 2, label_size.height() - 2),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        
        # Use the new set_image_pixmap method if available (for SliceViewLabel)
        if isinstance(label, SliceViewLabel):
            label.set_image_pixmap(scaled)
        else:
            label.setPixmap(scaled)

    def update_segmentation_view(self):
        if 'segmentation' in self.view_labels:
            label = self.view_labels['segmentation']
            label.setText("Segmentation View\n\n[Add your segmentation data here]")
            label.setStyleSheet("""
                QLabel {
                    background-color: black;
                    border: 1px solid #666666;
                    color: #ffffff;
                    font-size: 18px;
                }
            """)

    def update_visible_views(self):
        if self.main_views_enabled:
            self.update_view('frontal', 'coronal')
            self.update_view('sagittal', 'sagittal')
            self.update_view('axial', 'axial')
        if self.oblique_view_enabled:
            self.update_view('oblique', 'oblique')
        if self.segmentation_view_enabled:
            self.update_view('segmentation', 'segmentation')

    # --- Mode Toggles ---
    def show_main_views_initially(self):
        for view_name in ['frontal', 'sagittal', 'axial']:
            if view_name in self.view_panels:
                self.view_panels[view_name].show()
                view_type = 'coronal' if view_name == 'frontal' else view_name
                self.update_view(view_name, view_type)
        if 'oblique' in self.view_panels:
            self.view_panels['oblique'].hide()
        if 'segmentation' in self.view_panels:
            self.view_panels['segmentation'].hide()

    def toggle_main_views(self, checked):
        self.main_views_enabled = checked
        for view_name in ['frontal', 'sagittal', 'axial']:
            if view_name in self.view_panels:
                if checked:
                    self.view_panels[view_name].show()
                    view_type = 'coronal' if view_name == 'frontal' else view_name
                    self.update_view(view_name, view_type)
                else:
                    self.view_panels[view_name].hide()

    def toggle_oblique_view(self, checked):
        self.oblique_view_enabled = checked
        if 'oblique' in self.view_panels:
            if checked:
                if self.segmentation_view_enabled:
                    self.segmentation_view_enabled = False
                    if 'segmentation' in self.view_panels:
                        self.view_panels['segmentation'].hide()
                    seg_btn = self.findChild(QPushButton, "mode_btn_1")
                    if seg_btn:
                        seg_btn.setChecked(False)
                self.view_panels['oblique'].show()
                self.update_view('oblique', 'oblique')
            else:
                self.view_panels['oblique'].hide()

    def toggle_segmentation_view(self, checked):
        self.segmentation_view_enabled = checked
        if 'segmentation' in self.view_panels:
            if checked:
                if self.oblique_view_enabled:
                    self.oblique_view_enabled = False
                    if 'oblique' in self.view_panels:
                        self.view_panels['oblique'].hide()
                    obl_btn = self.findChild(QPushButton, "mode_btn_2")
                    if obl_btn:
                        obl_btn.setChecked(False)
                self.view_panels['segmentation'].show()
                self.update_segmentation_view()
            else:
                self.view_panels['segmentation'].hide()
    # --- End Toggles ---

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.Box)
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("""
            QFrame { background-color: #e0e0e0; border: 2px solid #888888; }
            QPushButton[objectName^="export_btn_"] { border: 3px solid #ff0000; }
            QPushButton[objectName^="export_btn_"]:hover {
                background-color: #ffe0e0; border: 3px solid #cc0000;
            }
            QPushButton[objectName^="export_btn_"]:pressed { background-color: #ffc0c0; }
            QPushButton[objectName^="open_btn_"] {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: 2px solid #45a049;
            }
            QPushButton[objectName^="open_btn_"]:hover {
                background-color: #45a049;
                border: 2px solid #3d8b40;
            }
            QPushButton[objectName^="open_btn_"]:pressed {
                background-color: #3d8b40;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # File Loading group
        file_group = QGroupBox("Load File:")
        file_layout = QVBoxLayout()
        
        # Open NIfTI button
        open_nifti_btn = QPushButton("Open NIfTI File")
        open_nifti_btn.setObjectName("open_btn_nifti")
        open_nifti_btn.setMinimumHeight(35)
        open_nifti_btn.clicked.connect(self.open_nifti_file)
        file_layout.addWidget(open_nifti_btn)
        
        # Open DICOM button
        open_dicom_btn = QPushButton("Open DICOM Folder")
        open_dicom_btn.setObjectName("open_btn_dicom")
        open_dicom_btn.setMinimumHeight(35)
        open_dicom_btn.clicked.connect(self.open_dicom_folder)
        file_layout.addWidget(open_dicom_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Mode group
        mode_group = QGroupBox("Mode:")
        mode_layout = QGridLayout()
        self.mode_group_buttons = QButtonGroup(self)
        self.mode_group_buttons.setExclusive(False)
        for i in range(3):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"mode_btn_{i}")
            btn.setCheckable(True)
            mode_layout.addWidget(btn, 0, i)
            self.mode_group_buttons.addButton(btn, i)
            if i == 0: btn.clicked.connect(self.toggle_main_views)
            elif i == 1: btn.clicked.connect(self.toggle_segmentation_view)
            elif i == 2: btn.clicked.connect(self.toggle_oblique_view)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Tools
        tools_group = QGroupBox("Tools:")
        tools_layout = QGridLayout()
        self.tools_group_buttons = QButtonGroup(self)
        self.tools_group_buttons.setExclusive(True)
        for r in range(2):
            for c in range(3):
                btn = QPushButton()
                btn.setFixedSize(40, 40)
                btn.setObjectName(f"tool_btn_{r}_{c}")
                btn.setCheckable(True)
                tools_layout.addWidget(btn, r, c)
                self.tools_group_buttons.addButton(btn, r * 3 + c)
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        # Export
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
            print(f"Export triggered: {button.toolTip()}")

    def create_viewing_area(self):
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setSpacing(10)

        panels = [
            ("Frontal", 'coronal', 0, 0),
            ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0),
            ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1)
        ]

        for title, view_type, row, col in panels:
            panel = QFrame()
            panel.setFrameStyle(QFrame.Box)
            panel.setStyleSheet("QFrame { background-color: #2a2a2a; border: 2px solid #555555; }")
            vbox = QVBoxLayout(panel)
            vbox.setContentsMargins(5, 5, 5, 5)

            title_lbl = QLabel(title)
            title_lbl.setAlignment(Qt.AlignCenter)
            title_lbl.setStyleSheet("""
                QLabel {
                    color: white; font-size: 20px; font-weight: bold;
                    background-color: transparent; padding: 5px;
                }
            """)
            vbox.addWidget(title_lbl)

            if view_type != 'segmentation':
                view_area = SliceViewLabel(self, view_type, title)
            else:
                view_area = QLabel()
                view_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                view_area.setScaledContents(False)
                view_area.setObjectName(f"view_{title.lower()}")
                view_area.setStyleSheet("""
                    QLabel {
                        background-color: black;
                        border: 1px solid #666666;
                        color: #555555;
                        font-size: 18px;
                    }
                """)
                view_area.setAlignment(Qt.AlignCenter)
                view_area.setText("")

            vbox.addWidget(view_area)

            self.view_panels[title.lower()] = panel
            self.view_labels[title.lower()] = view_area
            grid.addWidget(panel, row, col)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

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
    viewer = MPRViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()