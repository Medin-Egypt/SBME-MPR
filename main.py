import tensorflow as tf
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QSizePolicy, QButtonGroup, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QSize, QEvent, QTimer, QPoint
from PyQt5.QtGui import QPixmap, QIcon, QImage, QPainter, QPen, QColor
import utils.loader as loader
import utils.detect_orientation as od
import time


# --- Custom QLabel for Mouse Wheel Interaction and Crosshairs ---
class SliceViewLabel(QLabel):
    """
    Custom QLabel that handles mouse wheel events to scroll through volume slices,
    contrast adjustment, zoom functionality, cine mode, and now draws/syncs crosshairs.
    """
    def __init__(self, parent_viewer, view_type, ui_title):
        super().__init__()
        self.parent_viewer = parent_viewer
        self.view_type = view_type
        self.ui_title = ui_title
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName(f"view_{self.ui_title.lower()}")
        self.setText("")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)

        # Crosshair state (normalized coordinates: 0.0 to 1.0)
        # These are used to draw the crosshairs relative to the image size
        self.crosshair_pos = QPoint(0, 0)
        self._dragging_crosshair = False

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
            self.cine_timer.start(1000 // self.cine_fps)
            
    def stop_cine(self):
        """Stop cine mode playback."""
        if self.cine_active:
            self.cine_active = False
            self.cine_timer.stop()
        
    def wheelEvent(self, event):
        slide_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_0")
        zoom_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_2")
        
        if zoom_btn and zoom_btn.isChecked():
            current_time = time.time() * 1000
            if current_time - self._last_zoom_time < self._zoom_cooldown:
                event.accept()
                return
            self._last_zoom_time = current_time
            delta = event.angleDelta().y()
            if abs(delta) < 15:
                event.accept()
                return
            zoom_step = 1.15
            if delta > 0:
                new_zoom = self.zoom_factor * zoom_step
                self.zoom_factor = min(new_zoom, 10.0)
            else:
                new_zoom = self.zoom_factor / zoom_step
                self.zoom_factor = max(new_zoom, 1.0)
                if self.zoom_factor == 1.0:
                    self.pan_offset_x = 0
                    self.pan_offset_y = 0
            self.parent_viewer.update_view(self.ui_title.lower(), self.view_type)
            event.accept()
        elif slide_btn and slide_btn.isChecked():
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

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.parent_viewer.maximized_view is None:
                self.parent_viewer.maximize_view(self.ui_title.lower())
            else:
                self.parent_viewer.restore_views()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        crosshair_tool_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_0") # Slide mode button is used for crosshair movement
        contrast_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_1")
        zoom_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_2")
        cine_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_2")
        
        if crosshair_tool_btn and crosshair_tool_btn.isChecked() and event.button() == Qt.LeftButton:
            self._dragging_crosshair = True
            self._update_crosshair(event.pos())
        elif cine_btn and cine_btn.isChecked() and event.button() == Qt.LeftButton:
            if self.cine_active:
                self.stop_cine()
            else:
                self.start_cine()
        elif zoom_btn and zoom_btn.isChecked() and event.button() == Qt.LeftButton:
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

        if self._dragging_crosshair:
            self._update_crosshair(event.pos())
            return
            
        elif self._panning and self._pan_start and zoom_btn and zoom_btn.isChecked():
            dx = event.x() - self._pan_start.x()
            dy = event.y() - self._pan_start.y()
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self._pan_start = event.pos()
            self._apply_zoom_and_pan()
        elif self._dragging and self._last_pos:
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
        if self._dragging_crosshair and event.button() == Qt.LeftButton:
            self._dragging_crosshair = False
        elif self._panning and event.button() == Qt.LeftButton:
            self._panning = False
        elif self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
        else:
            super().mouseReleaseEvent(event)
    
    def _update_crosshair(self, pos):
        """Updates crosshair position based on mouse position and triggers sync."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
            
        # Get actual image area size (pixmap bounding box) for accurate mapping
        # We assume the image is centered and scaled to fit or fill.
        
        # Simplified: Use QLabel dimensions for normalization
        width = self.width()
        height = self.height()

        # Clamp mouse position to within the label bounds
        norm_x = max(0.0, min(1.0, pos.x() / width))
        norm_y = max(0.0, min(1.0, pos.y() / height))
        
        self.crosshair_pos = QPoint(int(norm_x * width), int(norm_y * height))
        
        # Trigger the viewer to update the other views based on this new crosshair position
        self.parent_viewer.set_slice_from_crosshair(self.view_type, norm_x, norm_y)
        
        # Repaint this view to draw the new crosshair position
        self.update()

    def set_normalized_crosshair(self, norm_x, norm_y):
        """Sets the crosshair based on normalized coordinates (0.0 to 1.0)."""
        width = self.width()
        height = self.height()
        
        self.crosshair_pos = QPoint(int(norm_x * width), int(norm_y * height))
        self.update()

    def paintEvent(self, event):
        """Overrides paintEvent to draw the image and then the crosshairs."""
        super().paintEvent(event) # Draws the image first
        
        if self.parent_viewer.file_loaded and self.crosshair_pos.x() > 0 and self.crosshair_pos.y() > 0:
            painter = QPainter(self)
            
            # Use a unique color for the crosshairs (e.g., bright yellow)
            pen = QPen(QColor(255, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            
            x = self.crosshair_pos.x()
            y = self.crosshair_pos.y()
            
            # Draw horizontal line
            painter.drawLine(0, y, self.width(), y)
            
            # Draw vertical line
            painter.drawLine(x, 0, x, self.height())

            # Draw a small circle/cross where the lines intersect
            painter.drawEllipse(x - 5, y - 5, 10, 10)

    def _apply_zoom_and_pan(self):
        """Apply zoom and pan transformation to the stored original pixmap."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        
        label_size = self.contentsRect().size()

        # Ensure we have valid dimensions
        if label_size.width() < 10 or label_size.height() < 10:
            return
        zoomed_width = int(label_size.width() * self.zoom_factor)
        zoomed_height = int(label_size.height() * self.zoom_factor)
        zoomed_width = max(10, min(zoomed_width, 50000))
        zoomed_height = max(10, min(zoomed_height, 50000))
        zoomed_pixmap = self._original_pixmap.scaled(
            QSize(zoomed_width, zoomed_height),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        if self.zoom_factor > 1.0:
            max_offset_x = (zoomed_pixmap.width() - label_size.width()) // 2
            max_offset_y = (zoomed_pixmap.height() - label_size.height()) // 2
            self.pan_offset_x = max(-max_offset_x, min(max_offset_x, self.pan_offset_x))
            self.pan_offset_y = max(-max_offset_y, min(max_offset_y, self.pan_offset_y))
            center_x = zoomed_pixmap.width() // 2
            center_y = zoomed_pixmap.height() // 2
            crop_x = center_x - label_size.width() // 2 - self.pan_offset_x
            crop_y = center_y - label_size.height() // 2 - self.pan_offset_y
            crop_x = max(0, min(crop_x, zoomed_pixmap.width() - label_size.width()))
            crop_y = max(0, min(crop_y, zoomed_pixmap.height() - label_size.height()))
            cropped = zoomed_pixmap.copy(
                crop_x, crop_y,
                min(label_size.width(), zoomed_pixmap.width()),
                min(label_size.height(), zoomed_pixmap.height())
            )
            self.setPixmap(cropped)
        else:
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
        
        self.setMinimumSize(800, 600)
        self.setMaximumSize(16777215, 16777215)

        self.data = None
        self.affine = None
        self.dims = None
        self.intensity_min = 0
        self.intensity_max = 255
        self.file_loaded = False
        
        # Coordinates for crosshair position (normalized: 0.0 to 1.0)
        # S: Sagittal index (0) | C: Coronal index (1) | A: Axial index (2)
        self.norm_coords = {'S': 0.5, 'C': 0.5, 'A': 0.5}

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
        self.maximized_view = None

        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        sidebar = self.create_sidebar()
        self.viewing_area_widget = self.create_viewing_area()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.viewing_area_widget)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        self.add_image_to_button("mode_btn_0", "Icons/windows.png", "3 Main Views")
        self.add_image_to_button("mode_btn_1", "Icons/heart.png", "Segmentation View")
        self.add_image_to_button("mode_btn_2", "Icons/diagram.png", "Oblique View")
        self.add_image_to_button("tool_btn_0_0", "Icons/tab.png", "Slide/Crosshair Mode") # Renamed for clarity
        self.add_image_to_button("tool_btn_0_1", "Icons/brightness.png", "Contrast Mode")
        self.add_image_to_button("tool_btn_0_2", "Icons/loupe.png", "Zoom Mode")
        self.add_image_to_button("tool_btn_1_0", "Icons/expand.png", "Crop Mode")
        self.add_image_to_button("tool_btn_1_1", "Icons/rotating-arrow-to-the-right.png", "Rotate Mode")
        self.add_image_to_button("tool_btn_1_2", "Icons/video.png", "Cine Mode (Click view to start/stop)")
        self.add_image_to_button("export_btn_0", "Icons/NII.png", "NIFTI Export")
        self.add_image_to_button("export_btn_1", "Icons/DIC.png", "DICOM Export")

        main_views_btn = self.findChild(QPushButton, "mode_btn_0")
        if main_views_btn:
            main_views_btn.setChecked(True)
        
        self.centralWidget().installEventFilter(self)
        
        cine_btn = self.findChild(QPushButton, "tool_btn_1_2")
        if cine_btn:
            cine_btn.clicked.connect(self.handle_cine_button_toggle)
            
        self.show_main_views_initially()

    def set_slice_from_crosshair(self, source_view, norm_x, norm_y):
        """
        Updates the slice indices and crosshair positions across all views
        based on a click/drag in the source_view.
        """
        if not self.file_loaded or self.dims is None:
            return

        # 1. Update the normalized coordinates (S, C, A indices)
        # Note: The mapping of (x, y) to (S, C, A) depends on the view orientation
        if source_view == 'axial': # Axial (Z-plane, row 1, col 0) shows X vs Y
            # X-axis (horizontal) corresponds to S (Sagittal) dimension index
            # Y-axis (vertical) corresponds to C (Coronal) dimension index
            self.norm_coords['S'] = norm_x
            self.norm_coords['C'] = norm_y
            
            # The A (Axial) slice index is fixed by the current slice number in this view
            self.norm_coords['A'] = self.slices['axial'] / self.dims[2]
            
            # Update Coronal slice based on Y-position in Axial (Coronal is dim 1)
            self.slices['coronal'] = int(norm_y * self.dims[1]) % self.dims[1]
            
            # Update Sagittal slice based on X-position in Axial (Sagittal is dim 0)
            self.slices['sagittal'] = int(norm_x * self.dims[0]) % self.dims[0]

        elif source_view == 'coronal': # Coronal (Y-plane, row 0, col 0) shows X vs Z
            # X-axis (horizontal) corresponds to S (Sagittal) dimension index
            # Y-axis (vertical) corresponds to A (Axial) dimension index
            self.norm_coords['S'] = norm_x
            self.norm_coords['A'] = norm_y
            
            # The C (Coronal) slice index is fixed by the current slice number in this view
            self.norm_coords['C'] = self.slices['coronal'] / self.dims[1]

            # Update Axial slice based on Y-position in Coronal (Axial is dim 2)
            self.slices['axial'] = int(norm_y * self.dims[2]) % self.dims[2]

            # Update Sagittal slice based on X-position in Coronal (Sagittal is dim 0)
            self.slices['sagittal'] = int(norm_x * self.dims[0]) % self.dims[0]

        elif source_view == 'sagittal': # Sagittal (X-plane, row 0, col 1) shows Y vs Z
            # X-axis (horizontal) corresponds to C (Coronal) dimension index
            # Y-axis (vertical) corresponds to A (Axial) dimension index
            self.norm_coords['C'] = norm_x
            self.norm_coords['A'] = norm_y
            
            # The S (Sagittal) slice index is fixed by the current slice number in this view
            self.norm_coords['S'] = self.slices['sagittal'] / self.dims[0]
            
            # Update Axial slice based on Y-position in Sagittal (Axial is dim 2)
            self.slices['axial'] = int(norm_y * self.dims[2]) % self.dims[2]
            
            # Update Coronal slice based on X-position in Sagittal (Coronal is dim 1)
            self.slices['coronal'] = int(norm_x * self.dims[1]) % self.dims[1]
            
        # 2. Force all views to update (both slices and crosshairs)
        self.update_all_views()


    def update_all_views(self):
        """Forces all visible views to redraw their image and sync their crosshairs."""
        views_to_update = []
        if self.main_views_enabled:
            views_to_update.extend([
                ('frontal', 'coronal'), 
                ('sagittal', 'sagittal'), 
                ('axial', 'axial')
            ])
        if self.oblique_view_enabled:
            views_to_update.append(('oblique', 'oblique'))
        if self.segmentation_view_enabled:
            views_to_update.append(('segmentation', 'segmentation'))

        for ui_title, view_type in views_to_update:
            self.update_view(ui_title, view_type, sync_crosshair=True)


    # --- Rest of the Methods ---

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
                # Initialize normalized coordinates to the center of the volume
                self.norm_coords = {'S': 0.5, 'C': 0.5, 'A': 0.5}
                
                self.show_main_views_initially()
                self.update_all_views() # Initial update
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
                self.slices = {
                    'axial': self.dims[2] // 2,
                    'coronal': self.dims[1] // 2,
                    'sagittal': self.dims[0] // 2,
                    'oblique': self.dims[2] // 2
                }
                self.norm_coords = {'S': 0.5, 'C': 0.5, 'A': 0.5}
                
                middle_slice_data = self.data[:, :, self.slices['axial']]
                orientation, confidence = od.predict_dicom_image(middle_slice_data)
                orientation_info = f"\n\nDetected Orientation: {orientation} ({confidence:.2f}% confidence)"
                
                self.show_main_views_initially()
                self.update_all_views() # Initial update
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"DICOM folder loaded successfully!\nDimensions: {self.dims}{orientation_info}\n\n{"\n".join(organ_data)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM folder:\n{str(e)}")

    def handle_cine_button_toggle(self, checked):
        """Handle cine button toggle - stop all cine playback when unchecked."""
        if not checked:
            for label in self.view_labels.values():
                if isinstance(label, SliceViewLabel):
                    label.stop_cine()

    def eventFilter(self, obj, event):
        if obj == self.centralWidget() and event.type() == QEvent.Resize:
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

    def update_view(self, ui_title: str, view_type: str, sync_crosshair=False):
        """Update a specific view with current slice data."""
        if ui_title not in self.view_labels:
            return
        label = self.view_labels[ui_title]
        if not label.isVisible() or label.width() < 10 or label.height() < 10:
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
            view_type=view_type
        )
        pixmap = self.numpy_to_qpixmap(slice_data)
        label_size = label.size()
        if isinstance(label, SliceViewLabel) and label.zoom_factor > 1.0:
            target_width = int(label_size.width() * label.zoom_factor)
            target_height = int(label_size.height() * label.zoom_factor)
            scaled = pixmap.scaled(
                QSize(target_width, target_height),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        else:
            scaled = pixmap.scaled(
                QSize(label_size.width() - 2, label_size.height() - 2),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        if isinstance(label, SliceViewLabel):
            label.set_image_pixmap(scaled)
            
            # --- Crosshair Synchronization Logic ---
            if sync_crosshair:
                # Map global S, C, A normalized coords to this view's X and Y
                if view_type == 'axial':
                    # Axial (X-axis=S, Y-axis=C)
                    norm_x = self.norm_coords['S']
                    norm_y = self.norm_coords['C']
                elif view_type == 'coronal':
                    # Coronal (X-axis=S, Y-axis=A)
                    norm_x = self.norm_coords['S']
                    norm_y = self.norm_coords['A']
                elif view_type == 'sagittal':
                    # Sagittal (X-axis=C, Y-axis=A)
                    norm_x = self.norm_coords['C']
                    norm_y = self.norm_coords['A']
                else:
                    norm_x, norm_y = 0.5, 0.5 # Default for oblique/segmentation
                
                label.set_normalized_crosshair(norm_x, norm_y)
            # --- End Crosshair Synchronization Logic ---

        else:
            label.setPixmap(scaled)

    def update_segmentation_view(self):
        if 'segmentation' in self.view_labels:
            label = self.view_labels['segmentation']
            label.setText("Segmentation View\n\n[Add your segmentation data here]")
            
    def update_visible_views(self):
        # Trigger an update of the visible views only
        visible_views = [name for name, panel in self.view_panels.items() if panel.isVisible()]
        for view_name in visible_views:
            view_type = view_name
            if view_name == 'frontal':
                view_type = 'coronal'
            self.update_view(view_name, view_type, sync_crosshair=True) # Always sync on redraw
            
    def maximize_view(self, view_name):
        """Hides all other views, keeping only the selected one visible."""
        if not (self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled):
            return

        self.maximized_view = view_name
        
        # Hide all panels except the one to be maximized
        for name, panel in self.view_panels.items():
            if name != view_name:
                panel.hide()
            else:
                self.viewing_grid.removeWidget(panel)
                # Span 2 rows and 2 columns to take full grid space
                self.viewing_grid.addWidget(panel, 0, 0, 2, 2) 
                panel.show()
                
        # Force a layout recalculation and redraw
        self.viewing_grid.invalidate()
        QApplication.processEvents()
        self.update_visible_views()
        
    def restore_views(self):
        """Restores the original 4-view layout."""
        if self.maximized_view is None:
            return
            
        max_panel = self.view_panels[self.maximized_view]
        self.viewing_grid.removeWidget(max_panel)
        self.maximized_view = None
        
        # Re-add all panels to their original grid positions
        panels = [
            ("Frontal", 'coronal', 0, 0),
            ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0),
            ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1)
        ]
        
        for title, view_type, row, col in panels:
            panel = self.view_panels[title.lower()]
            self.viewing_grid.addWidget(panel, row, col)


        # Ensure correct mode is displayed (this will hide the extras)
        if self.main_views_enabled:
            self.toggle_main_views(True)
        elif self.oblique_view_enabled:
            self.toggle_oblique_view(True)
        elif self.segmentation_view_enabled:
            self.toggle_segmentation_view(True)

        # Force a layout recalculation and redraw
        self.viewing_grid.invalidate()
        QApplication.processEvents()
        self.update_visible_views()


    def show_main_views_initially(self):
        """Sets the initial state: 3 main views visible, 4th panel empty/hidden."""
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False
        
        # Ensure correct initial placement/visibility
        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial']:
                panel.show()
                # Grid placement is handled in create_viewing_area for the 2x2 layout
                self.update_view(view_name, 'coronal' if view_name == 'frontal' else view_name)
            else:
                panel.hide()

    def toggle_main_views(self, checked):
        if not checked: return
        
        self.restore_views() 
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False
        
        # Uncheck other buttons manually
        obl_btn = self.findChild(QPushButton, "mode_btn_2")
        seg_btn = self.findChild(QPushButton, "mode_btn_1")
        if obl_btn: obl_btn.setChecked(False)
        if seg_btn: seg_btn.setChecked(False)

        # Show 3 main views, hide the other two
        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial']:
                panel.show()
            else:
                panel.hide()
        
        self.update_visible_views()

    def toggle_oblique_view(self, checked):
        if not checked and self.oblique_view_enabled:
            # If the user tries to uncheck it, switch to main views instead of hiding everything
            self.toggle_main_views(True)
            self.findChild(QPushButton, "mode_btn_0").setChecked(True)
            return

        self.restore_views()
        self.oblique_view_enabled = True
        self.main_views_enabled = False
        self.segmentation_view_enabled = False
        
        # Uncheck other buttons manually
        main_btn = self.findChild(QPushButton, "mode_btn_0")
        seg_btn = self.findChild(QPushButton, "mode_btn_1")
        if main_btn: main_btn.setChecked(False)
        if seg_btn: seg_btn.setChecked(False)

        # Show 3 main views + Oblique in 4th quadrant
        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial', 'oblique']:
                panel.show()
                if view_name == 'oblique':
                    self.update_view(view_name, view_name)
            else:
                panel.hide()
        
        self.update_visible_views()
            
    def toggle_segmentation_view(self, checked):
        if not checked and self.segmentation_view_enabled:
            # If the user tries to uncheck it, switch to main views instead of hiding everything
            self.toggle_main_views(True)
            self.findChild(QPushButton, "mode_btn_0").setChecked(True)
            return

        self.restore_views()
        self.segmentation_view_enabled = True
        self.main_views_enabled = False
        self.oblique_view_enabled = False
        
        # Uncheck other buttons manually
        main_btn = self.findChild(QPushButton, "mode_btn_0")
        obl_btn = self.findChild(QPushButton, "mode_btn_2")
        if main_btn: main_btn.setChecked(False)
        if obl_btn: obl_btn.setChecked(False)

        # Show 3 main views + Segmentation in 4th quadrant
        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial', 'segmentation']:
                panel.show()
                if view_name == 'segmentation':
                    self.update_segmentation_view()
            else:
                panel.hide()
                
        self.update_visible_views()
    
    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFrameStyle(QFrame.Box)
        sidebar.setFixedWidth(200)

        layout = QVBoxLayout(sidebar)
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
        self.viewing_grid = QGridLayout(widget)
        self.viewing_grid.setSpacing(10)
        self.viewing_grid.setContentsMargins(0, 0, 0, 0)
        
        # Define the exact layout positions for all 5 panels
        panels = [
            ("Frontal", 'coronal', 0, 0),
            ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0),
            ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1) # Overlaps oblique - visibility controls which is shown
        ]

        for title, view_type, row, col in panels:
            panel = QFrame()
            panel.setObjectName(f"viewing_panel_{title.lower()}")
            panel.setFrameStyle(QFrame.Box)
            vbox = QVBoxLayout(panel)
            vbox.setContentsMargins(5, 5, 5, 5)

            title_lbl = QLabel(title)
            title_lbl.setObjectName(f"view_title_{title.lower()}")
            title_lbl.setAlignment(Qt.AlignCenter)
            vbox.addWidget(title_lbl)

            if view_type != 'segmentation':
                # Note: We assign 'frontal' view_type to 'coronal' in code to match data indexing
                view_area = SliceViewLabel(self, view_type if view_type != 'frontal' else 'coronal', title)
            else:
                view_area = QLabel()
                view_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                view_area.setScaledContents(False)
                view_area.setObjectName(f"view_{title.lower()}")
                view_area.setAlignment(Qt.AlignCenter)
                view_area.setText("")

            vbox.addWidget(view_area)

            self.view_panels[title.lower()] = panel
            self.view_labels[title.lower()] = view_area
            # IMPORTANT: Add all panels to the grid initially in their defined positions
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
    
    # Load the stylesheet from the external file
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
