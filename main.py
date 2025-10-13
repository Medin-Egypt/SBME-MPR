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
    contrast adjustment, zoom functionality, cine mode, crop mode, and draws/syncs crosshairs.
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
        self.normalized_crosshair_x = 0.5
        self.normalized_crosshair_y = 0.5
        self._dragging_crosshair = False

        # Crop mode state
        self._crop_start = None
        self._crop_end = None
        self._is_cropping = False

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
        
        if self.view_type in ('axial', 'oblique'):
            max_dim_index = 2
        elif self.view_type == 'coronal':
            max_dim_index = 1
        elif self.view_type == 'sagittal':
            max_dim_index = 0
        else:
            return
        
        max_slice = self.parent_viewer.dims[max_dim_index]
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
        crosshair_tool_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_0")
        contrast_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_1")
        zoom_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_2")
        crop_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_0")
        cine_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_2")
        
        if crop_btn and crop_btn.isChecked() and event.button() == Qt.LeftButton:
            self._is_cropping = True
            self._crop_start = event.pos()
            self._crop_end = event.pos()
        elif crosshair_tool_btn and crosshair_tool_btn.isChecked() and event.button() == Qt.LeftButton:
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
        crop_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_0")

        if self._is_cropping and crop_btn and crop_btn.isChecked():
            self._crop_end = event.pos()
            self.update()
        elif self._dragging_crosshair:
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
            
            self.parent_viewer.update_all_views()
            
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        crop_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_0")
        
        if self._is_cropping and event.button() == Qt.LeftButton and crop_btn and crop_btn.isChecked():
            self._is_cropping = False
            if self._crop_start and self._crop_end:
                self.parent_viewer.finalize_crop(self.view_type, self._crop_start, self._crop_end, self.size())
        elif self._dragging_crosshair and event.button() == Qt.LeftButton:
            self._dragging_crosshair = False
        elif self._panning and event.button() == Qt.LeftButton:
            self._panning = False
        elif self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
        else:
            super().mouseReleaseEvent(event)
    
    def _update_crosshair(self, pos):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
            
        label_width = self.width()
        label_height = self.height()
        
        zoomed_width = int(label_width * self.zoom_factor)
        zoomed_height = int(label_height * self.zoom_factor)
        
        center_offset_x = (label_width - zoomed_width) / 2
        center_offset_y = (label_height - zoomed_height) / 2
        
        x_on_zoomed_image = pos.x() - center_offset_x + self.pan_offset_x 
        y_on_zoomed_image = pos.y() - center_offset_y + self.pan_offset_y
        
        norm_x = x_on_zoomed_image / zoomed_width
        norm_y = y_on_zoomed_image / zoomed_height

        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        self.normalized_crosshair_x = norm_x
        self.normalized_crosshair_y = norm_y
        
        self.parent_viewer.set_slice_from_crosshair(self.view_type, norm_x, norm_y)
        
        self.update()

    def set_normalized_crosshair(self, norm_x, norm_y):
        self.normalized_crosshair_x = norm_x
        self.normalized_crosshair_y = norm_y
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        crop_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_0")
        
        # Draw crop rectangle while dragging
        if crop_btn and crop_btn.isChecked() and self._is_cropping and self._crop_start and self._crop_end:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            
            x1 = min(self._crop_start.x(), self._crop_end.x())
            y1 = min(self._crop_start.y(), self._crop_end.y())
            x2 = max(self._crop_start.x(), self._crop_end.x())
            y2 = max(self._crop_start.y(), self._crop_end.y())
            
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.end()
        
        # Draw stored crop bounds
        if crop_btn and crop_btn.isChecked() and self.parent_viewer.crop_bounds:
            bounds = self.parent_viewer.crop_bounds
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            
            # Get current view dimensions
            label_width = self.width()
            label_height = self.height()
            
            # Calculate rectangle based on view type
            if self.view_type == 'axial':
                x1 = int(bounds['sagittal_min'] * label_width)
                x2 = int(bounds['sagittal_max'] * label_width)
                y1 = int(bounds['coronal_min'] * label_height)
                y2 = int(bounds['coronal_max'] * label_height)
            elif self.view_type == 'coronal':
                x1 = int(bounds['sagittal_min'] * label_width)
                x2 = int(bounds['sagittal_max'] * label_width)
                y1 = int(bounds['axial_min'] * label_height)
                y2 = int(bounds['axial_max'] * label_height)
            elif self.view_type == 'sagittal':
                x1 = int(bounds['coronal_min'] * label_width)
                x2 = int(bounds['coronal_max'] * label_width)
                y1 = int(bounds['axial_min'] * label_height)
                y2 = int(bounds['axial_max'] * label_height)
            else:
                painter.end()
                return
            
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.end()
        
        if self.parent_viewer.file_loaded and self._original_pixmap and not self._original_pixmap.isNull():
            painter = QPainter(self)
            
            label_width = self.width()
            label_height = self.height()
            
            zoomed_width = int(label_width * self.zoom_factor)
            zoomed_height = int(label_height * self.zoom_factor)

            center_offset_x = (label_width - zoomed_width) / 2
            center_offset_y = (label_height - zoomed_height) / 2
            
            draw_x = int( (self.normalized_crosshair_x * zoomed_width) + center_offset_x - self.pan_offset_x )
            draw_y = int( (self.normalized_crosshair_y * zoomed_height) + center_offset_y - self.pan_offset_y )

            colors = self.parent_viewer.view_colors
            h_color, v_color = None, None

            if self.view_type == 'axial':
                h_color = colors['coronal']
                v_color = colors['sagittal']
            elif self.view_type == 'coronal':
                h_color = colors['axial']
                v_color = colors['sagittal']
            elif self.view_type == 'sagittal':
                h_color = colors['axial']
                v_color = colors['coronal']
            
            if h_color:
                pen_h = QPen(h_color)
                pen_h.setWidth(1)
                painter.setPen(pen_h)
                painter.drawLine(0, draw_y, self.width(), draw_y)

            if v_color:
                pen_v = QPen(v_color)
                pen_v.setWidth(1)
                painter.setPen(pen_v)
                painter.drawLine(draw_x, 0, draw_x, self.height())
            
            if 0 <= draw_x <= self.width() and 0 <= draw_y <= self.height():
                intersect_pen = QPen(QColor(255, 255, 0))
                intersect_pen.setWidth(2)
                painter.setPen(intersect_pen)
                painter.drawEllipse(draw_x - 4, draw_y - 4, 8, 8)
            
            painter.end()

    def _apply_zoom_and_pan(self):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        label_size = self.contentsRect().size()
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
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self.setPixmap(zoomed_pixmap)
        
        self.update()

    def set_image_pixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._apply_zoom_and_pan()
    
    def reset_zoom(self):
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
        
        # Remove default title bar
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        self.view_colors = {
            'axial': QColor(100, 220, 100),    # Green
            'coronal': QColor(100, 150, 255), # Blue
            'sagittal': QColor(255, 100, 100),   # Red
        }

        self.data = None
        self.affine = None
        self.dims = None
        self.intensity_min = 0
        self.intensity_max = 255
        self.file_loaded = False
        
        # Store original contrast values
        self.original_intensity_min = 0
        self.original_intensity_max = 255
        
        # Crop bounds (normalized 0-1 coordinates)
        self.crop_bounds = None
        self.original_data = None
        
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
        main_layout.addWidget(self.viewing_area_widget)
        main_layout.addWidget(sidebar)
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

    def create_title_bar(self):
        """Create custom title bar with window controls"""
        title_bar = QWidget()
        title_bar.setObjectName("custom_title_bar")
        title_bar.setFixedHeight(35)
        
        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Title label
        title_label = QLabel("MPR VIEWER")
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
        """Iterates through all view labels and resets their zoom/pan."""
        for label in self.view_labels.values():
            if isinstance(label, SliceViewLabel):
                label.reset_zoom()

    def reset_contrast(self):
        """Resets the window/level to the initial values from file load."""
        self.intensity_min = self.original_intensity_min
        self.intensity_max = self.original_intensity_max

    def reset_rotation(self):
        """Resets the oblique rotation angles to zero."""
        self.rot_x_deg = 0
        self.rot_y_deg = 0

    def reset_crosshair_and_slices(self):
        """Resets crosshairs to the center and slices to the middle."""
        self.norm_coords = {'S': 0.5, 'C': 0.5, 'A': 0.5}
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
            self.slices['axial'] = int((1 - norm_y) * (self.dims[2] - 1))
            self.slices['sagittal'] = int(norm_x * (self.dims[0] - 1))
        elif source_view == 'sagittal':
            self.norm_coords['C'] = norm_x
            self.norm_coords['A'] = norm_y
            self.slices['axial'] = int((1 - norm_y) * (self.dims[2] - 1))
            self.slices['coronal'] = int(norm_x * (self.dims[1] - 1))
            
        self.update_all_views()

    def finalize_crop(self, view_type, start_pos, end_pos, label_size):
        """Convert screen coordinates to normalized crop bounds and store them."""
        if not self.file_loaded:
            return
        
        # Normalize coordinates to 0-1 range
        x1 = min(start_pos.x(), end_pos.x()) / label_size.width()
        x2 = max(start_pos.x(), end_pos.x()) / label_size.width()
        y1 = min(start_pos.y(), end_pos.y()) / label_size.height()
        y2 = max(start_pos.y(), end_pos.y()) / label_size.height()
        
        # Clamp to valid range
        x1, x2 = max(0.0, x1), min(1.0, x2)
        y1, y2 = max(0.0, y1), min(1.0, y2)
        
        # Initialize crop bounds if not exists
        if self.crop_bounds is None:
            self.crop_bounds = {
                'sagittal_min': 0.0, 'sagittal_max': 1.0,
                'coronal_min': 0.0, 'coronal_max': 1.0,
                'axial_min': 0.0, 'axial_max': 1.0
            }
        
        # Update bounds based on view type
        if view_type == 'axial':
            self.crop_bounds['sagittal_min'] = x1
            self.crop_bounds['sagittal_max'] = x2
            self.crop_bounds['coronal_min'] = y1
            self.crop_bounds['coronal_max'] = y2
        elif view_type == 'coronal':
            self.crop_bounds['sagittal_min'] = x1
            self.crop_bounds['sagittal_max'] = x2
            self.crop_bounds['axial_min'] = y1
            self.crop_bounds['axial_max'] = y2
        elif view_type == 'sagittal':
            self.crop_bounds['coronal_min'] = x1
            self.crop_bounds['coronal_max'] = x2
            self.crop_bounds['axial_min'] = y1
            self.crop_bounds['axial_max'] = y2
        
        # Update all views to show the crop rectangle
        self.update_all_views()

    def apply_crop(self):
        """Apply the current crop bounds to the data, permanently cropping the volume."""
        if not self.file_loaded or self.crop_bounds is None or self.original_data is None:
            return
        
        # Convert normalized bounds to voxel indices
        sag_min = int(self.crop_bounds['sagittal_min'] * self.dims[0])
        sag_max = int(self.crop_bounds['sagittal_max'] * self.dims[0])
        cor_min = int(self.crop_bounds['coronal_min'] * self.dims[1])
        cor_max = int(self.crop_bounds['coronal_max'] * self.dims[1])
        ax_min = int(self.crop_bounds['axial_min'] * self.dims[2])
        ax_max = int(self.crop_bounds['axial_max'] * self.dims[2])
        
        # Ensure valid ranges
        sag_min, sag_max = max(0, sag_min), min(self.dims[0], sag_max)
        cor_min, cor_max = max(0, cor_min), min(self.dims[1], cor_max)
        ax_min, ax_max = max(0, ax_min), min(self.dims[2], ax_max)
        
        # Apply crop to data
        self.data = self.data[sag_min:sag_max, cor_min:cor_max, ax_min:ax_max].copy()
        self.dims = self.data.shape
        
        # Clear crop bounds after applying
        self.crop_bounds = None
        
        # Reset slices to middle of cropped volume
        self.reset_crosshair_and_slices()
        
        # Update all views
        self.update_all_views()
        
        QMessageBox.information(self, "Crop Applied", f"Volume cropped successfully!\nNew dimensions: {self.dims}")

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

        for ui_title, view_type in views_to_update:
            self.update_view(ui_title, view_type, sync_crosshair=True)

    def open_nifti_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if file_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max = loader.load_nifti_data(file_path)
                self.file_loaded = True
                
                # Store original contrast values on load
                self.original_intensity_min = self.intensity_min
                self.original_intensity_max = self.intensity_max
                
                # Store original data for crop reset
                self.original_data = self.data.copy()
                self.crop_bounds = None

                self.reset_crosshair_and_slices()
                self.reset_all_zooms()
                self.reset_rotation()
                
                self.show_main_views_initially()
                self.update_all_views()
                QMessageBox.information(self, "Success", f"NIfTI file loaded successfully!\nDimensions: {self.dims}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load NIfTI file:\n{str(e)}")

    def open_dicom_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", "", QFileDialog.ShowDirsOnly)
        if folder_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max, organ_data = loader.load_dicom_data(folder_path)
                self.file_loaded = True
                
                # Store original contrast values on load
                self.original_intensity_min = self.intensity_min
                self.original_intensity_max = self.intensity_max
                
                # Store original data for crop reset
                self.original_data = self.data.copy()
                self.crop_bounds = None

                self.reset_crosshair_and_slices()
                self.reset_all_zooms()
                self.reset_rotation()
                
                middle_slice_data = self.data[:, :, self.slices['axial']]
                orientation, confidence = od.predict_dicom_image(middle_slice_data)
                orientation_info = f"\n\nDetected Orientation: {orientation} ({confidence:.2f}% confidence)"
                
                self.show_main_views_initially()
                self.update_all_views()
                
                QMessageBox.information(
                    self, "Success",
                    f"DICOM folder loaded successfully!\nDimensions: {self.dims}{orientation_info}\n\n{'\n'.join(organ_data)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM folder:\n{str(e)}")

    def handle_cine_button_toggle(self, checked):
        if not checked:
            for label in self.view_labels.values():
                if isinstance(label, SliceViewLabel):
                    label.stop_cine()

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

    def update_view(self, ui_title: str, view_type: str, sync_crosshair=False):
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
        
        if isinstance(label, SliceViewLabel):
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
        else:
            scaled = pixmap.scaled(
                QSize(label.size().width() - 2, label.size().height() - 2),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled)

    def update_segmentation_view(self):
        if 'segmentation' in self.view_labels:
            label = self.view_labels['segmentation']
            label.setText("Segmentation View\n\n[Add your segmentation data here]")
            
    def update_visible_views(self):
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
        
        panels = [
            ("Frontal", 'coronal', 0, 0), ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0), ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1)
        ]
        
        for title, view_type, row, col in panels:
            panel = self.view_panels[title.lower()]
            self.viewing_grid.addWidget(panel, row, col)

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
        
        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial']:
                panel.show()
                self.update_view(view_name, 'coronal' if view_name == 'frontal' else view_name)
            else:
                panel.hide()

    def toggle_main_views(self, checked):
        if not checked:
            return
        
        self.restore_views()
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False
        
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
        
        self.findChild(QPushButton, "mode_btn_0").setChecked(False)
        self.findChild(QPushButton, "mode_btn_1").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial', 'oblique']:
                panel.show()
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
        
        self.findChild(QPushButton, "mode_btn_0").setChecked(False)
        self.findChild(QPushButton, "mode_btn_2").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['frontal', 'sagittal', 'axial', 'segmentation']:
                panel.show()
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
            if i == 0:
                btn.clicked.connect(self.toggle_main_views)
            elif i == 1:
                btn.clicked.connect(self.toggle_segmentation_view)
            elif i == 2:
                btn.clicked.connect(self.toggle_oblique_view)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

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
                btn.setObjectName(f"tool_btn_{r}_{c}")
                btn.setCheckable(True)
                tools_layout.addWidget(btn, r, c)
                self.tools_group_buttons.addButton(btn, r * 3 + c)
        
        tools_main_layout.addWidget(tools_grid_widget)

        # Add the Reset button
        reset_btn = QPushButton("Reset")
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
            
            if button.objectName() == "export_btn_1":  # DICOM export
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
                
                print(f"Export triggered: {button.toolTip()}")
                QMessageBox.information(self, "Export", f"DICOM export initiated.\nCurrent dimensions: {self.dims}")
            else:
                print(f"Export triggered: {button.toolTip()}")

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