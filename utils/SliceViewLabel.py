from PyQt5.QtWidgets import (
    QPushButton, QLabel, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor
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
        # This will be kept in sync with parent_viewer.global_zoom_factor
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

    # --- MODIFIED METHOD ---
    def _cine_next_slice(self):
        """Advance to the previous slice in cine mode."""
        if not self.cine_active or not self.parent_viewer.file_loaded:
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

        # Call the new central method to update slice, crosshairs, and all views
        self.parent_viewer.set_slice_from_scroll(self.view_type, new_slice)

    # --- END MODIFIED METHOD ---

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

    # --- MODIFIED METHOD ---
    def wheelEvent(self, event):
        """Handle mouse wheel events for scrolling through slices or zooming."""
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

            self.parent_viewer.change_global_zoom(delta)
            event.accept()

        elif slide_btn and slide_btn.isChecked():
            if not self.parent_viewer.file_loaded:
                return

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

            # Call the new central method to update slice, crosshairs, and all views
            self.parent_viewer.set_slice_from_scroll(self.view_type, new_slice)
            event.accept()
        else:
            super().wheelEvent(event)

    # --- END MODIFIED METHOD ---

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
            # Check the global zoom factor for panning
            if self.parent_viewer.global_zoom_factor > 1.0:
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

        # Use combined zoom factor for crosshair calculation (Uniform Scale + User Zoom)
        default_scale = self.parent_viewer.default_scale_factor if hasattr(self.parent_viewer,
                                                                           'default_scale_factor') else 1.0
        self.zoom_factor = self.parent_viewer.global_zoom_factor  # Sync local factor
        combined_zoom_factor = default_scale * self.zoom_factor

        original_img_w = self._original_pixmap.width()
        original_img_h = self._original_pixmap.height()

        zoomed_width = int(original_img_w * combined_zoom_factor)
        zoomed_height = int(original_img_h * combined_zoom_factor)

        center_offset_x = (label_width - zoomed_width) / 2
        center_offset_y = (label_height - zoomed_height) / 2

        # Adjust position for pan offset
        x_on_zoomed_image = pos.x() - center_offset_x - self.pan_offset_x  # CORRECTED
        y_on_zoomed_image = pos.y() - center_offset_y - self.pan_offset_y  # CORRECTED

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

            # Draw the normalized rectangle on the label
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.end()

        if self.parent_viewer.file_loaded and self._original_pixmap and not self._original_pixmap.isNull():
            painter = QPainter(self)

            label_width = self.width()
            label_height = self.height()

            # Recalculate zoomed size for crosshair positioning (must match _apply_zoom_and_pan)
            default_scale = self.parent_viewer.default_scale_factor if hasattr(self.parent_viewer,
                                                                               'default_scale_factor') else 1.0
            self.zoom_factor = self.parent_viewer.global_zoom_factor
            combined_zoom_factor = default_scale * self.zoom_factor

            original_img_w = self._original_pixmap.width()
            original_img_h = self._original_pixmap.height()

            zoomed_width = int(original_img_w * combined_zoom_factor)
            zoomed_height = int(original_img_h * combined_zoom_factor)

            center_offset_x = (label_width - zoomed_width) / 2
            center_offset_y = (label_height - zoomed_height) / 2

            # Calculate screen coordinates for crosshair based on normalized position, zoom, and pan
            draw_x = int(
                (self.normalized_crosshair_x * zoomed_width) + center_offset_x + self.pan_offset_x)  # CORRECTED
            draw_y = int(
                (self.normalized_crosshair_y * zoomed_height) + center_offset_y + self.pan_offset_y)  # CORRECTED

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
        # Sync local zoom factor from viewer's global factor
        self.zoom_factor = self.parent_viewer.global_zoom_factor

        if self._original_pixmap is None or self._original_pixmap.isNull():
            return

        label_size = self.contentsRect().size()
        if label_size.width() < 10 or label_size.height() < 10:
            return

        # CHANGE 1: Use a combined scale factor (Uniform Scale + User Zoom)
        default_scale = self.parent_viewer.default_scale_factor if hasattr(self.parent_viewer,
                                                                           'default_scale_factor') else 1.0

        # --- New Core Scaling Logic ---
        # The image size (base scaled by default_scale) is multiplied by the user's zoom.
        original_img_w = self._original_pixmap.width()
        original_img_h = self._original_pixmap.height()

        # 1. Calculate the BASE size (uniform scale)
        base_w = int(original_img_w * default_scale)
        base_h = int(original_img_h * default_scale)

        # 2. Apply the user zoom (self.zoom_factor) to the BASE size
        zoomed_width = int(base_w * self.zoom_factor)
        zoomed_height = int(base_h * self.zoom_factor)
        # -----------------------------

        zoomed_width = max(10, min(zoomed_width, 50000))
        zoomed_height = max(10, min(zoomed_height, 50000))

        # Re-scale the original pixmap to the final calculated size (zoomed_width, zoomed_height)
        zoomed_pixmap = self._original_pixmap.scaled(
            QSize(zoomed_width, zoomed_height),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Pan/Crop logic based on whether the content is bigger than the container
        # This check now determines if panning/cropping is necessary, not scaling.
        content_is_bigger = zoomed_width > label_size.width() or zoomed_height > label_size.height()

        if content_is_bigger:
            # Pan constraints and application (identical to previous version)
            max_offset_x = (zoomed_pixmap.width() - label_size.width()) // 2
            max_offset_y = (zoomed_pixmap.height() - label_size.height()) // 2

            # Clamp pan offsets
            self.pan_offset_x = max(-max_offset_x, min(max_offset_x, self.pan_offset_x))
            self.pan_offset_y = max(-max_offset_y, min(max_offset_y, self.pan_offset_y))

            center_x = zoomed_pixmap.width() // 2
            center_y = zoomed_pixmap.height() // 2

            # Calculate crop area for panning
            crop_x = center_x - label_size.width() // 2 - self.pan_offset_x
            crop_y = center_y - label_size.height() // 2 - self.pan_offset_y

            # Clamp crop area to stay within zoomed pixmap bounds
            crop_x = max(0, min(crop_x, zoomed_pixmap.width() - label_size.width()))
            crop_y = max(0, min(crop_y, zoomed_pixmap.height() - label_size.height()))

            cropped = zoomed_pixmap.copy(
                crop_x, crop_y,
                min(label_size.width(), zoomed_pixmap.width()),
                min(label_size.height(), zoomed_pixmap.height())
            )
            self.setPixmap(cropped)
        else:
            # If the zoomed image is smaller than the label, we just center it.
            # The key change: We use zoomed_pixmap (already correctly scaled) directly.
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self.setPixmap(zoomed_pixmap)  # <-- Use the centrally scaled image

        self.update()

    def set_image_pixmap(self, pixmap):
        # We need the original, unscaled pixmap size to determine the correct scaling
        self._original_pixmap = pixmap
        # Apply zoom and pan, which now uses the uniform default scale
        self._apply_zoom_and_pan()

    def reset_zoom(self):
        # Note: self.zoom_factor is controlled globally by MPRViewer.reset_all_zooms
        self.zoom_factor = 1.0  # Set local factor to 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        if self._original_pixmap:
            self._apply_zoom_and_pan()