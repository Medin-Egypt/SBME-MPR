from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QDialog, QFormLayout,
    QSpinBox, QDialogButtonBox, QPushButton, QLabel, QSizePolicy
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor
import time


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

        form_layout.addRow("Show slices from:", self.start_slice)
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


# --- Custom QLabel for Mouse Wheel Interaction and Crosshairs ---
class SliceViewLabel(QLabel):
    """
    Custom QLabel that handles mouse wheel events to scroll through volume slices,
    contrast adjustment, zoom functionality, cine mode, crop mode, and draws/syncs crosshairs.
    """

    def __init__(self, parent_viewer, view_type, ui_title):
        super().__init__()
        # parent_viewer is the main MPRViewer window
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

        # State for contrast mode
        self._dragging = False
        self._last_pos = None

        # State for zoom mode
        # This will be kept in sync with parent_viewer.mpr_widget.global_zoom_factor
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
        self.oblique_axis_visible = False
        self.oblique_axis_angle = 0
        self.oblique_axis_dragging = False
        self.oblique_axis_start_angle = 0
        self.oblique_axis_drag_start_pos = None

        # Flags for crosshair display control
        self.show_only_center_point = False
        self.hide_crosshair_completely = False  # Hide all crosshair elements

    def _cine_next_slice(self):
        """Advance to the previous slice in cine mode."""
        if not self.cine_active or not self.parent_viewer.file_loaded:
            return

        # Access attributes via mpr_widget
        current_slice = self.parent_viewer.mpr_widget.slices[self.view_type]

        if self.view_type in ('axial', 'oblique'):
            max_dim_index = 2
        elif self.view_type == 'coronal':
            max_dim_index = 1
        elif self.view_type == 'sagittal':
            max_dim_index = 0
        else:
            return

        # Access attributes via mpr_widget
        max_slice = self.parent_viewer.mpr_widget.dims[max_dim_index]
        new_slice = (current_slice - 1) % max_slice

        # Call the central method via mpr_widget
        self.parent_viewer.mpr_widget.set_slice_from_scroll(self.view_type, new_slice)

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

            # Call the central method via mpr_widget
            self.parent_viewer.mpr_widget.change_global_zoom(delta)
            event.accept()

        elif slide_btn and slide_btn.isChecked():
            # file_loaded is still on the main window
            if not self.parent_viewer.file_loaded:
                return

            delta = event.angleDelta().y()
            step = 1 if abs(delta) > 0 else 0
            direction = step * (-1 if delta > 0 else 1)
            if direction == 0:
                return

            # Access attributes via mpr_widget
            current_slice = self.parent_viewer.mpr_widget.slices[self.view_type]

            if self.view_type in ('axial', 'oblique'):
                max_dim_index = 2
            elif self.view_type == 'coronal':
                max_dim_index = 1
            elif self.view_type == 'sagittal':
                max_dim_index = 0
            else:
                return

            # Access attributes via mpr_widget
            max_slice = self.parent_viewer.mpr_widget.dims[max_dim_index]
            new_slice = (current_slice + direction) % max_slice

            # Call the central method via mpr_widget
            self.parent_viewer.mpr_widget.set_slice_from_scroll(self.view_type, new_slice)

            # Access attributes via mpr_widget
            if self.parent_viewer.mpr_widget.segmentation_view_enabled:
                # This attribute needs to be added to mpr_widget
                self.parent_viewer.mpr_widget._last_segmentation_source_view = self.view_type
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Access attributes and methods via mpr_widget
            if self.parent_viewer.mpr_widget.maximized_view is None:
                self.parent_viewer.mpr_widget.maximize_view(self.ui_title.lower())
            else:
                self.parent_viewer.mpr_widget.restore_views()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        crosshair_tool_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_0")
        contrast_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_1")
        zoom_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_2")
        crop_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_0")
        rotate_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_1")
        cine_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_2")

        # Check for oblique axis interaction first (highest priority in rotate mode)
        if (rotate_btn and rotate_btn.isChecked() and
                self.oblique_axis_visible and
                self.view_type == 'coronal' and
                event.button() == Qt.LeftButton):

            # Calculate crosshair screen position
            label_width = self.width()
            label_height = self.height()

            # Access attributes via mpr_widget
            default_scale = self.parent_viewer.mpr_widget.default_scale_factor if hasattr(self.parent_viewer.mpr_widget,
                                                                                          'default_scale_factor') else 1.0
            combined_zoom_factor = default_scale * self.parent_viewer.mpr_widget.global_zoom_factor

            if self._original_pixmap and not self._original_pixmap.isNull():
                original_img_w = self._original_pixmap.width()
                original_img_h = self._original_pixmap.height()

                zoomed_width = int(original_img_w * combined_zoom_factor)
                zoomed_height = int(original_img_h * combined_zoom_factor)

                center_offset_x = (label_width - zoomed_width) / 2
                center_offset_y = (label_height - zoomed_height) / 2

                # Calculate crosshair position (center of oblique axis)
                center_x = int((self.normalized_crosshair_x * zoomed_width) + center_offset_x + self.pan_offset_x)
                center_y = int((self.normalized_crosshair_y * zoomed_height) + center_offset_y + self.pan_offset_y)
            else:
                center_x = self.width() / 2
                center_y = self.height() / 2

            length = min(self.width(), self.height()) * 0.4

            import math
            angle_rad = math.radians(self.oblique_axis_angle)
            handle_x = center_x + length * math.cos(angle_rad)
            handle_y = center_y - length * math.sin(angle_rad)

            # Check distance to handle
            dx = event.x() - handle_x
            dy = event.y() - handle_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < 30:  # Click tolerance
                self.oblique_axis_dragging = True
                self.oblique_axis_drag_start_pos = event.pos()
                self.oblique_axis_start_angle = self.oblique_axis_angle
                event.accept()
                return

        if crosshair_tool_btn and crosshair_tool_btn.isChecked() and event.button() == Qt.LeftButton:
            self._dragging_crosshair = True
            self._update_crosshair(event.pos())
        elif cine_btn and cine_btn.isChecked() and event.button() == Qt.LeftButton:
            if self.cine_active:
                self.stop_cine()
            else:
                self.start_cine()
        elif zoom_btn and zoom_btn.isChecked() and event.button() == Qt.LeftButton:
            # Check the global zoom factor for panning
            # Access attributes via mpr_widget
            if self.parent_viewer.mpr_widget.global_zoom_factor > 1.0:
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

        # Handle oblique axis dragging first
        if self.oblique_axis_dragging:
            # Calculate crosshair screen position as the rotation center
            label_width = self.width()
            label_height = self.height()

            # Access attributes via mpr_widget
            default_scale = self.parent_viewer.mpr_widget.default_scale_factor if hasattr(self.parent_viewer.mpr_widget,
                                                                                          'default_scale_factor') else 1.0
            combined_zoom_factor = default_scale * self.parent_viewer.mpr_widget.global_zoom_factor

            if self._original_pixmap and not self._original_pixmap.isNull():
                original_img_w = self._original_pixmap.width()
                original_img_h = self._original_pixmap.height()

                zoomed_width = int(original_img_w * combined_zoom_factor)
                zoomed_height = int(original_img_h * combined_zoom_factor)

                center_offset_x = (label_width - zoomed_width) / 2
                center_offset_y = (label_height - zoomed_height) / 2

                # Use crosshair as rotation center
                center_x = (self.normalized_crosshair_x * zoomed_width) + center_offset_x + self.pan_offset_x
                center_y = (self.normalized_crosshair_y * zoomed_height) + center_offset_y + self.pan_offset_y
            else:
                center_x = self.width() / 2
                center_y = self.height() / 2

            import math
            # Calculate angle from center to current mouse position
            dx = event.x() - center_x
            dy = center_y - event.y()  # Inverted Y
            angle = math.degrees(math.atan2(dy, dx))

            # Normalize angle to 0-360 range
            if angle < 0:
                angle += 360

            # Update angle
            self.oblique_axis_angle = angle
            # Access attributes via mpr_widget
            self.parent_viewer.mpr_widget.oblique_axis_angle = self.oblique_axis_angle

            # Update oblique view with new rotation
            # Access attributes via mpr_widget
            self.parent_viewer.mpr_widget.rot_y_deg = self.oblique_axis_angle
            self.parent_viewer.mpr_widget.update_view('oblique', 'oblique')

            self.update()
            event.accept()
            return

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

            # intensity_min/max are on the main window
            window = self.parent_viewer.intensity_max - self.parent_viewer.intensity_min
            level = (self.parent_viewer.intensity_max + self.parent_viewer.intensity_min) / 2
            new_window = max(1, window + window_change)
            new_level = level + level_change
            self.parent_viewer.intensity_min = int(new_level - new_window / 2)
            self.parent_viewer.intensity_max = int(new_level + new_window / 2)
            self._last_pos = event.pos()

            # Call the central method via mpr_widget
            self.parent_viewer.mpr_widget.update_all_views()

        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        crop_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_0")

        if self.oblique_axis_dragging and event.button() == Qt.LeftButton:
            self.oblique_axis_dragging = False
            event.accept()
            return

        if self._dragging_crosshair and event.button() == Qt.LeftButton:
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
        # Access attributes via mpr_widget
        default_scale = self.parent_viewer.mpr_widget.default_scale_factor if hasattr(self.parent_viewer.mpr_widget,
                                                                                      'default_scale_factor') else 1.0
        self.zoom_factor = self.parent_viewer.mpr_widget.global_zoom_factor  # Sync local factor
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

        # Call the central method via mpr_widget
        self.parent_viewer.mpr_widget.set_slice_from_crosshair(self.view_type, norm_x, norm_y)

        # Access attributes via mpr_widget
        if self.parent_viewer.mpr_widget.segmentation_view_enabled:
            # This attribute needs to be added to mpr_widget
            self.parent_viewer.mpr_widget._last_segmentation_source_view = self.view_type

        self.update()

    def set_normalized_crosshair(self, norm_x, norm_y):
        self.normalized_crosshair_x = norm_x
        self.normalized_crosshair_y = norm_y
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        crop_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_0")
        rotate_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_1_1")

        # file_loaded is still on the main window
        if self.parent_viewer.file_loaded and self._original_pixmap and not self._original_pixmap.isNull():
            # Skip all crosshair drawing if hide_crosshair_completely is True
            if self.hide_crosshair_completely:
                return

            painter = QPainter(self)

            label_width = self.width()
            label_height = self.height()

            # Recalculate zoomed size for crosshair positioning (must match _apply_zoom_and_pan)
            # Access attributes via mpr_widget
            default_scale = self.parent_viewer.mpr_widget.default_scale_factor if hasattr(self.parent_viewer.mpr_widget,
                                                                                          'default_scale_factor') else 1.0
            self.zoom_factor = self.parent_viewer.mpr_widget.global_zoom_factor
            combined_zoom_factor = default_scale * self.zoom_factor

            original_img_w = self._original_pixmap.width()
            original_img_h = self._original_pixmap.height()

            zoomed_width = int(original_img_w * combined_zoom_factor)
            zoomed_height = int(original_img_h * combined_zoom_factor)

            center_offset_x = (label_width - zoomed_width) / 2
            center_offset_y = (label_height - zoomed_height) / 2

            # Calculate screen coordinates for crosshair based on normalized position, zoom, and pan
            draw_x = int(
                (self.normalized_crosshair_x * zoomed_width) + center_offset_x + self.pan_offset_x)
            draw_y = int(
                (self.normalized_crosshair_y * zoomed_height) + center_offset_y + self.pan_offset_y)

            # Draw crosshair lines only if not in "center point only" mode
            if not self.show_only_center_point:
                # Access attributes via mpr_widget
                colors = self.parent_viewer.mpr_widget.view_colors
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

            # Always draw the center point marker
            if 0 <= draw_x <= self.width() and 0 <= draw_y <= self.height():
                intersect_pen = QPen(QColor(255, 255, 0))
                intersect_pen.setWidth(2)
                painter.setPen(intersect_pen)
                painter.drawEllipse(draw_x - 4, draw_y - 4, 8, 8)

            # Draw oblique axis if visible and in oblique view mode
            # Access attributes via mpr_widget
            if (self.oblique_axis_visible and
                    self.view_type == 'coronal' and
                    self.parent_viewer.mpr_widget.oblique_view_enabled):
                import math

                # Use crosshair position as the center point for the oblique axis
                center_x = draw_x  # Use crosshair X position
                center_y = draw_y  # Use crosshair Y position
                length = min(self.width(), self.height()) * 0.4  # 40% of smaller dimension

                angle_rad = math.radians(self.oblique_axis_angle)
                end_x = center_x + length * math.cos(angle_rad)
                end_y = center_y - length * math.sin(angle_rad)  # Negative because Y increases downward

                # Draw yellow axis line
                axis_pen = QPen(QColor(255, 255, 100), 3)
                painter.setPen(axis_pen)
                painter.drawLine(int(center_x), int(center_y), int(end_x), int(end_y))

                # Draw draggable handle at the end
                painter.setBrush(QColor(255, 255, 100))
                painter.setPen(QPen(QColor(200, 200, 0), 2))
                painter.drawEllipse(int(end_x - 8), int(end_y - 8), 16, 16)

                # Draw angle annotation above the line
                annotation_text = f"{self.oblique_axis_angle:.1f}°"
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(painter.font())

                # Position text slightly above and to the right of center
                text_x = int(center_x + 20)
                text_y = int(center_y - 20)
                painter.drawText(text_x, text_y, annotation_text)

            painter.end()

    def _apply_zoom_and_pan(self):
        # Sync local zoom factor from viewer's global factor
        # Access attributes via mpr_widget
        self.zoom_factor = self.parent_viewer.mpr_widget.global_zoom_factor

        if self._original_pixmap is None or self._original_pixmap.isNull():
            return

        label_size = self.contentsRect().size()
        if label_size.width() < 10 or label_size.height() < 10:
            return

        # CHANGE 1: Use a combined scale factor (Uniform Scale + User Zoom)
        # Access attributes via mpr_widget
        default_scale = self.parent_viewer.mpr_widget.default_scale_factor if hasattr(self.parent_viewer.mpr_widget,
                                                                                      'default_scale_factor') else 1.0

        # The image size (base scaled by default_scale) is multiplied by the user's zoom.
        original_img_w = self._original_pixmap.width()
        original_img_h = self._original_pixmap.height()

        # 1. Calculate the BASE size (uniform scale)
        base_w = int(original_img_w * default_scale)
        base_h = int(original_img_h * default_scale)

        # 2. Apply the user zoom (self.zoom_factor) to the BASE size
        # Access attributes via mpr_widget
        zoomed_width = int(base_w * self.parent_viewer.mpr_widget.global_zoom_factor)
        zoomed_height = int(base_h * self.parent_viewer.mpr_widget.global_zoom_factor)

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
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self.setPixmap(zoomed_pixmap)  # <-- Use the centrally scaled image

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