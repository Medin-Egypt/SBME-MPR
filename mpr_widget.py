import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QSizePolicy, QComboBox, QApplication
)
from PyQt5.QtCore import Qt, QSize, QEvent, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen
import utils.loader as loader
from utils.ui_classes import SliceViewLabel


class MPRWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Reference to the main QMainWindow

        self.view_colors = {
            'axial': QColor(100, 220, 100),  # Green
            'coronal': QColor(100, 150, 255),  # Blue
            'sagittal': QColor(255, 100, 100),  # Red
            'oblique': QColor(255, 255, 100),  # Yellow
        }

        # Local copies of dims/affine for quick access
        self.dims = None
        self.affine = None

        self.pixel_dims = {'axial': (0, 0), 'coronal': (0, 0), 'sagittal': (0, 0)}

        # Crop bounds (normalized 0-1 coordinates)
        self.segmentation_visible = False  # Whether to show segmentation overlays
        self.segmentation_view_selector = None  # Will hold the QComboBox
        self.current_segmentation_source = 'axial'  # Default view to show
        self._last_segmentation_source_view = 'axial'

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

        # NEW properties for coordinated scaling/zooming
        self.global_zoom_factor = 1.0
        self.default_scale_factor = 1.0
        # Oblique axis properties
        self.oblique_axis_visible = False
        self.oblique_axis_angle = 0  # Default angle in degrees
        self.oblique_axis_dragging = False
        self.oblique_axis_handle_size = 10  # Size of draggable handle

        # Create the viewing area layout
        self.create_viewing_area()

        # Install event filter on self to catch resize events
        self.installEventFilter(self)

        # Set initial view
        self.show_main_views_initially()

    def set_data(self, data, affine, dims, intensity_min, intensity_max):
        """Called by main window when new data is loaded."""
        # We access data directly from main_window, but store local copies of metadata
        self.affine = affine
        self.dims = dims

        self._calculate_pixel_dims()
        self.reset_crosshair_and_slices()
        self.reset_all_zooms()
        self.reset_rotation()

        self.show_main_views_initially()
        self.update_all_views()

    def update_data(self, data, dims, segmentation_data_list):
        """Called by main window when data is modified (e.g., cropped)."""
        self.dims = dims

        # Recalculate dimensions and reset views
        self._calculate_pixel_dims()
        self.reset_crosshair_and_slices()
        self.update_all_views()

    def set_segmentation_visibility(self, visible):
        self.segmentation_visible = visible

    def _calculate_pixel_dims(self):
        """
        Calculates the aspect-ratio-corrected pixel dimensions for axial,
        coronal, and sagittal views based on voxel spacing.
        This should be called once after a file is loaded.
        """
        if self.dims is None or self.affine is None:
            self.pixel_dims = {'axial': (0, 0), 'coronal': (0, 0), 'sagittal': (0, 0)}
            return

        x_spacing = self.affine[0, 0]
        y_spacing = self.affine[1, 1]
        z_spacing = self.affine[2, 2]
        sag_vox, cor_vox, ax_vox = self.dims[0], self.dims[1], self.dims[2]

        ax_w = sag_vox
        ax_h = int(cor_vox * (y_spacing / x_spacing)) if x_spacing > 0 else cor_vox
        self.pixel_dims['axial'] = (ax_w, ax_h)

        cor_w = sag_vox
        cor_h = int(ax_vox * (z_spacing / x_spacing)) if x_spacing > 0 else ax_vox
        self.pixel_dims['coronal'] = (cor_w, cor_h)

        sag_w = cor_vox
        sag_h = int(ax_vox * (z_spacing / y_spacing)) if y_spacing > 0 else ax_vox
        self.pixel_dims['sagittal'] = (sag_w, sag_h)

    def create_viewing_area(self):
        """Creates the 2x2 grid layout and panels."""
        self.viewing_grid = QGridLayout(self)  # Use self as the layout container
        self.viewing_grid.setSpacing(10)
        self.viewing_grid.setContentsMargins(0, 0, 0, 0)

        panels = [
            ("Coronal", 'coronal', 0, 0), ("Sagittal", 'sagittal', 0, 1),
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
            if title.lower() == 'coronal':
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

            # Add dropdown for segmentation view
            if title.lower() == 'segmentation':
                title_bar_layout.addStretch()

                self.segmentation_view_selector = QComboBox()
                self.segmentation_view_selector.setObjectName("segmentation_view_dropdown")
                self.segmentation_view_selector.addItems(["Axial", "Coronal", "Sagittal"])
                self.segmentation_view_selector.setCurrentText("Axial")
                self.segmentation_view_selector.setFixedWidth(100)
                self.segmentation_view_selector.currentTextChanged.connect(self.on_segmentation_view_changed)
                title_bar_layout.addWidget(self.segmentation_view_selector)

            panel_layout.addWidget(title_bar_widget)

            if view_type != 'segmentation':
                view_area = SliceViewLabel(self.main_window, view_type, title)
            else:
                view_area = QLabel()
                view_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                view_area.setScaledContents(False)
                view_area.setObjectName(f"view_{title.lower()}")
                view_area.setAlignment(Qt.AlignCenter)
                view_area.setText("Segmentation View")  # Placeholder

            panel_layout.addWidget(view_area, stretch=1)

            self.view_panels[title.lower()] = panel
            self.view_labels[title.lower()] = view_area
            self.viewing_grid.addWidget(panel, row, col)

        self.viewing_grid.setRowStretch(0, 1)
        self.viewing_grid.setRowStretch(1, 1)
        self.viewing_grid.setColumnStretch(0, 1)
        self.viewing_grid.setColumnStretch(1, 1)

        # Do not return widget, as 'self' is the widget

    # --- Coordinated Zoom Logic ---

    def change_global_zoom(self, delta):
        """Updates the global zoom factor and applies it to all views."""
        if not self.main_window.file_loaded:
            return

        zoom_step = 1.15

        if delta > 0:
            new_zoom = self.global_zoom_factor * zoom_step
            self.global_zoom_factor = min(new_zoom, 10.0)
        else:
            new_zoom = self.global_zoom_factor / zoom_step
            self.global_zoom_factor = max(new_zoom, 1.0)

        for label in self.view_labels.values():
            if isinstance(label, SliceViewLabel):
                label.zoom_factor = self.global_zoom_factor
                if self.global_zoom_factor == 1.0:
                    label.pan_offset_x = 0
                    label.pan_offset_y = 0
                label._apply_zoom_and_pan()

    # --- Reset logic methods ---

    def reset_all_zooms(self):
        """Resets the global zoom factor and all view-specific pan offsets."""
        self.global_zoom_factor = 1.0
        for label in self.view_labels.values():
            if isinstance(label, SliceViewLabel):
                label.zoom_factor = 1.0
                label.pan_offset_x = 0
                label.pan_offset_y = 0
        self.update_all_views()

    def reset_rotation(self):
        """Resets the oblique rotation angles to default."""
        self.rot_x_deg = 0
        self.rot_y_deg = 0
        self.oblique_axis_angle = 0

    def reset_crosshair_and_slices(self):
        """Resets crosshairs to the center and slices to the middle."""
        self.norm_coords = {'S': 0.5, 'C': 0.5, 'A': 0.5}
        if self.dims:
            self.slices['axial'] = self.dims[2] // 2
            self.slices['coronal'] = self.dims[1] // 2
            self.slices['sagittal'] = self.dims[0] // 2
            self.slices['oblique'] = self.dims[2] // 2

    # --- Slice/View Update Logic ---

    def set_slice_from_crosshair(self, source_view, norm_x, norm_y):
        """Updates slice indices based on the normalized crosshair position from a source view."""
        if not self.main_window.file_loaded or self.dims is None:
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

    def set_slice_from_scroll(self, view_type, new_slice_index):
        """
        Updates a slice index from scrolling or cine mode, recalculates the
        corresponding normalized coordinate for the crosshair, and updates all views.
        """
        if not self.main_window.file_loaded or self.dims is None:
            return

        self.slices[view_type] = new_slice_index

        if view_type == 'oblique':
            self.update_view('oblique', 'oblique')
            return

        if view_type == 'axial':
            if self.dims[2] > 1:
                self.norm_coords['A'] = 1.0 - (new_slice_index / (self.dims[2] - 1))
        elif view_type == 'coronal':
            if self.dims[1] > 1:
                self.norm_coords['C'] = new_slice_index / (self.dims[1] - 1)
        elif view_type == 'sagittal':
            if self.dims[0] > 1:
                self.norm_coords['S'] = new_slice_index / (self.dims[0] - 1)

        self.update_all_views()

    def calculate_and_set_uniform_default_scale(self):
        """Calculates the minimum non-distorting scale factor across all visible views."""
        if not self.main_window.file_loaded or not self.dims:
            self.default_scale_factor = 1.0
            return

        min_scale = float('inf')

        views_to_check = []
        if self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled:
            views_to_check.extend(['coronal', 'sagittal', 'axial'])
        if self.oblique_view_enabled:
            views_to_check.append('oblique')

        for view_name in views_to_check:
            label = self.view_labels.get(view_name)
            if not label or not label.isVisible() or label.width() < 10 or label.height() < 10:
                continue

            if view_name in self.pixel_dims:
                img_w, img_h = self.pixel_dims[view_name]
            else:
                img_w, img_h = self.pixel_dims['axial']

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
                ('coronal', 'coronal'), ('sagittal', 'sagittal'), ('axial', 'axial')
            ])
        if self.oblique_view_enabled:
            views_to_update.append(('oblique', 'oblique'))
        if self.segmentation_view_enabled:
            views_to_update.append(('segmentation', 'segmentation'))

        self.calculate_and_set_uniform_default_scale()

        for ui_title, view_type in views_to_update:
            self.update_view(ui_title, view_type, sync_crosshair=True)

        # Notify 3D widget of slice changes for plane updates
        # Only update if the 3D widget is visible to prevent rendering conflicts
        if hasattr(self.main_window, 'td_widget') and self.main_window.td_widget.isVisible():
            self.main_window.td_widget.update_slice_positions(self.slices)

    def update_view(self, ui_title: str, view_type: str, sync_crosshair=False):
        if ui_title not in self.view_labels:
            return
        label = self.view_labels[ui_title]
        if not label.isVisible():
            return
        if not self.main_window.file_loaded or self.main_window.data is None:
            return

        if view_type == 'segmentation':
            self.update_segmentation_view()
            return

        slice_data = loader.get_slice_data(
            self.main_window.data, self.dims, self.slices, self.affine,
            self.main_window.intensity_min, self.main_window.intensity_max,
            rot_x_deg=self.rot_x_deg, rot_y_deg=self.rot_y_deg,
            view_type=view_type,
            norm_coords=self.norm_coords
        )

        pixmap = self.numpy_to_qpixmap(slice_data)

        if self.segmentation_visible and self.main_window.segmentation_data_list and view_type != 'segmentation':
            pixmap = self.add_segmentation_overlay(pixmap, view_type)

        if isinstance(label, SliceViewLabel):
            label.zoom_factor = self.global_zoom_factor
            label.set_image_pixmap(pixmap)

            if sync_crosshair:
                if view_type == 'axial':
                    norm_x, norm_y = self.norm_coords['S'], self.norm_coords['C']
                elif view_type == 'coronal':
                    norm_x, norm_y = self.norm_coords['S'], self.norm_coords['A']
                elif view_type == 'sagittal':
                    norm_x, norm_y = self.norm_coords['C'], self.norm_coords['A']
                elif view_type == 'oblique':
                    norm_x, norm_y = 0.5, 0.5
                else:
                    norm_x, norm_y = 0.5, 0.5

                label.set_normalized_crosshair(norm_x, norm_y)

                if view_type == 'oblique':
                    label.show_only_center_point = False
                    label.hide_crosshair_completely = True
                else:
                    label.show_only_center_point = False
                    label.hide_crosshair_completely = False

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

    def update_segmentation_view(self):
        if 'segmentation' not in self.view_labels:
            return
        label = self.view_labels['segmentation']

        if not self.main_window.segmentation_data_list:
            label.setText("Segmentation View\n\n[Load segmentation data]")
            return

        source_view = self.current_segmentation_source
        slice_idx = self.slices.get(source_view, self.slices['axial'])
        view_type = source_view

        if view_type in self.pixel_dims:
            correct_width, correct_height = self.pixel_dims[view_type]
        else:
            correct_width, correct_height = self.pixel_dims['axial']

        if correct_width == 0 or correct_height == 0:
            return  # Not initialized yet

        seg_image = QImage(correct_width, correct_height, QImage.Format_RGB32)
        seg_image.fill(QColor(0, 0, 0))
        painter = QPainter(seg_image)

        for seg_data in self.main_window.segmentation_data_list:
            if view_type == 'axial':
                seg_slice = seg_data[:, :, slice_idx]
                seg_slice = np.flipud(np.rot90(seg_slice))
            elif view_type == 'coronal':
                seg_slice = seg_data[:, slice_idx, :]
                seg_slice = np.rot90(seg_slice)
            elif view_type == 'sagittal':
                seg_slice = seg_data[slice_idx, :, :]
                seg_slice = np.rot90(seg_slice)

            from scipy import ndimage
            mask = seg_slice > 0.5
            if not mask.any():
                continue

            eroded = ndimage.binary_erosion(mask)
            edges = mask & ~eroded

            scale_y = correct_height / edges.shape[0]
            scale_x = correct_width / edges.shape[1]

            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)

            edge_coords = np.argwhere(edges)
            for y, x in edge_coords:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                painter.drawPoint(scaled_x, scaled_y)

        painter.end()
        pixmap = QPixmap.fromImage(seg_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def on_segmentation_view_changed(self, view_name):
        """Callback when the segmentation view dropdown changes."""
        self.current_segmentation_source = view_name.lower()
        if self.segmentation_view_enabled:
            self.update_segmentation_view()

    def handle_cine_button_toggle(self, checked):
        if not checked:
            for label in self.view_labels.values():
                if isinstance(label, SliceViewLabel):
                    label.stop_cine()

    def handle_rotate_mode_toggle(self, checked):
        """Show/hide oblique axis based on rotate mode and current view"""
        if self.oblique_view_enabled:
            self.update_view('coronal', 'coronal', sync_crosshair=True)

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

    def update_visible_views(self):
        self.calculate_and_set_uniform_default_scale()
        visible_views = [name for name, panel in self.view_panels.items() if panel.isVisible()]
        for view_name in visible_views:
            self.update_view(view_name, view_name, sync_crosshair=True)

    def numpy_to_qpixmap(self, array_2d: np.ndarray) -> QPixmap:
        if array_2d.dtype != np.uint8:
            array_2d = array_2d.astype(np.uint8)
        h, w = array_2d.shape
        q_img = QImage(array_2d.tobytes(), w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def add_segmentation_overlay(self, base_pixmap, view_type):
        """Adds red outline overlay from segmentation data to the pixmap."""
        image = base_pixmap.toImage()
        painter = QPainter(image)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)

        if view_type == 'axial':
            slice_idx = self.slices['axial']
        elif view_type == 'coronal':
            slice_idx = self.slices['coronal']
        elif view_type == 'sagittal':
            slice_idx = self.slices['sagittal']
        else:
            painter.end()
            return base_pixmap

        for seg_data in self.main_window.segmentation_data_list:
            if view_type == 'axial':
                seg_slice = seg_data[:, :, slice_idx]
                seg_slice = np.flipud(np.rot90(seg_slice))
            elif view_type == 'coronal':
                seg_slice = seg_data[:, slice_idx, :]
                seg_slice = np.rot90(seg_slice)
            elif view_type == 'sagittal':
                seg_slice = seg_data[slice_idx, :, :]
                seg_slice = np.rot90(seg_slice)

            from scipy import ndimage
            mask = seg_slice > 0.5
            if not mask.any():
                continue

            eroded = ndimage.binary_erosion(mask)
            edges = mask & ~eroded

            scale_y = base_pixmap.height() / edges.shape[0]
            scale_x = base_pixmap.width() / edges.shape[1]

            edge_coords = np.argwhere(edges)
            for y, x in edge_coords:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                painter.drawPoint(scaled_x, scaled_y)

        painter.end()
        return QPixmap.fromImage(image)

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
            ("Coronal", 'coronal', 0, 0), ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0), ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1)
        ]

        for title, view_type, row, col in panels:
            panel = self.view_panels[title.lower()]
            if self.viewing_grid.indexOf(panel) == -1:
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
        self.restore_views()
        for view_name, panel in self.view_panels.items():
            if view_name in ['coronal', 'sagittal', 'axial']:
                panel.show()
            else:
                panel.hide()

    def toggle_main_views(self, checked):
        if not checked:
            return
        self.restore_views()
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False
        self.segmentation_visible = True if self.main_window.segmentation_data_list else False

        self.main_window.findChild(QPushButton, "mpr_mode_btn_2").setChecked(False)
        self.main_window.findChild(QPushButton, "mpr_mode_btn_1").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['coronal', 'sagittal', 'axial']:
                panel.show()
            else:
                panel.hide()
        self.update_visible_views()

    def toggle_oblique_view(self, checked):
        if not checked and self.oblique_view_enabled:
            self.toggle_main_views(True)
            self.main_window.findChild(QPushButton, "mpr_mode_btn_0").setChecked(True)
            return
        self.restore_views()
        self.oblique_view_enabled = True
        self.main_views_enabled = False
        self.segmentation_view_enabled = False
        self.segmentation_visible = True if self.main_window.segmentation_data_list else False
        self.oblique_axis_visible = True

        self.main_window.findChild(QPushButton, "mpr_mode_btn_0").setChecked(False)
        self.main_window.findChild(QPushButton, "mpr_mode_btn_1").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['coronal', 'sagittal', 'axial', 'oblique']:
                panel.show()
                if view_name == 'oblique':
                    self.viewing_grid.removeWidget(panel)
                    self.viewing_grid.addWidget(panel, 1, 1)
            else:
                panel.hide()
        self.update_visible_views()

    def toggle_segmentation_view(self, checked):
        if not checked and self.segmentation_view_enabled:
            self.toggle_main_views(True)
            self.main_window.findChild(QPushButton, "mpr_mode_btn_0").setChecked(True)
            return
        self.restore_views()
        self.segmentation_view_enabled = True
        self.main_views_enabled = False
        self.oblique_view_enabled = False

        self.main_window.findChild(QPushButton, "mpr_mode_btn_0").setChecked(False)
        self.main_window.findChild(QPushButton, "mpr_mode_btn_2").setChecked(False)

        for view_name, panel in self.view_panels.items():
            if view_name in ['coronal', 'sagittal', 'axial', 'segmentation']:
                panel.show()
                if view_name == 'segmentation':
                    self.viewing_grid.removeWidget(panel)
                    self.viewing_grid.addWidget(panel, 1, 1)
            else:
                panel.hide()
        self.update_visible_views()