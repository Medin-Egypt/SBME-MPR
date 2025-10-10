import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QSizePolicy, QButtonGroup
)
from PyQt5.QtCore import Qt, QSize, QEvent
from PyQt5.QtGui import QPixmap, QIcon, QImage
import utils.loader as loader


# --- Custom QLabel for Mouse Wheel Interaction ---
class SliceViewLabel(QLabel):
    """
    Custom QLabel that handles mouse wheel events to scroll through volume slices.
    Relies on parent_viewer for state and update logic.
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

        # state for contrast mode
        self._dragging = False
        self._last_pos = None
        
    def wheelEvent(self, event):
            slide_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_0")
            if slide_btn and slide_btn.isChecked():
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
            else:
                super().wheelEvent(event)

    def mousePressEvent(self, event):
        contrast_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_1")
        if contrast_btn and contrast_btn.isChecked() and event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._last_pos:
            dx = event.x() - self._last_pos.x()
            dy = event.y() - self._last_pos.y()

            # sensitivity factors
            window_change = dx * 2
            level_change = -dy * 2

            window = self.parent_viewer.intensity_max - self.parent_viewer.intensity_min
            level = (self.parent_viewer.intensity_max + self.parent_viewer.intensity_min) / 2

            new_window = max(1, window + window_change)
            new_level = level + level_change

            self.parent_viewer.intensity_min = int(new_level - new_window / 2)
            self.parent_viewer.intensity_max = int(new_level + new_window / 2)

            self._last_pos = event.pos()

            # refresh
            self.parent_viewer.update_view(self.ui_title.lower(), self.view_type)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
        else:
            super().mouseReleaseEvent(event)

class MPRViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPR VIEWER")
        self.setGeometry(100, 100, 1200, 800)

        # Load data (NIfTI by default)
        self.data, self.affine, self.dims, self.intensity_min, self.intensity_max = loader.load_nifti_data("utils/ct.nii.gz")
        # self.data, self.affine, self.dims, self.intensity_min, self.intensity_max = loader.load_dicom_data("full")

        # Slice indices
        self.slices = {
            'axial': self.dims[2] // 2,
            'coronal': self.dims[1] // 2,
            'sagittal': self.dims[0] // 2,
            'oblique': self.dims[2] // 2
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
        self.add_image_to_button("tool_btn_1_2", "Icons/video.png", "Cine Mode")
        self.add_image_to_button("export_btn_0", "Icons/all.png", "Export All")
        self.add_image_to_button("export_btn_1", "Icons/crop.png", "Crop & Export")

        # Default view
        main_views_btn = self.findChild(QPushButton, "mode_btn_0")
        if main_views_btn:
            main_views_btn.setChecked(True)
        self.show_main_views_initially()

        # Redraw on resize
        self.centralWidget().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.centralWidget() and event.type() == QEvent.Resize:
            if self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled:
                self.update_visible_views()
        return super().eventFilter(obj, event)

    def numpy_to_qpixmap(self, array_2d: np.ndarray) -> QPixmap:
        if array_2d.dtype != np.uint8:
            array_2d = array_2d.astype(np.uint8)
        h, w = array_2d.shape
        q_img = QImage(array_2d.tobytes(), w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def update_view(self, ui_title: str, view_type: str):
        if ui_title in self.view_labels:
            label = self.view_labels[ui_title]
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
            scaled = pixmap.scaled(
                QSize(label_size.width() - 2, label_size.height() - 2),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
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
        """)

        layout = QVBoxLayout(sidebar)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)

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
