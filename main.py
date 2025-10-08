import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon, QImage
import utils.loader as loader


# --- MPR VIEWER CLASS ---
class MPRViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPR VIEWER")
        self.setGeometry(100, 100, 1200, 800)

        # Load data using the integrated function
        self.data, self.affine, self.dims, self.intensity_min, self.intensity_max = loader.load_nifti_data("file.nii.gz")

        # Initialize slices to the center of each dimension
        self.slices = {
            'axial': self.dims[2] // 2,
            'coronal': self.dims[1] // 2,
            'sagittal': self.dims[0] // 2,
            'oblique': self.dims[2] // 2
        }

        # Oblique rotation defaults
        self.rot_x_deg = 0
        self.rot_y_deg = 0

        # Store QLabel references
        self.view_labels = {}

        # Main layout setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Sidebar + Viewing Area
        sidebar = self.create_sidebar()
        viewing_area = self.create_viewing_area()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(viewing_area)

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

        # Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
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
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border: 2px solid #999999;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)

        # Add icons to buttons
        self.add_image_to_button("mode_btn_0", "Icons/windows.png")
        self.add_image_to_button("mode_btn_1", "Icons/heart.png")
        self.add_image_to_button("mode_btn_2", "Icons/diagram.png")
        self.add_image_to_button("tool_btn_0_0", "icons/tool1.png")
        self.add_image_to_button("tool_btn_0_1", "Icons/brightness.png")
        self.add_image_to_button("tool_btn_0_2", "Icons/loupe.png")
        self.add_image_to_button("tool_btn_1_0", "Icons/expand.png")
        self.add_image_to_button("tool_btn_1_1", "Icons/rotating-arrow-to-the-right.png")
        self.add_image_to_button("tool_btn_1_2", "Icons/video.png")
        self.add_image_to_button("tool_btn_2_0", "icons/tool7.png")
        self.add_image_to_button("tool_btn_2_1", "icons/tool8.png")
        self.add_image_to_button("tool_btn_2_2", "icons/tool9.png")
        self.add_image_to_button("export_btn_0", "Icons/all.png")
        self.add_image_to_button("export_btn_1", "Icons/crop.png")

        # Initial render
        self.update_all_views()

        # Re-render when resized
        self.centralWidget().installEventFilter(self)

    def eventFilter(self, obj, event):
        """Handle resize to update images with proper scaling."""
        if obj == self.centralWidget() and event.type() == event.Resize:
            self.update_all_views()
        return super().eventFilter(obj, event)

    def numpy_to_qpixmap(self, array_2d: np.ndarray) -> QPixmap:
        """Convert a 2D numpy array (uint8) to QPixmap."""
        if array_2d.dtype != np.uint8:
            array_2d = array_2d.astype(np.uint8)

        height, width = array_2d.shape
        bytes_per_line = width
        q_img = QImage(array_2d.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def update_view(self, ui_title: str, view_type: str):
        """Update a single view (axial, sagittal, coronal, oblique)."""
        if ui_title.lower() in self.view_labels:
            slice_data = loader.get_slice_data(
                self.data, self.dims, self.slices,
                self.intensity_min, self.intensity_max,
                rot_x_deg=self.rot_x_deg,
                rot_y_deg=self.rot_y_deg,
                view_type=view_type
            )

            pixmap = self.numpy_to_qpixmap(slice_data)
            label = self.view_labels[ui_title.lower()]

            # ✅ Scale to label size while keeping aspect ratio
            label_size = label.size()
            scaled_pixmap = pixmap.scaled(
                QSize(label_size.width() - 2, label_size.height() - 2),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setText("")

    def update_all_views(self):
        """Update all four MPR views."""
        self.update_view('frontal', 'coronal')
        self.update_view('sagittal', 'sagittal')
        self.update_view('axial', 'axial')
        self.update_view('oblique', 'oblique')

    def create_sidebar(self):
        """Create the left sidebar with Mode, Tools, and Export sections."""
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.Box)
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #e0e0e0;
                border: 2px solid #888888;
            }
            QPushButton[objectName^="export_btn_"] {
                border: 3px solid #ff0000;
            }
            QPushButton[objectName^="export_btn_"]:hover {
                background-color: #ffe0e0;
                border: 3px solid #cc0000;
            }
            QPushButton[objectName^="export_btn_"]:pressed {
                background-color: #ffc0c0;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(20)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        # Mode section
        mode_group = QGroupBox("Mode:")
        mode_layout = QGridLayout()
        mode_layout.setSpacing(5)
        for i in range(3):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"mode_btn_{i}")
            mode_layout.addWidget(btn, 0, i)
        mode_group.setLayout(mode_layout)
        sidebar_layout.addWidget(mode_group)

        # Tools section
        tools_group = QGroupBox("Tools:")
        tools_layout = QGridLayout()
        tools_layout.setSpacing(5)
        for row in range(3):
            for col in range(3):
                btn = QPushButton()
                btn.setFixedSize(40, 40)
                btn.setObjectName(f"tool_btn_{row}_{col}")
                tools_layout.addWidget(btn, row, col)
        tools_group.setLayout(tools_layout)
        sidebar_layout.addWidget(tools_group)

        # Export section
        export_group = QGroupBox("Export:")
        export_layout = QGridLayout()
        export_layout.setSpacing(5)
        for i in range(2):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"export_btn_{i}")
            export_layout.addWidget(btn, 0, i)
        export_group.setLayout(export_layout)
        sidebar_layout.addWidget(export_group)
        sidebar_layout.addStretch()

        return sidebar

    def create_viewing_area(self):
        viewing_widget = QWidget()
        viewing_layout = QGridLayout(viewing_widget)
        viewing_layout.setSpacing(10)

        panels = [
            ("Frontal", 'coronal', 0, 0),
            ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0),
            ("Oblique", 'oblique', 1, 1)
        ]

        for title, view_type, row, col in panels:
            panel = self.create_viewing_panel(title)
            viewing_layout.addWidget(panel, row, col)

            label = panel.findChild(QLabel, f"view_{title.lower()}")
            if label:
                self.view_labels[title.lower()] = label

        viewing_layout.setRowStretch(0, 1)
        viewing_layout.setRowStretch(1, 1)
        viewing_layout.setColumnStretch(0, 1)
        viewing_layout.setColumnStretch(1, 1)

        return viewing_widget

    def create_viewing_panel(self, title):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        panel.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 2px solid #555555;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                background-color: transparent;
                padding: 5px;
            }
        """)
        layout.addWidget(title_label)

        view_area = QLabel()
        view_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view_area.setScaledContents(False)  # ✅ DO NOT STRETCH IMAGE
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
        view_area.setText(f"Loading {title} View...")
        layout.addWidget(view_area)

        return panel

    def add_image_to_button(self, button_name, image_path):
        """Try to add icon to button; fallback to text."""
        button = self.findChild(QPushButton, button_name)
        if button:
            try:
                icon = QIcon(QPixmap(image_path))
                if not icon.isNull():
                    button.setIcon(icon)
                    button.setIconSize(QSize(32, 32))
            except Exception:
                button.setText(button_name.replace('_', ' ').title().split(' ')[0])


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = MPRViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
