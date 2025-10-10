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
    A custom QLabel that handles mouse wheel events to scroll through volume slices.
    It relies on a reference to the main MPRViewer window to access state and update views.
    """
    def __init__(self, parent_viewer, view_type, ui_title):
        super().__init__()
        self.parent_viewer = parent_viewer
        # view_type is the key used in self.slices (e.g., 'axial', 'coronal')
        self.view_type = view_type 
        # ui_title is the display name (e.g., 'Axial', 'Frontal')
        self.ui_title = ui_title 
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

    def wheelEvent(self, event):
        # 1. Check if Slide Mode is active (tool_btn_0_0)
        slide_btn = self.parent_viewer.findChild(QPushButton, "tool_btn_0_0")
        if not slide_btn or not slide_btn.isChecked():
            super().wheelEvent(event)
            return

        # 2. Determine scroll direction (typically 120 or -120 for a full detent)
        # Negative delta (scroll down/back) -> increase slice index (forward)
        # Positive delta (scroll up/forward) -> decrease slice index (backward)
        delta = event.angleDelta().y()
        # Scale the increment/decrement based on scroll intensity, but use 1 for simplicity
        step = 1 if abs(delta) > 0 else 0 
        direction = step * (-1 if delta > 0 else 1)
        
        if direction == 0:
            return

        # 3. Get current slice and max dimension
        current_slice = self.parent_viewer.slices[self.view_type]
        
        # Determine the dimension index based on view type to find max slice count
        if self.view_type == 'axial' or self.view_type == 'oblique':
            max_dim_index = 2 # Z dimension
        elif self.view_type == 'coronal':
            max_dim_index = 1 # Y dimension
        elif self.view_type == 'sagittal':
            max_dim_index = 0 # X dimension
        else:
            return # Should not happen

        max_slice = self.parent_viewer.dims[max_dim_index]

        # 4. Calculate new slice index with wrapping
        new_slice = current_slice + direction
        
        if new_slice >= max_slice:
            new_slice = 0 # Wrap around to start
        elif new_slice < 0:
            new_slice = max_slice - 1 # Wrap around to end
        
        # 5. Update the slice index and redraw the view
        self.parent_viewer.slices[self.view_type] = new_slice
        
        # Call the update function on the main viewer
        self.parent_viewer.update_view(self.ui_title.lower(), self.view_type)


class MPRViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPR VIEWER")
        self.setGeometry(100, 100, 1200, 800)

        # Load data (assuming utils/loader.py and ct.nii.gz are available)
        # self.data, self.affine, self.dims, self.intensity_min, self.intensity_max = loader.load_nifti_data("file.nii.gz")
        self.data, self.affine, self.dims, self.intensity_min, self.intensity_max = loader.load_dicom_data("full")

        # Slice indices for each main view type (coronal is 'frontal' UI view)
        self.slices = {
            'axial': self.dims[2] // 2,
            'coronal': self.dims[1] // 2,
            'sagittal': self.dims[0] // 2,
            'oblique': self.dims[2] // 2
        }

        self.rot_x_deg = 0
        self.rot_y_deg = 0
        self.view_labels = {}
        self.view_panels = {}  # Store panel references
        
        # Flags for view states
        self.main_views_enabled = True
        self.oblique_view_enabled = False
        self.segmentation_view_enabled = False

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
                padding: 2px; /* Adjusted padding for better icon display */
            }
            QPushButton:checked {
                background-color: #a0c8ff;
                border: 2px solid #3366ff;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border: 2px solid #999999;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)

        # Add icons
        self.add_image_to_button("mode_btn_0", "Icons/windows.png", "3 Main Views")
        self.add_image_to_button("mode_btn_1", "Icons/heart.png", "Segmentation View")
        self.add_image_to_button("mode_btn_2", "Icons/diagram.png", "Oblique View")
        self.add_image_to_button("tool_btn_0_0", "Icons/tab.png", "Slide Mode (Scroll Slices)") # Slide Mode
        self.add_image_to_button("tool_btn_0_1", "Icons/brightness.png", "Contrast Mode")
        self.add_image_to_button("tool_btn_0_2", "Icons/loupe.png", "Zoom Mode")
        self.add_image_to_button("tool_btn_1_0", "Icons/expand.png", "Crop Mode")
        self.add_image_to_button("tool_btn_1_1", "Icons/rotating-arrow-to-the-right.png", "Rotate Mode")
        self.add_image_to_button("tool_btn_1_2", "Icons/video.png", "Cine Mode")
        self.add_image_to_button("export_btn_0", "Icons/all.png", "Export All")
        self.add_image_to_button("export_btn_1", "Icons/crop.png", "Crop & Export")

        # Set 3 Main Views button as checked by default
        main_views_btn = self.findChild(QPushButton, "mode_btn_0")
        if main_views_btn:
            main_views_btn.setChecked(True)

        # Show main views by default (hide others)
        self.show_main_views_initially()
        
        # Install event filter to redraw on resize
        self.centralWidget().installEventFilter(self)

    def eventFilter(self, obj, event):
        """Redraw the views on resize to keep the aspect ratio correct."""
        if obj == self.centralWidget() and event.type() == QEvent.Resize:
            if self.main_views_enabled or self.oblique_view_enabled or self.segmentation_view_enabled:
                self.update_visible_views()
        return super().eventFilter(obj, event)

    def numpy_to_qpixmap(self, array_2d: np.ndarray) -> QPixmap:
        """Converts a 2D numpy array (grayscale) into a QPixmap."""
        if array_2d.dtype != np.uint8:
            array_2d = array_2d.astype(np.uint8)
        height, width = array_2d.shape
        bytes_per_line = width
        q_img = QImage(array_2d.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def update_view(self, ui_title: str, view_type: str):
        """
        Updates a specific view panel with a new slice of data.
        
        :param ui_title: The user-facing name of the panel ('frontal', 'sagittal', etc.).
        :param view_type: The internal data type ('coronal', 'sagittal', etc.)
        """
        if ui_title in self.view_labels:
            label = self.view_labels[ui_title]
            
            # Segmentation view is handled separately
            if view_type == 'segmentation':
                self.update_segmentation_view()
                return
            
            slice_data = loader.get_slice_data(
                self.data, self.dims, self.slices, self.affine,
                self.intensity_min, self.intensity_max,
                rot_x_deg=self.rot_x_deg,
                rot_y_deg=self.rot_y_deg,
                view_type=view_type
            )
            pixmap = self.numpy_to_qpixmap(slice_data)
            
            # Scale the pixmap to fit the current label size
            label_size = label.size()
            scaled_pixmap = pixmap.scaled(
                QSize(label_size.width() - 2, label_size.height() - 2),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            # Ensure the custom label's styling is reapplied if it was changed
            label.setStyleSheet(self.view_labels[ui_title].styleSheet())


    def update_segmentation_view(self):
        """
        Placeholder for segmentation visualization.
        """
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
        """Trigger update for currently visible views."""
        if self.main_views_enabled:
            self.update_view('frontal', 'coronal')
            self.update_view('sagittal', 'sagittal')
            self.update_view('axial', 'axial')
        if self.oblique_view_enabled:
            self.update_view('oblique', 'oblique')
        if self.segmentation_view_enabled:
            self.update_view('segmentation', 'segmentation')

    # --- Mode Toggle Functions (Unchanged) ---
    def show_main_views_initially(self):
        """Show main views and hide others on startup"""
        for view_name in ['frontal', 'sagittal', 'axial']:
            if view_name in self.view_panels:
                self.view_panels[view_name].show()
                # 'frontal' UI title maps to 'coronal' view_type
                view_type = 'coronal' if view_name == 'frontal' else view_name
                self.update_view(view_name, view_type)
        
        if 'oblique' in self.view_panels:
            self.view_panels['oblique'].hide()
        if 'segmentation' in self.view_panels:
            self.view_panels['segmentation'].hide()

    def toggle_main_views(self, checked):
        """Toggle the visibility of main views (Frontal, Sagittal, Axial)"""
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
        """Toggle the visibility of oblique view"""
        self.oblique_view_enabled = checked
        if 'oblique' in self.view_panels:
            if checked:
                if self.segmentation_view_enabled:
                    self.segmentation_view_enabled = False
                    if 'segmentation' in self.view_panels:
                        self.view_panels['segmentation'].hide()
                    seg_btn = self.findChild(QPushButton, "mode_btn_1")
                    if seg_btn: seg_btn.setChecked(False)
                
                self.view_panels['oblique'].show()
                self.update_view('oblique', 'oblique')
            else:
                self.view_panels['oblique'].hide()

    def toggle_segmentation_view(self, checked):
        """Toggle the visibility of segmentation view"""
        self.segmentation_view_enabled = checked
        if 'segmentation' in self.view_panels:
            if checked:
                if self.oblique_view_enabled:
                    self.oblique_view_enabled = False
                    if 'oblique' in self.view_panels:
                        self.view_panels['oblique'].hide()
                    obl_btn = self.findChild(QPushButton, "mode_btn_2")
                    if obl_btn: obl_btn.setChecked(False)
                
                self.view_panels['segmentation'].show()
                self.update_segmentation_view()
            else:
                self.view_panels['segmentation'].hide()
    # --- End Mode Toggle Functions ---

    def create_sidebar(self):
        # ... (Sidebar creation code remains mostly the same) ...
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

        # ===== Mode =====
        mode_group = QGroupBox("Mode:")
        mode_layout = QGridLayout()
        mode_layout.setSpacing(5)

        self.mode_group_buttons = QButtonGroup(self)
        self.mode_group_buttons.setExclusive(False)

        for i in range(3):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"mode_btn_{i}")
            btn.setCheckable(True)
            mode_layout.addWidget(btn, 0, i)
            self.mode_group_buttons.addButton(btn, i)
            
            if i == 0:  # 3 Main Views
                btn.clicked.connect(self.toggle_main_views)
            elif i == 1:  # Segmentation View
                btn.clicked.connect(self.toggle_segmentation_view)
            elif i == 2:  # Oblique View
                btn.clicked.connect(self.toggle_oblique_view)

        mode_group.setLayout(mode_layout)
        sidebar_layout.addWidget(mode_group)

        # ===== Tools =====
        tools_group = QGroupBox("Tools:")
        tools_layout = QGridLayout()
        tools_layout.setSpacing(5)
        self.tools_group_buttons = QButtonGroup(self)
        self.tools_group_buttons.setExclusive(True)

        for row in range(2):
            for col in range(3):
                btn = QPushButton()
                btn.setFixedSize(40, 40)
                btn.setObjectName(f"tool_btn_{row}_{col}")
                btn.setCheckable(True)
                tools_layout.addWidget(btn, row, col)
                self.tools_group_buttons.addButton(btn, row * 3 + col)
        tools_group.setLayout(tools_layout)
        sidebar_layout.addWidget(tools_group)

        # ===== Export =====
        export_group = QGroupBox("Export:")
        export_layout = QGridLayout()
        export_layout.setSpacing(5)

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
        sidebar_layout.addWidget(export_group)

        sidebar_layout.addStretch()
        return sidebar

    def toggle_export_button(self, button):
        if button.isChecked():
            # Simply uncheck the button immediately to simulate a transient action
            # since export is not a persistent 'mode'
            button.setChecked(False)
            # You would put the actual export logic here
            print(f"Export functionality triggered for: {button.toolTip()}")

    def create_viewing_area(self):
        viewing_widget = QWidget()
        self.viewing_layout = QGridLayout(viewing_widget)
        self.viewing_layout.setSpacing(10)

        panels = [
            ("Frontal", 'coronal', 0, 0),  # UI Title, Internal View Type, Row, Col
            ("Sagittal", 'sagittal', 0, 1),
            ("Axial", 'axial', 1, 0),
            ("Oblique", 'oblique', 1, 1),
            ("Segmentation", 'segmentation', 1, 1)
        ]

        for title, view_type, row, col in panels:
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
            
            # --- START MODIFICATION ---
            # Use the custom SliceViewLabel for views that support scrolling
            if view_type != 'segmentation':
                view_area = SliceViewLabel(self, view_type, title)
            else:
                # Use a standard QLabel for segmentation placeholder
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
                
            layout.addWidget(view_area)
            # --- END MODIFICATION ---

            # Store panel and label references
            self.view_panels[title.lower()] = panel
            self.view_labels[title.lower()] = view_area
            self.viewing_layout.addWidget(panel, row, col)

        self.viewing_layout.setRowStretch(0, 1)
        self.viewing_layout.setRowStretch(1, 1)
        self.viewing_layout.setColumnStretch(0, 1)
        self.viewing_layout.setColumnStretch(1, 1)

        return viewing_widget

    def add_image_to_button(self, button_name, image_path, tooltip_text=None):
        button = self.findChild(QPushButton, button_name)
        if button:
            try:
                icon = QIcon(QPixmap(image_path))
                if not icon.isNull():
                    button.setIcon(icon)
                    button.setIconSize(QSize(32, 32))
            except Exception:
                # Fallback to text if icon file is missing
                button.setText(button_name.replace('_', ' ').title().split(' ')[0])
            if tooltip_text:
                button.setToolTip(tooltip_text)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = MPRViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
