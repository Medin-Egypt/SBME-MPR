from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt
from utils.renderer_3d import SegmentationViewer3D


class TDWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Reference to the main QMainWindow
        self.viewer_3d = None

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.placeholder = QLabel("3D View\n\nLoad segmentations to view in 3D.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.placeholder.setStyleSheet("""
            background-color: black;
            color: #4A5568;
            font-size: 18px;
        """)
        self.layout.addWidget(self.placeholder)

        # 3D view modes (kept for button connections)
        self.surface_mode_enabled = False
        self.planes_mode_enabled = False

        self.hide()  # Initially hidden by default

    def set_data(self, data, affine):
        """Called by main window when new data is loaded."""
        # This widget primarily displays segmentations, but could be extended
        # to show volume rendering or slices of the main data.
        print(f"3D Widget received data with shape: {data.shape}")

    def set_segmentations(self, file_paths):
        """Creates or updates the 3D segmentation viewer."""
        # Clear current content
        if self.viewer_3d:
            self.layout.removeWidget(self.viewer_3d)
            self.viewer_3d.deleteLater()
            self.viewer_3d = None
        if self.placeholder:
            self.layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None

        if not file_paths:
            self.clear_segmentations()
            return

        try:
            # Create and add the 3D viewer
            self.viewer_3d = SegmentationViewer3D(
                nifti_files=file_paths,
                parent=self
            )
            self.layout.addWidget(self.viewer_3d)
            self.viewer_3d.initialize()
        except Exception as e:
            print(f"Failed to create 3D viewer: {e}")
            self.clear_segmentations()  # Revert to placeholder on error
            # Optionally, show an error message in the placeholder
            if self.placeholder:
                self.placeholder.setText(f"Failed to create 3D view:\n{e}")

    def clear_segmentations(self):
        """Removes the 3D viewer and shows the placeholder."""
        if self.viewer_3d:
            self.layout.removeWidget(self.viewer_3d)
            self.viewer_3d.deleteLater()
            self.viewer_3d = None

        if self.placeholder is None:
            self.placeholder = QLabel("3D View\n\nLoad segmentations to view in 3D.")
            self.placeholder.setAlignment(Qt.AlignCenter)
            self.placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.placeholder.setStyleSheet("""
                background-color: black;
                color: #4A5568;
                font-size: 18px;
            """)
            self.layout.addWidget(self.placeholder)
        self.placeholder.show()

    def toggle_surface_mode(self, checked):
        """Toggle surface rendering mode in 3D view"""
        if checked:
            self.surface_mode_enabled = True
            self.planes_mode_enabled = False
            print("3D: Surface mode is the default for segmentations.")
            # The viewer is already in surface mode. No action needed.

    def toggle_planes_mode(self, checked):
        """Toggle planes mode in 3D view"""
        if checked:
            self.planes_mode_enabled = True
            self.surface_mode_enabled = False
            print("3D: Planes mode is not yet implemented.")
            # Future implementation could add MPR planes to the PyVista plotter.
