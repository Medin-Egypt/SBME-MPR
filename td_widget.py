from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


class TDWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Reference to the main QMainWindow

        # 3D view modes
        self.surface_mode_enabled = False
        self.planes_mode_enabled = False

        # Create the 3D view layout
        self.create_3d_view()

    def set_data(self, data, affine):
        """Called by main window when new data is loaded."""
        # Placeholder: In a real app, you'd pass this to a 3D rendering library (VTK, VisPy)
        print(f"3D Widget received data with shape: {data.shape}")

    def create_3d_view(self):
        """Create the 3D view area"""
        layout = QVBoxLayout(self)  # Use self as the layout container
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create a single panel for 3D view
        panel = QFrame()
        panel.setObjectName("viewing_panel_3d")
        panel.setFrameStyle(QFrame.Box)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        panel_layout.setSpacing(5)

        # Title bar
        title_bar_widget = QWidget()
        title_bar_layout = QHBoxLayout(title_bar_widget)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_bar_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Color indicator for 3D view (use yellow like oblique)
        color_indicator = QLabel()
        color_indicator.setFixedSize(12, 12)
        color_indicator.setStyleSheet(f"""
            background-color: #FFFF64;
            border-radius: 6px;
            border: 1px solid #E2E8F0;
        """)
        title_bar_layout.addWidget(color_indicator)

        title_lbl = QLabel("3D View")
        title_lbl.setObjectName("view_title_3d")
        title_bar_layout.addWidget(title_lbl)

        panel_layout.addWidget(title_bar_widget)

        # 3D view area (placeholder for now)
        view_area = QLabel()
        view_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view_area.setScaledContents(False)
        view_area.setObjectName("view_3d")
        view_area.setAlignment(Qt.AlignCenter)
        view_area.setText("3D View\n\n[3D visualization will be displayed here]")
        view_area.setStyleSheet("""
            background-color: black;
            color: #4A5568;
            font-size: 18px;
        """)

        panel_layout.addWidget(view_area, stretch=1)
        layout.addWidget(panel)

        self.hide()  # Initially hidden by default

    def toggle_surface_mode(self, checked):
        """Toggle surface rendering mode in 3D view"""
        if checked:
            self.surface_mode_enabled = True
            self.planes_mode_enabled = False
            # Add your surface mode logic here
            print("3D: Surface mode activated")

    def toggle_planes_mode(self, checked):
        """Toggle planes mode in 3D view"""
        if checked:
            self.planes_mode_enabled = True
            self.surface_mode_enabled = False
            # Add your planes mode logic here
            print("3D: Planes mode activated")