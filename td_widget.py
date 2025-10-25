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

        # Volume data for plane rendering
        self.volume_data = None
        self.affine = None
        self.dims = None
        self.intensity_min = 0
        self.intensity_max = 255

        self.hide()  # Initially hidden by default

    def set_data(self, data, affine, dims=None, intensity_min=0, intensity_max=255):
        """Called by main window when new data is loaded."""
        # Store data for plane rendering
        self.volume_data = data
        self.affine = affine
        self.dims = dims
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        print(f"3D Widget received data with shape: {data.shape}")

        # If viewer exists, update its volume data
        if self.viewer_3d is not None:
            self.viewer_3d.volume_data = data
            self.viewer_3d.affine = affine
            self.viewer_3d.dims = dims
            self.viewer_3d.intensity_min = intensity_min
            self.viewer_3d.intensity_max = intensity_max

            # Update slider ranges for new dimensions
            self.viewer_3d.current_slices = {
                'axial': dims[2] // 2 if dims else 0,
                'sagittal': dims[0] // 2 if dims else 0,
                'coronal': dims[1] // 2 if dims else 0
            }
            self.viewer_3d.update_slider_ranges()

            # If in planes mode, recreate planes with new data
            if self.planes_mode_enabled:
                self.viewer_3d.create_planes()
        else:
            # If no viewer exists and we have volume data, create a basic viewer
            # This allows planes mode without segmentations
            self._create_viewer_if_needed()

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
            # Create and add the 3D viewer with volume data
            self.viewer_3d = SegmentationViewer3D(
                nifti_files=file_paths,
                parent=self,
                volume_data=self.volume_data,
                affine=self.affine,
                dims=self.dims,
                intensity_min=self.intensity_min,
                intensity_max=self.intensity_max
            )
            self.layout.addWidget(self.viewer_3d)

            # Connect the finished signal to notify main window
            self.viewer_3d.loading_finished.connect(self._on_3d_loading_finished)

            self.viewer_3d.initialize()
            # Update slider ranges after initialization
            if self.dims:
                self.viewer_3d.update_slider_ranges()
        except Exception as e:
            print(f"Failed to create 3D viewer: {e}")
            self.clear_segmentations()  # Revert to placeholder on error
            # Optionally, show an error message in the placeholder
            if self.placeholder:
                self.placeholder.setText(f"Failed to create 3D view:\n{e}")

    def _on_3d_loading_finished(self):
        """Called when 3D mesh loading completes."""
        # Notify main window that loading is complete
        if self.main_window and hasattr(self.main_window, '_on_segmentation_loading_complete'):
            self.main_window._on_segmentation_loading_complete()

    def set_segmentations_with_merge(self, file_paths, seg_manager):
        """
        Creates 3D viewer and handles both merged volume building and 3D mesh loading.
        Shows unified progress dialog for both operations.
        """
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
            # Create and add the 3D viewer with volume data
            self.viewer_3d = SegmentationViewer3D(
                nifti_files=file_paths,
                parent=self,
                volume_data=self.volume_data,
                affine=self.affine,
                dims=self.dims,
                intensity_min=self.intensity_min,
                intensity_max=self.intensity_max
            )
            self.layout.addWidget(self.viewer_3d)

            # Connect the finished signal to notify main window
            self.viewer_3d.loading_finished.connect(self._on_3d_loading_finished)
            # Connect the merged volume ready signal to update MPR views
            self.viewer_3d.merged_volume_ready.connect(self._on_merged_volume_ready)

            # Initialize with unified loading (merge + 3D)
            self.viewer_3d.initialize_with_merge(seg_manager)

            # Update slider ranges after initialization
            if self.dims:
                self.viewer_3d.update_slider_ranges()
        except Exception as e:
            print(f"Failed to create 3D viewer: {e}")
            self.clear_segmentations()  # Revert to placeholder on error
            # Optionally, show an error message in the placeholder
            if self.placeholder:
                self.placeholder.setText(f"Failed to create 3D view:\n{e}")

    def _on_merged_volume_ready(self):
        """Called when merged volume is ready - update 2D MPR views."""
        if self.main_window:
            # Notify MPR widget to update (2D views ready)
            self.main_window.mpr_widget.set_segmentation_visibility(True)
            # Only update MPR views if the widget is currently visible
            if self.main_window.mpr_widget.isVisible():
                self.main_window.mpr_widget.update_all_views()
            print("2D segmentation views updated")

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
            print("3D: Surface mode enabled")

            # Disable planes in viewer if it exists
            if self.viewer_3d is not None:
                self.viewer_3d.toggle_planes_mode(False)

    def toggle_planes_mode(self, checked):
        """Toggle planes mode in 3D view"""
        if checked:
            self.planes_mode_enabled = True
            self.surface_mode_enabled = False
            print("3D: Planes mode enabled")

            # Create viewer if it doesn't exist but we have volume data
            if self.viewer_3d is None and self.volume_data is not None:
                self._create_viewer_if_needed()

            # Enable planes in viewer if it exists
            if self.viewer_3d is not None:
                # Update viewer with current slice positions from MPR widget
                if hasattr(self.main_window, 'mpr_widget'):
                    slices = self.main_window.mpr_widget.slices
                    self.viewer_3d.update_slices(slices)
                self.viewer_3d.toggle_planes_mode(True)
            else:
                print("3D: Please load volume data first")
        else:
            # Disable planes when unchecked
            if self.viewer_3d is not None:
                self.viewer_3d.toggle_planes_mode(False)

    def update_slice_positions(self, slices_dict):
        """Update plane positions when slices change in MPR view"""
        # Only update if widget is visible to prevent rendering conflicts
        if self.isVisible() and self.viewer_3d is not None and self.planes_mode_enabled:
            self.viewer_3d.update_slices(slices_dict)

    def _create_viewer_if_needed(self):
        """Create a basic 3D viewer even without segmentations, for planes mode"""
        if self.viewer_3d is not None or self.volume_data is None:
            return

        # Remove placeholder
        if self.placeholder:
            self.layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None

        try:
            # Create viewer with empty segmentation list
            self.viewer_3d = SegmentationViewer3D(
                nifti_files=[],  # No segmentations
                parent=self,
                volume_data=self.volume_data,
                affine=self.affine,
                dims=self.dims,
                intensity_min=self.intensity_min,
                intensity_max=self.intensity_max
            )
            self.layout.addWidget(self.viewer_3d)
            self.viewer_3d.initialize()
            # Update slider ranges after initialization
            if self.dims:
                self.viewer_3d.update_slider_ranges()
            print("3D: Created viewer for planes mode")
        except Exception as e:
            print(f"Failed to create 3D viewer: {e}")
            # Recreate placeholder on error
            if self.placeholder is None:
                self.placeholder = QLabel("3D View\n\nLoad data to view in 3D.")
                self.placeholder.setAlignment(Qt.AlignCenter)
                self.placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.placeholder.setStyleSheet("""
                    background-color: black;
                    color: #4A5568;
                    font-size: 18px;
                """)
                self.layout.addWidget(self.placeholder)
