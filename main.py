import tensorflow as tf
import os
import sys
import nibabel as nib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QSizePolicy, QButtonGroup, QFileDialog, QMessageBox,
    QDialog, QComboBox, QAction, QMenu
)
from PyQt5.QtCore import Qt, QSize, QEvent, QTimer, QPoint
from PyQt5.QtGui import QPixmap, QIcon, QImage, QColor
import utils.loader as loader
import utils.detect_orientation as od
from utils.ui_classes import SliceCropDialog

# Import the new widget classes
from mpr_widget import MPRWidget
from td_widget import TDWidget


class MPRViewer(QMainWindow):
    def __init__(self, file_path=None):
        super().__init__()
        # --- Instantiate Child Widgets ---
        self.mpr_widget = MPRWidget(self)
        self.td_widget = TDWidget(self)

        self.metadata = None
        self.setWindowTitle("MPR VIEWER")
        self.setGeometry(100, 100, 1200, 800)

        self.setMinimumSize(800, 600)
        self.setMaximumSize(16777215, 16777215)

        # Remove default title bar
        self.setWindowFlags(Qt.FramelessWindowHint)

        # --- Data Properties (Owned by Main Window) ---
        self.data = None
        self.affine = None
        self.dims = None
        self.intensity_min = 0
        self.intensity_max = 255
        self.file_loaded = False
        self.original_intensity_min = 0
        self.original_intensity_max = 255
        self.crop_bounds = None
        self.original_data = None
        self.segmentation_files = []
        self.segmentation_data_list = []
        self.original_segmentation_data_list = []

        # --- Window Dragging ---
        self.drag_position = None
        self.is_maximized = False

        # --- Create main container ---
        container = QWidget()
        self.setCentralWidget(container)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Create custom title bar
        title_bar = self.create_title_bar()
        container_layout.addWidget(title_bar)

        # --- Create content widget ---
        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        sidebar = self.create_sidebar()



        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.mpr_widget)
        main_layout.addWidget(self.td_widget)
        main_layout.setStretch(0, 0)  # Sidebar
        main_layout.setStretch(1, 1)  # MPR Widget
        main_layout.setStretch(2, 1)  # 3D Widget

        container_layout.addWidget(content_widget)

        # --- Add Icons ---
        self.add_image_to_button("mpr_mode_btn_0", "Icons/windows.png", "3 Main Views")
        self.add_image_to_button("mpr_mode_btn_1", "Icons/heart.png", "Segmentation View")
        self.add_image_to_button("mpr_mode_btn_2", "Icons/diagram.png", "Oblique View")
        self.add_image_to_button("td_mode_btn_0", "Icons/surface.png", "Surface Mode")
        self.add_image_to_button("td_mode_btn_1", "Icons/Planes.png", "Planes Mode")
        self.add_image_to_button("tool_btn_0_0", "Icons/mouse.png", "Navigation")
        self.add_image_to_button("tool_btn_0_1", "Icons/brightness.png", "Contrast")
        self.add_image_to_button("tool_btn_0_2", "Icons/loupe.png", "Zoom/Pan")
        self.add_image_to_button("tool_btn_1_0", "Icons/expand.png", "Crop")
        self.add_image_to_button("tool_btn_1_1", "Icons/rotating-arrow-to-the-right.png", "Rotate")
        self.add_image_to_button("tool_btn_1_2", "Icons/video.png", "Cine Mode")
        self.add_image_to_button("export_btn_0", "Icons/NII.png", "NIFTI Export")
        self.add_image_to_button("export_btn_1", "Icons/DIC.png", "DICOM Export")

        # --- Set Initial State ---
        main_views_btn = self.findChild(QPushButton, "mpr_mode_btn_0")
        if main_views_btn:
            main_views_btn.setChecked(True)

        default_tool = self.findChild(QPushButton, "tool_btn_0_0")
        if default_tool:
            default_tool.setChecked(True)

        self.switch_to_tab("mpr")  # Show MPR tab by default

    def create_title_bar(self):
        """Create custom title bar with window controls and tab bar"""
        title_bar_container = QWidget()
        title_bar_container.setObjectName("title_bar_container")

        container_layout = QVBoxLayout(title_bar_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Top title bar
        title_bar = QWidget()
        title_bar.setObjectName("custom_title_bar")
        title_bar.setFixedHeight(35)

        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title label
        title_label = QLabel("SBME29 MPR")
        title_label.setObjectName("title_label")
        layout.addWidget(title_label)
        layout.addSpacing(10)

        # Add Import button next to title
        import_btn = QPushButton("Import")
        import_btn.setObjectName("import_btn")
        import_btn.setFixedHeight(30)
        import_btn.clicked.connect(self.show_import_menu)
        layout.addWidget(import_btn)
        layout.addSpacing(10)

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

        container_layout.addWidget(title_bar)

        # Tab bar
        tab_bar = QWidget()
        tab_bar.setObjectName("tab_bar")
        tab_bar.setFixedHeight(35)

        tab_layout = QHBoxLayout(tab_bar)
        tab_layout.setContentsMargins(10, 0, 0, 0)
        tab_layout.setSpacing(5)

        # Create button group for tabs to make them mutually exclusive
        self.tab_button_group = QButtonGroup(self)
        self.tab_button_group.setExclusive(True)

        # MPR tab
        mpr_tab = QPushButton("MPR")
        mpr_tab.setObjectName("mpr_tab")
        mpr_tab.setCheckable(True)
        mpr_tab.setChecked(True)
        mpr_tab.setFixedSize(80, 30)
        tab_layout.addWidget(mpr_tab)
        self.tab_button_group.addButton(mpr_tab)

        # 3D tab
        td_tab = QPushButton("3D")
        td_tab.setObjectName("td_tab")
        td_tab.setCheckable(True)
        td_tab.setFixedSize(80, 30)
        tab_layout.addWidget(td_tab)
        self.tab_button_group.addButton(td_tab)

        tab_layout.addStretch()
        # Connect tab buttons to switch function
        mpr_tab.clicked.connect(lambda: self.switch_to_tab("mpr"))
        td_tab.clicked.connect(lambda: self.switch_to_tab("3d"))
        container_layout.addWidget(tab_bar)

        return title_bar_container

    def switch_to_tab(self, tab_name):
        """Switch between MPR and 3D tabs"""
        if tab_name == "mpr":
            # Show MPR view, hide 3D view
            self.mpr_widget.show()
            self.td_widget.hide()
            # Show MPR mode buttons, hide 3D mode buttons
            self.mpr_mode_group.show()
            self.td_mode_group.hide()
        elif tab_name == "3d":
            # Hide MPR view, show 3D view
            self.mpr_widget.hide()
            self.td_widget.show()
            # Hide MPR mode buttons, show 3D mode buttons
            self.mpr_mode_group.hide()
            self.td_mode_group.show()

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFrameStyle(QFrame.Box)
        sidebar.setFixedWidth(200)

        layout = QVBoxLayout(sidebar)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)

        # MPR Mode section (only visible in MPR tab)
        self.mpr_mode_group = QGroupBox("Mode:")
        mpr_mode_layout = QVBoxLayout()
        mpr_mode_buttons_widget = QWidget()
        mpr_mode_buttons_layout = QGridLayout(mpr_mode_buttons_widget)
        self.mpr_mode_group_buttons = QButtonGroup(self)
        self.mpr_mode_group_buttons.setExclusive(False)
        for i in range(3):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"mpr_mode_btn_{i}")
            btn.setCheckable(True)
            mpr_mode_buttons_layout.addWidget(btn, 0, i)
            self.mpr_mode_group_buttons.addButton(btn, i)
            if i == 0:
                # Connect to the method in MPRWidget
                btn.clicked.connect(self.mpr_widget.toggle_main_views)
            elif i == 1:
                # Connect to the method in MPRWidget
                btn.clicked.connect(self.mpr_widget.toggle_segmentation_view)
            elif i == 2:
                # Connect to the method in MPRWidget
                btn.clicked.connect(self.mpr_widget.toggle_oblique_view)

        mpr_mode_layout.addWidget(mpr_mode_buttons_widget)
        self.mpr_mode_group.setLayout(mpr_mode_layout)
        layout.addWidget(self.mpr_mode_group)

        # 3D Mode section (only visible in 3D tab)
        self.td_mode_group = QGroupBox("3D Mode:")
        td_mode_layout = QVBoxLayout()
        td_mode_buttons_widget = QWidget()
        td_mode_buttons_layout = QGridLayout(td_mode_buttons_widget)
        self.td_mode_group_buttons = QButtonGroup(self)
        self.td_mode_group_buttons.setExclusive(True)
        for i in range(2):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setObjectName(f"td_mode_btn_{i}")
            btn.setCheckable(True)
            td_mode_buttons_layout.addWidget(btn, 0, i)
            self.td_mode_group_buttons.addButton(btn, i)
            if i == 0:
                # Connect to the method in TDWidget
                btn.clicked.connect(self.td_widget.toggle_surface_mode)
            elif i == 1:
                # Connect to the method in TDWidget
                btn.clicked.connect(self.td_widget.toggle_planes_mode)

        td_mode_layout.addWidget(td_mode_buttons_widget)
        self.td_mode_group.setLayout(td_mode_layout)
        layout.addWidget(self.td_mode_group)
        self.td_mode_group.hide()  # Initially hidden

        # Add Segmentation section
        seg_group = QGroupBox("Segmentation:")
        seg_layout = QVBoxLayout()

        load_seg_btn = QPushButton("Load Segmentation")
        load_seg_btn.setObjectName("load_seg_btn")
        load_seg_btn.setMinimumHeight(35)
        load_seg_btn.clicked.connect(self.load_segmentation_files)
        seg_layout.addWidget(load_seg_btn)

        delete_seg_btn = QPushButton("Remove Segmentation")
        delete_seg_btn.setObjectName("delete_seg_btn")
        delete_seg_btn.setMinimumHeight(35)
        delete_seg_btn.clicked.connect(self.delete_segmentation_files)
        seg_layout.addWidget(delete_seg_btn)

        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)

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
                object_name = f"tool_btn_{r}_{c}"
                btn.setObjectName(object_name)
                tools_layout.addWidget(btn, r, c)

                if object_name == "tool_btn_1_0":
                    btn.setCheckable(False)
                    btn.clicked.connect(self.show_slice_crop_dialog)
                else:
                    btn.setCheckable(True)
                    self.tools_group_buttons.addButton(btn, r * 3 + c)

        # Connect cine and rotate buttons to their handlers in MPRWidget
        cine_btn = self.findChild(QPushButton, "tool_btn_1_2")
        if cine_btn:
            cine_btn.clicked.connect(self.mpr_widget.handle_cine_button_toggle)

        rotate_btn = self.findChild(QPushButton, "tool_btn_1_1")
        if rotate_btn:
            rotate_btn.clicked.connect(self.mpr_widget.handle_rotate_mode_toggle)

        tools_main_layout.addWidget(tools_grid_widget)

        # Add the Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("reset_btn")
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

    def show_import_menu(self):
        """Show import options menu"""
        menu = QMenu(self)

        open_dicom = menu.addAction("Open DICOM")
        open_dicom.triggered.connect(self.open_dicom_folder)

        open_nifti = menu.addAction("Open NIfTI")
        open_nifti.triggered.connect(self.open_nifti_file)

        # Show menu at the import button
        import_btn = self.sender()
        menu.exec_(import_btn.mapToGlobal(import_btn.rect().bottomLeft()))

    # --- Window Dragging Methods ---

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

    # --- Data Loading Methods ---

    def open_nifti_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "",
                                                   "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if file_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max, self.metadata = loader.load_nifti_data(
                    file_path)
                self.file_loaded = True

                self.original_intensity_min = self.intensity_min
                self.original_intensity_max = self.intensity_max
                self.original_data = self.data.copy()
                self.crop_bounds = None

                # --- Pass data to child widgets ---
                self.mpr_widget.set_data(self.data, self.affine, self.dims, self.intensity_min, self.intensity_max)
                self.td_widget.set_data(self.data, self.affine)  # 3D widget may need data

                QMessageBox.information(self, "Success", f"NIfTI file loaded successfully!\nDimensions: {self.dims}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load NIfTI file:\n{str(e)}")

    def open_dicom_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder", "", QFileDialog.ShowDirsOnly)
        if folder_path:
            try:
                self.data, self.affine, self.dims, self.intensity_min, self.intensity_max, self.metadata = loader.load_dicom_data(
                    folder_path)
                self.file_loaded = True

                self.original_intensity_min = self.intensity_min
                self.original_intensity_max = self.intensity_max
                self.original_data = self.data.copy()
                self.crop_bounds = None

                # --- Pass data to child widgets ---
                self.mpr_widget.set_data(self.data, self.affine, self.dims, self.intensity_min, self.intensity_max)
                self.td_widget.set_data(self.data, self.affine)  # 3D widget may need data

                orientation, confidence, _ = od.predict_middle_dicom_from_folder(folder_path)
                orientation_info = f"\n\nDetected Orientation: {orientation} with confidence: {(confidence * 100):.2f}%"
                meta_info = f"Body Part Examined: {self.metadata.get('BodyPartExamined')}\nStudy Description: {self.metadata.get('StudyDescription')}"

                QMessageBox.information(
                    self, "Success",
                    f"DICOM folder loaded successfully!\nDimensions: {self.dims}{orientation_info}\n\n{meta_info}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DICOM folder:\n{str(e)}")

    def load_segmentation_files(self):
        """Opens a file dialog to select multiple NIfTI segmentation files."""
        if not self.file_loaded:
            QMessageBox.warning(self, "No Data", "Please load a main file first.")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Segmentation Files", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        if not file_paths:
            return

        self.segmentation_files = []
        self.segmentation_data_list = []
        self.original_segmentation_data_list = []

        for file_path in file_paths:
            try:
                nifti_file = nib.load(file_path)
                seg_data = nifti_file.get_fdata()
                seg_data = seg_data[::-1, :, :]  # Apply same flip as main data

                if seg_data.shape != self.data.shape:
                    QMessageBox.warning(
                        self, "Dimension Mismatch",
                        f"Segmentation file {os.path.basename(file_path)} has different dimensions.\n"
                        f"Expected: {self.data.shape}, Got: {seg_data.shape}"
                    )
                    continue

                self.segmentation_files.append(file_path)
                self.segmentation_data_list.append(seg_data)
                self.original_segmentation_data_list.append(seg_data.copy())  # Store original

            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load segmentation file {os.path.basename(file_path)}:\n{str(e)}"
                )

        if self.segmentation_data_list:
            # Notify MPR widget to update
            self.mpr_widget.set_segmentation_visibility(True)
            self.mpr_widget.update_all_views()
            QMessageBox.information(
                self, "Success", f"Loaded {len(self.segmentation_data_list)} segmentation file(s)."
            )

    def delete_segmentation_files(self):
        """Deletes all loaded segmentation files."""
        if not self.segmentation_data_list:
            QMessageBox.information(self, "No Segmentation", "No segmentation files are currently loaded.")
            return

        reply = QMessageBox.question(
            self, "Delete Segmentation",
            f"Are you sure you want to delete all {len(self.segmentation_data_list)} segmentation file(s)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.segmentation_files = []
            self.segmentation_data_list = []
            self.original_segmentation_data_list = []

            # Notify MPR widget to update
            self.mpr_widget.set_segmentation_visibility(False)
            self.mpr_widget.update_all_views()

            QMessageBox.information(self, "Success", "All segmentation files have been deleted.")

    # --- Data Manipulation Methods (Crop) ---

    def show_slice_crop_dialog(self):
        """Shows a dialog to get a slice range and applies the crop."""
        if not self.file_loaded or self.original_data is None:
            QMessageBox.warning(self, "No File", "Please load a file first.")
            return

        total_slices = self.original_data.shape[2]
        dialog = SliceCropDialog(total_slices, self)

        if dialog.exec_() == QDialog.Accepted:
            start, end = dialog.get_values()
            if start >= end:
                QMessageBox.critical(self, "Input Error", "The 'from' slice must be smaller than the 'to' slice.")
                return

            start_idx = start - 1
            end_idx = end - 1
            self.apply_slice_crop(start_idx, end_idx)

    def apply_slice_crop(self, start_idx, end_idx):
        """Crops the data volume to the specified slice range (axial)."""
        if self.original_data is None:
            return

        self.data = self.original_data[:, :, start_idx: end_idx + 1].copy()
        self.dims = self.data.shape

        # Also crop loaded segmentations
        new_seg_list = []
        for seg_data in self.original_segmentation_data_list:
            new_seg_list.append(seg_data[:, :, start_idx: end_idx + 1].copy())
        self.segmentation_data_list = new_seg_list

        # Notify MPR widget of the data change
        self.mpr_widget.update_data(self.data, self.dims, self.segmentation_data_list)

        QMessageBox.information(self, "Crop Applied",
                                f"Volume cropped to show slices {start_idx + 1} to {end_idx + 1}.\n"
                                f"New dimensions: {self.dims}")

    # --- Reset Logic Methods ---

    def on_reset_clicked(self):
        """Handler for the Reset button. Delegates to child widgets or self."""
        if not self.file_loaded:
            return

        checked_btn = self.tools_group_buttons.checkedButton()
        if not checked_btn:
            # Handle non-checkable crop button
            active_tool = self.tools_group_buttons.sender()
            if active_tool and active_tool.objectName() == "tool_btn_1_0":
                self.reset_crop()
                self.mpr_widget.update_all_views()
            return

        btn_name = checked_btn.objectName()

        if btn_name == "tool_btn_0_0":
            self.mpr_widget.reset_crosshair_and_slices()
        elif btn_name == "tool_btn_0_1":
            self.reset_contrast()
        elif btn_name == "tool_btn_0_2":
            self.mpr_widget.reset_all_zooms()
        elif btn_name == "tool_btn_1_0":
            self.reset_crop()
        elif btn_name == "tool_btn_1_1":
            self.mpr_widget.reset_rotation()

        self.mpr_widget.update_all_views()  # Update views after reset

    def reset_contrast(self):
        """Resets the window/level to the initial values from file load."""
        self.intensity_min = self.original_intensity_min
        self.intensity_max = self.original_intensity_max

    def reset_crop(self):
        """Resets the crop to show the full volume."""
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self.dims = self.data.shape
            self.crop_bounds = None

            self.segmentation_data_list = [seg.copy() for seg in self.original_segmentation_data_list]

            # Notify MPR widget of data change
            self.mpr_widget.update_data(self.data, self.dims, self.segmentation_data_list)

    # --- Export Methods ---

    def toggle_export_button(self, button):
        if button.isChecked():
            button.setChecked(False)

            if not self.file_loaded:
                QMessageBox.warning(self, "No Data", "Please load a file first.")
                return

            # Note: The original code had a self.crop_bounds check
            # but apply_crop() was not defined.
            # This logic assumes self.data holds the (potentially cropped) data to be exported.
            # If self.crop_bounds was used for a different cropping, that logic is missing.

            if button.objectName() == "export_btn_1":  # DICOM export
                output_dir = QFileDialog.getExistingDirectory(
                    self, "Select folder to save DICOM file", ""
                )
                if output_dir:
                    success = loader.export_to_dicom(self.data, self.affine, output_dir, self.metadata)
                    if success:
                        QMessageBox.information(self, "Export Successful", f"DICOM file saved to:\n{output_dir}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to save DICOM file.")


            elif button.objectName() == "export_btn_0":  # NIFTI export
                output_file = QFileDialog.getSaveFileName(
                    self, "Save NIfTI file", "", "NIfTI Files (*.nii.gz *.nii)"
                )
                if output_file[0]:
                    success = loader.export_to_nifti(self.data, self.affine, output_file[0], self.metadata)
                    if success:
                        QMessageBox.information(self, "Export Successful",
                                                f"NIfTI file saved to:\n{output_file[0]}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to save NIfTI file.")

    # --- Utility Methods ---

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