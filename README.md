# 🩻 MPR Viewer
**Multi-Planar Reconstruction Viewer for Medical Imaging**
# 🧠 Overview
**MPR Viewer** (Multi-Planar Reconstruction Viewer) is a powerful desktop application built with **Python** and **PyQt5** that allows users to **view , analyze ,and interact** with medical imaging data such as **DICOM** and **NIfTI** scans.
It reconstructs 3D medical images and displays them in multiple planes **— Axial, Coronal, Sagittal, and Oblique —** providing doctors, students, and researchers with a precise way to explore medical data.
# ⚙️ Features
• **Data Import**   
Load **NIfTI (.nii, .nii.gz)** files or **DICOM** folders (multi-slice series)     
• **Multi-Planar Views**       
View synchronized **Axial, Coronal, Sagittal**, and **Oblique** slices simultaneously.         
• **Interactive Tools**         
Zoom, Pan, Crop, and adjust Contrast dynamically using the mouse.          
• **Cine Mode**          
Automatically scroll through slices (like a playback animation).      
• **Crosshair Synchronization**       
Crosshair moves in real-time across all viewing planes.        
• **Automatic Orientation & Intensity Handling**           
Uses DICOM **WindowCenter/WindowWidth** and spacing metadata for accurate reconstruction.       
• **3D Transformations**        
Rotation and oblique slicing supported for advanced visualization.       
• **Image Correction**        
Automatic left–right flipping and sorting for proper anatomical alignment.       
• **Modern Dark-Mode UI**         
Custom QSS styling with clean design and smooth contrast.        
• **Export**       
Save processed data back to **DICOM** or **NIfTI** format.        
