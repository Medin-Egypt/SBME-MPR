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
## ⚙️ Tech Stack  

| Layer | Technology / Library | Purpose |
|-------|-----------------------|----------|
| **Language** | Python 3 | Core programming language |
| **GUI Framework** | PyQt5 | For the main application interface |
| **Medical Data Handling** | pydicom, nibabel | Load and interpret DICOM & NIfTI data |
| **Computation** | numpy, scipy | Image array operations and interpolation |
| **Machine Learning** | TensorFlow | Detect orientation & image metadata |
| **Styling** | QSS | Custom dark UI theme |
| **System Tools** | os, sys, time | File I/O and app runtime functions |
## 💻 Requirements      
Before running, install all dependencies:     
```bash
pip install pyqt5 tensorflow numpy scipy nibabel pydicom
```
## 🚀 How It Works        
1.**Run the application**       
2.**Load Medical Data**        
Choose **“Open DICOM Folder”** or **“Open NIfTI File”** from the sidebar.    
The `loader.py` module:     
• Reads the DICOM or NIfTI data.    
• Sorts slices by their spatial position.      
• Fixes orientation and left–right mirroring issues.    
• Applies proper intensity windowing for clear contrast.   
![](https://github.com/MhmdSheref/CUFE-MPR/blob/ce36d382a09da99d1d801f93bf7d0f8cc2a9d1e4/assets/Screenshot%201.png)
3.**Visualize the Scans**      
View the 3D dataset in four synchronized panels:     
**Axial**     
**Coronal**        
**Sagittal**       
**Oblique**              
Move through slices using your mouse wheel or the cine playback controls.      
4.**Interact with the Images**       
Use tools from the sidebar to:       
🔍 **Zoom / Pan** — explore image details.   
![](https://github.com/MhmdSheref/CUFE-MPR/blob/0ed7c0efd7d05154d8f3640a1c143b9616fc6f04/assets/IMG_0410.gif)   
💡 **Contrast** — adjust brightness and contrast levels. 
![](https://github.com/MhmdSheref/CUFE-MPR/blob/1b0376c0a818e0f042fa8fcffb17220b39ab76e7/assets/IMG_0407.gif)
✂️ **Crop** — focus on a region of interest.       
🎞 **Cine Mode** — play slices automatically like a short animation.
![](https://github.com/MhmdSheref/CUFE-MPR/blob/09e97cac40f461157c84ce14168445d3a77fd1a1/assets/IMG_0408.gif)
4.**Switch View Modes**          
Toggle between:     
**Main View** — standard three-plane mode.    
**Oblique View** — diagonal reconstruction.        
**Segmentation View** — if segmentation data is available.        
5.**Export Processed Data**      
Save results in:    
**NIfTI format (.nii, .nii.gz)**      
**DICOM series**       
Export preserves the current orientation and cropping settings.       
## 🧪 Example Workflow      



## 👩‍💻 Contributors

**🧑‍🤝‍🧑 Team Members**
  - **Mohamed Sherif** 
  - **Bassel Mostafa**
  - **Mahmoud Zahran** 
  - **Rawan Kotb** 

 **🧭 Supervised By**     
         - **Prof. Tamer Basha**     
         - **Eng. Alaa Tarek**     




   

