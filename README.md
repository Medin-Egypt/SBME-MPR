# 🩻 MPR Viewer
**Multi-Planar Reconstruction Viewer for Medical Imaging**
# Overview

**MPR (Multi-Planar Reconstruction) Viewer** is a comprehensive medical imaging application built with **PyQt5** that enables visualization and manipulation of **3D medical imaging data**.

The application supports both **NIfTI** and **DICOM** file formats, offering advanced viewing capabilities including:

* Multi-planar reconstruction (MPR)
* Segmentation overlay
* Oblique slicing
* AI-powered orientation detection
  
 ![](https://github.com/MhmdSheref/CUFE-MPR/blob/10a433384a0e6b7e8cadd6f265bdb146c25e09e1/assets/Overview.png)
<div align="center">
  <em>Main interface showing simultaneous multi-planar views</em>
</div>

This tool is designed for **medical professionals, researchers, and students** working with volumetric medical imaging data, providing intuitive controls and powerful visualization features for comprehensive data analysis.
#  Features
**File Support**  

**NIfTI Format:** Load and visualize `.nii` and`.nii.gz` files 

**DICOM Format:** Import entire `.DICOM` series from folders  

**Export Capabilities:** Export processed volumes to both **NIfTI** and **DICOM** formats with full metadata preservation         
<div align="center">
Show Image 
  </div>
<div align="center">                                  
Loading and visualizing medical imaging files                             
</div>

**Viewing Modes**

**3 Main Views:** Simultaneous axial, coronal, and sagittal plane visualization
**Oblique View:** Custom oblique plane slicing with interactive rotation controls
**Segmentation View:** Overlay and visualize segmentation masks with edge detection

<div align="center">

# Viewing Modes

| Main Views | Oblique View | Segmentation View |
|:----------:|:------------:|:-----------------:|
| ![Main Views](path/to/main-views.png) | ![Oblique View](path/to/oblique-view.png) | ![Segmentation](path/to/segmentation-view.png) |
| Simultaneous axial, coronal, and sagittal visualization | Custom oblique plane with rotation | Overlay segmentation masks with edge detection |

</div>
  
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

5.**Switch View Modes**          
Toggle between:     
**Main View** — standard three-plane mode.    
**Oblique View** — diagonal reconstruction.        
**Segmentation View** — if segmentation data is available.        

6.**Export Processed Data**      
Save results in:    
**NIfTI format (.nii, .nii.gz)**      
**DICOM series**       
Export preserves the current orientation and cropping settings.       
## 🧪 Example Workflow 
![](https://github.com/MhmdSheref/CUFE-MPR/blob/3b9614a7bb250bf6c3e039d1a2b5cfee9b165e6e/assets/IMG_0409.gif)



## 👩‍💻 Contributors

**🧑‍🤝‍🧑 Team Members**
  - **Mohamed Sherif** 
  - **Bassel Mostafa**
  - **Mahmoud Zahran** 
  - **Rawan Kotb** 

 **🧭 Supervised By**     
         - **Prof. Tamer Basha**     
         - **Eng. Alaa Tarek**     




   

