# MPR Viewer
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
</div>

This tool is designed for **medical professionals, researchers, and students** working with volumetric medical imaging data, providing intuitive controls and powerful visualization features for comprehensive data analysis.
#  Features
* **File Support**  

**NIfTI Format:** Load and visualize `.nii` and`.nii.gz` files 

**DICOM Format:** Import entire `.DICOM` series from folders  

**Export Capabilities:** Export processed volumes to both **NIfTI** and **DICOM** formats with full metadata preservation         


* **Viewing Modes**

**3 Main Views:** Simultaneous axial, coronal, and sagittal plane visualization                

**Oblique View:** Custom oblique plane slicing with interactive rotation controls                

**Segmentation View:** Overlay and visualize segmentation masks with edge detection                    


 * **Interactive Tools**

<div align="center">

<table>
<tr>
<td align="center" width="33%">
<img src="assets/icon1.png" width="60"/><br/>
<b>Slide/Crosshair Mode</b><br/>
<sub>Navigate through slices with synchronized crosshairs across all views</sub>
</td>
<td align="center" width="33%">
<img src="assets/icon2.png" width="60"/><br/>
<b>Contrast Mode</b><br/>
<sub>Adjust window/level settings for optimal visualization</sub>
</td>
<td align="center" width="33%">
<img src="assets/icon3.png" width="60"/><br/>
<b>Zoom/Pan Mode</b><br/>
<sub>Coordinated zooming and panning across all views</sub>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="assets/icon4.png" width="60"/><br/>
<b>Crop Mode</b><br/>
<sub>Slice-based cropping to focus on regions of interest</sub>
</td>
<td align="center" width="33%">
<img src="assets/icon5.png" width="60"/><br/>
<b>Rotate Mode</b><br/>
<sub>Interactive oblique plane rotation with visual indicators</sub>
</td>
<td align="center" width="33%">
<img src="assets/icon6.png" width="60"/><br/>
<b>Cine Mode</b><br/>
<sub>Automated slice-by-slice playback for dynamic viewing</sub>
</td>
</tr>
</table>

</div>

* **Advanced Features**

**Segmentation Overlay**: Load multiple segmentation files with edge detection and red outline visualization
  
 **Aspect Ratio Correction**: Automatic voxel spacing-based aspect ratio correction for accurate anatomical representation
 
 **AI Orientation Detection**: TensorFlow-based model for automatic anatomical orientation classification
  
 **Coordinated Zoom**: Uniform scaling across all views maintaining spatial relationships
  
 **Custom Title Bar**: Modern, frameless window design with drag, minimize, maximize, and restore functionality     

# Requirements
```
pip install -r requirements.txt
```
# Example Workflow
**1) Load Medical Data**        
Click **"Open NIfTI File"** or **"Open DICOM Folder"** to import your medical imaging data.
**The application automatically:**

* Detects orientation (for DICOM files)
* Calculates optimal window/level settings
* Applies aspect ratio correction

![](https://github.com/MhmdSheref/CUFE-MPR/blob/7394bdc6530d9d705ee93e7f0ee1b7e5331d3209/assets/Overview.png)




**2) Navigate and Explore**
Use intuitive controls to explore your data:

* **Mouse wheel:** Scroll through slices
* **Click and drag:** Move crosshair to specific locations
* **Double-click:** Maximize any view for detailed inspection

![](https://github.com/MhmdSheref/CUFE-MPR/blob/fd047e37bd47328033e9fe22546262ddac866ee8/assets/gif%20converted/Navigation%20tool.gif)



**3) Adjust Visualization**
Fine-tune the display for optimal visualization:

* **Contrast Mode:** Drag to adjust window/level
* **Zoom/Pan Mode:** Wheel to zoom, drag to pan
* **Reset:** Restore original settings anytime

<div align="center">

  <table>
    <tr>
      <td align="center">
        <img src="assets/gif converted/Contrast tool.gif" width="420"/><br>
        <b>Contrast Mode</b>
      </td>
      <td align="center">
        <img src="assets/gif converted/Zoom tool.gif" width="420"/><br>
        <b>Zoom/Pan Mode</b>
      </td>
    </tr>
  </table>

</div>


**4) Work with Segmentations**
**Load and visualize segmentation masks:**

* Click "Load Segmentation" to add masks
* Switch to "Segmentation View" for dedicated visualization
* Overlays appear as red outlines in all views

  ![](https://github.com/MhmdSheref/CUFE-MPR/blob/832743130d833ccc8c2c38e6fe9c97585858b586/assets/Segmentaion%20view.png)


**5) Use Oblique Slicing**
**Create custom viewing angles:**

* Switch to "Oblique View" mode
* Enable "Rotate Mode"
* Drag the yellow axis handle to adjust angle
* Oblique view updates in real-time

![](https://github.com/MhmdSheref/CUFE-MPR/blob/b0622ede04e272644e075070303e5f96be9579c9/assets/gif%20converted/Rotate%20tool.gif)

**6) Crop and Export**
**Process and export your data:**

* Click the Crop tool to select slice range
* Choose export format (NIfTI or DICOM)
* All metadata and modifications are preserved

![](https://github.com/MhmdSheref/CUFE-MPR/blob/fd047e37bd47328033e9fe22546262ddac866ee8/assets/gif%20converted/Crop%20tool.gif)

##  Contributors

<div align="center">

###  Team Members

<table>
<tr>
<td align="center">
<a href="https://github.com/MhmdSheref">
<img src="https://github.com/MhmdSheref.png" width="100px;" alt="MhmdSheref"/><br />
<sub><b>MhmdSheref</b></sub>

</td>

<td align="center">
<a href="https://github.com/BasselM0stafa">
<img src="https://github.com/BasselM0stafa.png" width="100px;" alt="BasselM0stafa"/><br />
<sub><b>Bassel Mostafa</b></sub>
</td>

<td align="center">
<a href="https://github.com/MahmoudZah">
<img src="https://github.com/MahmoudZah.png" width="100px;" alt="MahmoudZah"/><br />
<sub><b>Mahmoud Zahran</b></sub>

</td>

<td align="center">
<a href="https://github.com/RwanOtb">
<img src="https://github.com/RwanOtb.png" width="100px;" alt="RwanOtb"/><br />
<sub><b>RwanOtb</b></sub>
</td>
</tr>
</table>

                                        
</div>   

# Supervised By

* **Prof. Tamer Basha**                              
* **Eng. Alaa Tarek**   
