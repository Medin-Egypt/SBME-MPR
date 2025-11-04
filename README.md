# MPR Viewer
**Multi-Planar Reconstruction Viewer for Medical Imaging**                           
# Overview
**MPR (Multi-Planar Reconstruction) Viewer** is a comprehensive medical imaging application built with **PyQt5** that enables visualization and manipulation of **3D medical imaging data**.                                                              

The application supports both **NIfTI** and **DICOM** file formats, offering advanced viewing capabilities including:                                

* Multi-planar reconstruction (MPR)                        
* Segmentation overlay with memory-efficient management                     
* Oblique slicing with interactive rotation                    
* Curved MPR for vessel analysis                   
* AI-powered orientation detection                   
* Advanced 3D visualization with PyVista                       
* Interactive blood flow simulation                     
* Vessel flythrough navigation                  
* Theme switching (Dark/Light modes)                                     

 ![](https://github.com/Medin-Egypt/SBME-MPR/blob/e8d6dd1334719758716d2ccff2d066d229e1718d/assets/Overview.png)
<div align="center">
</div>

This tool is designed for **medical professionals, researchers, and students** working with volumetric medical imaging data, providing intuitive controls and powerful visualization features for comprehensive data analysis.
#  Features
* **File Support**  

**NIfTI Format:** Load and visualize `.nii` and`.nii.gz` files 

**DICOM Format:** Import entire `.DICOM` series from folders  

**Export Capabilities:** Export processed volumes to both **NIfTI** and **DICOM** formats with full metadata preservation         


**Viewing Modes**                              

**2D MPR Views:**                        

* **3 Main Views:** Simultaneous axial, coronal, and sagittal plane visualization                

* **Oblique View:** Custom oblique plane slicing with interactive rotation controls                

* **Segmentation View:** Dedicated view for segmentation visualization with plane selection
  
* **Curved View:** Advanced curved MPR for vessel straightening and analysis

**3D Visualization:**
  
* **Surface Mode:** High-quality 3D surface rendering of segmented structures
  
* **Planes Mode:** Interactive slice planes in 3D space
  
* **atomical System Organization:** Structures automatically categorized by system (Cardiovascular, Skeletal, Muscular, etc.)
  
* **Smart Caching:** Mesh and volume caching for faster repeated loads                                                                       


**Interactive Tools**
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
<tr>
<td align="center" width="33%">
<img src="assets/icon7.png" width="60"/><br/>
<b>Curved MPR Tool</b><br/>
<sub>Draw curves on vessels for straightened visualization</sub>
</td>
<td align="center" width="33%">
<img src="assets/icon8.png" width="60"/><br/>
<b>Flythrough Navigation</b><br/>
<sub>Camera flythrough along vessel centerlines</sub>
</td>
<td align="center" width="33%">
<img src="assets/icon9.png" width="60"/><br/>
<b>Focus Navigation</b><br/>
<sub>Right-click to isolate and focus on specific structures</sub>
</td>
</tr>
</table>
</div>

**Advanced Features**                  
 
**Segmentation Management:**                                                               

* Load multiple segmentation files with memory-efficient lazy loading              
* Edge detection with red outline visualization                 
* Merged volume view for comprehensive segmentation analysis               
* Smart caching for improved performance                 
 
**3D Advanced Features:**         

* Blood Flow Visualization: Animated blood flow simulation in vessels with adjustable heart rate                 
* Vessel Centerline Extraction: Automatic centerline computation for tubular structures           
* Flythrough Navigation: Interactive camera paths through vessels and spinal canal         
* Focus Navigation: Right-click to isolate and examine individual structures             
* Anatomical Categorization: Automatic organization by body systems               
* Opacity Control: Per-system opacity adjustment               

**UI/UX Enhancements:**                  
        
* Theme Switching: Toggle between dark and light modes with animated switch              
* Custom Title Bar: Modern, frameless window design with drag, minimize, maximize, and restore             
* Tabbed Interface: Seamless switching between 2D MPR and 3D views                  
* Progress Indicators: Background loading with cancellable progress dialogs               
* Coordinated Zoom: Uniform scaling across all views maintaining spatial relationships                           

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
* Organizes segmentations by anatomical system

![](https://github.com/MhmdSheref/CUFE-MPR/blob/7394bdc6530d9d705ee93e7f0ee1b7e5331d3209/assets/Overview.png)




**2) Navigate and Explore** 

Use intuitive controls to explore your data:                    

* **Mouse wheel:** Scroll through slices
* **Click and drag:** Move crosshair to specific locations
* **Double-click:** Maximize any view for detailed inspection
* **Tab switching:** Toggle between 2D MPR and 3D views



![](https://github.com/MhmdSheref/CUFE-MPR/blob/fd047e37bd47328033e9fe22546262ddac866ee8/assets/gif%20converted/Navigation%20tool.gif)



**3) Adjust Visualization**
Fine-tune the display for optimal visualization:

* **Contrast Mode:** Drag to adjust window/level
* **Zoom/Pan Mode:** Wheel to zoom, drag to pan
* **Reset:** Restore original settings anytime                
* **Theme Switch:** Toggle between dark and light modes                              



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
      <td align="center">
        <img src="assets/gif converted/switchthemes.gif" width="420"/><br>
        <b>Switch Themes</b>
      </td>
    </tr>
  </table>

</div>


**4) Work with Segmentations**
**Load and visualize segmentation masks:**

* Click "Load Segmentation" to add masks
* Switch to "Segmentation View" for dedicated visualization
* Overlays appear as red outlines in all views
* View merged segmentations in dedicated panel
* Background loading with progress tracking

Show Image

  ![](https://github.com/MhmdSheref/CUFE-MPR/blob/832743130d833ccc8c2c38e6fe9c97585858b586/assets/Segmentaion%20view.png)


**5) Use Oblique Slicing**
**Create custom viewing angles:**

* Switch to "Oblique View" mode
* Enable "Rotate Mode"
* Drag the yellow axis handle to adjust angle
* Oblique view updates in real-time

![](https://github.com/MhmdSheref/CUFE-MPR/blob/b0622ede04e272644e075070303e5f96be9579c9/assets/gif%20converted/Rotate%20tool.gif)


**6) Curved MPR for Vessels**                          
**Straighten curved vessels for better analysis:**                   

* Switch to "Curved View" mode
* Enable "Curved MPR" tool                             
* Left-click: Add control points along vessel                                       
* Right-click: Remove last point                     
* Double-click: Confirm curve and generate straightened view                         
* View frontal projection of curved structure                       

![](https://github.com/Medin-Egypt/SBME-MPR/blob/e8d6dd1334719758716d2ccff2d066d229e1718d/assets/CurvedMPR%20view.png)

**7) 3D Visualization**                                                
**Explore data in three dimensions:**

* Switch to "3D" tab                  
* Choose Surface Mode for 3D structure rendering                 
* Choose Planes Mode for interactive slice visualization                
* Toggle visibility by anatomical system                  
* Adjust opacity per system                
* Use smart loading - structures load on-demand                         

![](https://github.com/Medin-Egypt/SBME-MPR/blob/e8d6dd1334719758716d2ccff2d066d229e1718d/assets/3Dvis.png)
 
**8) Advanced 3D Features**  

**Blood Flow Visualization:**                                     

* Set desired heart rate (BPM)                     
* Click "Start Blood Flow"                   
* Watch animated blood flow through vessels                    
* Pulsatile flow in arteries, steady flow in veins                             

![](https://github.com/Medin-Egypt/SBME-MPR/blob/21d4fbf573044b48efb3c04d4e56cf3ee6622075/assets/gif%20converted/bloodflow.gif)

**Vessel Flythrough:**               

* Select a vessel or "Spine" from dropdown
* Click "Play" to start automatic flythrough
* Adjust velocity with slider
* Scrub through path with progress slider
* Camera follows vessel centerline automatically

![](https://github.com/Medin-Egypt/SBME-MPR/blob/21d4fbf573044b48efb3c04d4e56cf3ee6622075/assets/gif%20converted/flythrough.gif)


**Focus Navigation:**

* Enable "Focus Navigation" tool
* Right-click on any structure to isolate it
* Right-click again to restore all structures
* Perfect for examining specific anatomy

![](https://github.com/Medin-Egypt/SBME-MPR/blob/21d4fbf573044b48efb3c04d4e56cf3ee6622075/assets/gif%20converted/focusnav.gif)


**9) Crop and Export**
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
