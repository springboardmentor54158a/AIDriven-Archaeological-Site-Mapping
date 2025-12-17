**Name** : Shreyansh   
**Branch:**CSE AIML(FInal Year)

### **AI-Related Coursework / Projects / Experience**

**• Intel Unnati Industrial Training — AI/ML (2 Months)**  
 Completed a two-month intensive industrial training under the Intel Unnati Programme focused on Artificial Intelligence and Machine Learning. Gained hands-on experience with ML algorithms, model development workflows, and real-world problem-solving.

**• Software Bug Detection and Correction Model — Project**  
 Developed an AI-based model capable of identifying software bugs from code and suggesting corrections.

# Mile Stone 1 Dataset Collection and Preparation

## Week 1 Day 1

### Findings

Finding about Bhuvan dataset:

I explored the Bhuvan Geoportal by ISRO and found that it does not provide labeled archaeological datasets such as ruins or artifacts. However, it offers useful supporting data like satellite imagery, Digital Elevation Model , land use–land cover (LULC), soil and terrain information. This data is helpful for analyzing vegetation, terrain features, and predicting erosion-prone zones around archaeological sites. Therefore, Bhuvan is not suitable for direct training of AI models, but it is valuable for terrain analysis and conservation planning in our project.  Link: [https://bhuvan-app3.nrsc.gov.in/data/download/index.php](https://bhuvan-app3.nrsc.gov.in/data/download/index.php)

Questions  \- it is acceptable to create our own dataset by taking high-resolution screenshots from satellite platforms like Google Earth Pro or OpenAerialMap, and then manually annotating those images for ruins, vegetation, and artifact labeling.  
Is this method suitable and valid for our project dataset creation?

## Week 1 Day 2

## **1\. Understanding Key Classes: Ruins and Vegetation**

### **Ruins (Archaeological Ruins)- \-** Ruins refer to the remains of ancient human-built structures that have partially or fully decayed over time.

**How they appear in satellite/drone imagery:**

* Irregular stone outlines  
* Rectangular or circular foundations  
* Wall fragments  
* Depressions or raised mounds forming old room layouts  
* Patterns that do not occur naturally (straight lines, right angles)

**Vegetation :** Vegetation refers to all forms of plant cover including grass, shrubs, bushes, and trees that grow over or around archaeological sites.

**How they appear in satellite/drone imagery:**

* Irregular stone outlines  
* Rectangular or circular foundations  
* Wall fragments  
* Depressions or raised mounds forming old room layouts  
* Patterns that do not occur naturally (straight lines, right angles)


  
 **Image Quality Findings**

1. Some images contained clouds, shadows, or lighting variations that reduce clarity.

2. Low-resolution images (≥1m) made small artifacts difficult to detect.

3. High-resolution drone-like imagery (10–20 cm) was most effective.

4. Color inconsistencies and spatial distortions were common and require preprocessing**.**

## Week 1 Day 3

1\. Understanding of Semantic Segmentation Models

Researched how U-Net and DeepLabV3+ architectures work for pixel-level segmentation.

Identified which model is more suitable for small datasets (U-Net) and large/complex scenes (DeepLabV3+).

2\. Clarified Data Requirements for Ruins & Vegetation Segmentation

Understood that pixel-wise annotated masks are required for training.

Learned the mask format (0 \= background, 1 \= ruins, 2 \= vegetation).

Identified suitable annotation tools (LabelMe, CVAT, Labelbox).

Week 1 day 4

### **1\. Understood the Two Different AI Tasks Required**

I learned that my project involves **two completely separate computer vision tasks**, which need **different datasets and different annotation methods**:

1. **Semantic Segmentation** (U-Net / DeepLabV3+)

   * Goal: Pixel-wise labeling of **ruins** and **vegetation**

   * Output: A mask where each pixel belongs to a class

   * Dataset format: Images \+ Mask PNGs (same size)

2. **Object Detection & Classification** (YOLOv5 / Faster R-CNN)

   * Goal: Detect and classify **artifacts** (like pottery, coins, tools)

   * Output: Bounding boxes \+ class labels

###     **3\. Learned Annotation Requirements**

####         **A. Segmentation (Ruins & Vegetation)**

*  Use polygon or brush tools to create **pixel-wise masks**

* Classes:

  * 0 \= Background

  * 1 \= Ruins

  * 2 \= Vegetation

* Important annotation rules:

  * Maintain clear edges between ruins & vegetation

# Week 1 Day 5

## U-Net Architecture

## **1\. Understand Core Concepts**

* Read about **semantic segmentation** and its purpose.

* Understand what **encoder–decoder architectures** are.

* Learn the importance of **skip connections** in segmentation models.

* Study where U-Net is used (medical imaging, satellite, vegetation/ruins segmentation).

## **2\. Study U-Net Architecture**

* Review the **U-shaped structure**: Encoder → Bottleneck → Decoder.

# 

# Week 2 day 1

###  **Annotation Tool Setup & Planning**

* Selected a single annotation tool (**Labelbox / CVAT / Label Studio**) for uniformity

* Reviewed dataset and defined annotation guidelines

* Finalized annotation classes:

  * **Ruins**

  * **Vegetation**

  * **Artifacts (for object detection)**

* Planned export formats for segmentation and detection tasks

# Week 2 Day 2

### **Semantic Segmentation Annotation**

* Performed **polygon-based semantic segmentation** for:

  * Ruins

  * Vegetation

* Carefully annotated regions to ensure pixel-level accuracy

* Exported segmentation masks in **PNG format**

* Verified alignment between:

  * Original images

  * Corresponding masks

# Week 2 Day 3

### **Object Detection Annotation**

* Annotated visible artifacts using **bounding boxes**

* Ensured consistent labeling across images

* Exported annotations in:

  * **YOLO format**

  * **COCO format**

* Validated bounding boxes visually for correctness

# 

# Week 2 Day 4

### **Dataset Preprocessing**

* Resized all images to **512 × 512**

* Normalized pixel intensity values

* Organized dataset into:

  * Training set

  * Validation set

  * Test set

* Ensured correct pairing of:

  * Images

  * Segmentation masks

  * Detection annotations


Week 2 Day 5

 **Terrain Analysis Preparation & Model Study**

* Verified availability of **geolocation metadata** (where applicable)

* Ensured imagery has consistent:

  * Scale

  * Orientation

* Prepared dataset for future extraction of:

  * Slope maps

  * NDVI

  * Elevation / DEM

* Studied a reference Kaggle notebook:

  * **U-Net for Building Segmentation (PyTorch)**

  * [https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch](https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch)

* Understood segmentation pipeline and training workflow

# 

# Week 3 Day 1

## **Semantic Segmentation (U-Net)**

##  **Dataset Organization & Inputs Preparation**

* Organized dataset into required folder structure:

  * `images/`

  * `masks/`

* Verified correct mapping between:

  * Input images

  * Corresponding segmentation masks

* Checked class labels for:

  * Ruins

  * Vegetation
## Day 2: U-Net Model Setup

- Implemented **U-Net architecture** for semantic segmentation
- Integrated a **pretrained encoder** to improve feature extraction:
  - ResNet34
  - ResNet50
- Initialized encoder weights using **pretrained models** to achieve faster and more stable convergence during training






