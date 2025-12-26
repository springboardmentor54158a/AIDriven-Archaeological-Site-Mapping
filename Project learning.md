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

## Week 2 Day 4

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

# Week 3: Semantic Segmentation (U-Net / DeepLabV3+)

## Week 3 Day 1: Problem Understanding & Dataset Analysis
- Studied the concept of **semantic segmentation** and its importance in archaeological site mapping.
- Understood the difference between **object detection** and **semantic segmentation**.
- Analyzed the dataset structure containing:
  - `images/` – satellite/drone images
  - `masks/` – pixel-wise annotated segmentation masks
- Verified that each image has a corresponding mask with proper alignment.
- Identified target classes such as **ruins** and **vegetation**.

**Learning Outcome:**
- Gained understanding of pixel-level labeling and its role in identifying archaeological features.

---

## Week 3 Day 2: Study of Segmentation Models (U-Net & DeepLabV3+)
- Studied the architecture of **U-Net**, focusing on encoder–decoder design and skip connections.
- Explored **DeepLabV3+** and its use of atrous convolutions for multi-scale context.
- Compared both models in terms of complexity and performance.
- Selected **U-Net** for initial implementation due to its efficiency and suitability for small datasets.

**Learning Outcome:**
- Learned how different segmentation architectures handle spatial information.

---

## Week Day 3: Model Implementation & Transfer Learning
- Implemented a **U-Net-based semantic segmentation model**.
- Integrated a **pretrained encoder (ResNet34 / ResNet50)** to leverage transfer learning.
- Fine-tuned the model for **ruins and vegetation classes**.
- Configured training parameters such as:
  - Loss functions (Binary Cross Entropy / Dice Loss)
  - Optimizer (Adam)
  - Input image resizing and normalization

**Learning Outcome:**
- Understood the benefits of transfer learning in improving segmentation performance.

---

## Week 3 Day 4: Model Training & Validation
- Trained the segmentation model using the prepared dataset.
- Monitored training and validation loss across epochs.
- Performed qualitative analysis by visualizing predicted masks against ground truth.
- Tuned hyperparameters like learning rate and batch size to improve results.

**Learning Outcome:**
- Learned how to evaluate training behavior and improve model stability.

---

## Week3 Day 5: Model Evaluation & Performance Analysis
- Evaluated the trained model using standard segmentation metrics:
  - **Intersection over Union (IoU)**
  - **Dice Score**
- Analyzed segmentation results for ruins and vegetation regions.
- Identified strengths and limitations of the model predictions.
- Documented evaluation results for reporting and presentation.

**Learning Outcome:**
- Gained practical understanding of segmentation evaluation metrics and their significance.

---
# 📘 Week 4: YOLO Checkpoint

---

## 📅 Week 4 – Day 1  
### Understanding Object Detection & Dataset Preparation

### 🎯 Objectives
- To understand the fundamentals of object detection  
- To study the YOLOv5 architecture and workflow  
- To prepare a YOLO-compatible dataset structure  
- To distinguish between semantic segmentation and object detection  

---

### 📚 Learning Outcomes
- Gained an understanding of how YOLO performs real-time object detection  
- Learned the difference between pixel-level segmentation and bounding-box-based detection  
- Understood the YOLO annotation format and dataset requirements  

---

### 🧠 Concepts Covered
- Object Detection vs Semantic Segmentation  
- YOLOv5 pipeline  
- Bounding box parameters:
  - `x_center`
  - `y_center`
  - `width`
  - `height`  
  *(all values normalized)*  

---

### 🛠 Practical Work
- Defined object detection classes:
  - **0 → Structure**
  - **1 → Wall**

- Created YOLO dataset directory structure:
https://drive.google.com/drive/folders/1gBeCxVyhRdQPSs7sygmdZ89h_oD23nxH?usp=sharing


- Identified that existing `.png` files were segmentation masks and that YOLO requires bounding box annotations in `.txt` format.

---

## 📅 Week 4 – Day 2  
### Annotation & YOLO Configuration

### 🎯 Objectives
- To create YOLO-compatible annotation files  
- To prepare configuration files for YOLO training  
- To set up the YOLOv5 environment in Google Colab  

---

### 📚 Learning Outcomes
- Learned the manual bounding box annotation process  
- Understood how to write YOLO label files  
- Learned how to create and validate a dataset YAML configuration file  

---

### 🧠 Concepts Covered
- YOLO annotation format:
- Importance of normalized bounding box coordinates  
- Train–Validation dataset split strategy  

---

## 📅 Week 4 – Day 3  
### YOLO Training, Evaluation, and Results

### 🎯 Objectives
- To train the YOLOv5 model on annotated satellite imagery  
- To evaluate model performance using standard metrics  
- To visualize object detection results  

---

### 📚 Learning Outcomes
- Gained experience in training a YOLOv5 object detection model  
- Understood how to interpret loss values and evaluation metrics  
- Learned to analyze detection results visually  

---

### Week 4 Day 4
- Trained the YOLOv5 model using the prepared dataset with the following parameters:
- **Image size:** 640 × 640  
- **Batch size:** 2  
- **Epochs:** 10–20  

- Monitored training progress through loss values and **mAP** scores  
- Evaluated the trained model on validation images  
- Performed inference to visualize bounding boxes and confidence scores on detected archaeological structures  

---

## 🏁 Week 4 Summary
> Week 4 focused on implementing YOLOv5 for archaeological artifact detection, covering dataset preparation, annotation, training, evaluation, and result visualization.







