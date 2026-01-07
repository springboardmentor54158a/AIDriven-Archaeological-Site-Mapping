**Name** : Shreyansh   
**Branch:**CSE AIML(FInal Year)

### **AI-Related Coursework / Projects / Experience**

**• Intel Unnati Industrial Training — AI/ML (2 Months)**  
 Completed a two-month intensive industrial training under the Intel Unnati Programme focused on Artificial Intelligence and Machine Learning. Gained hands-on experience with ML algorithms, model development workflows, and real-world problem-solving.

**• Software Bug Detection and Correction Model — Project**  
 Developed an AI-based model capable of identifying software bugs from code and suggesting corrections.

## Mile stone 1: Completed

## Mile stone2: train images-120 test images-30

    iou score- 0.5260  
    Dice score-0.5133

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

  # 

# Week 3 day 2

# **U-Net Model Setup**

* Implemented **U-Net architecture** for semantic segmentation

* Integrated a **pretrained encoder**:

  * ResNet34 / ResNet50

* Initialized encoder weights using pretrained models for better convergence

# Week 3 day 3

###  **Work Done:**

* Studied the concept of **semantic segmentation** and its difference from object detection.

* Understood the role of **pixel-wise classification** in archaeological site mapping.

* Analyzed the dataset structure consisting of:

  * `images/` → satellite images

  * `masks/` → corresponding ground-truth segmentation masks

* Verified image–mask alignment and ensured consistent file naming.

### **🔹 Learning Outcome:**

* Gained clarity on how segmentation masks act as labels.

# Week 3 day 4

* Studied **U-Net architecture**, including:

  * Encoder–decoder structure

  * Skip connections

* Explored **DeepLabV3+** and its advantages in handling multi-scale context.

* Compared both models and selected **U-Net** for initial experimentation due to simplicity and efficiency.

### **🔹 Learning Outcome:**

* Understood why U-Net is widely used for satellite image segmentation.

* Learned the importance of encoder backbones in deep learning models.

# Week 3 day 5

### **Work Done:**

* Trained the segmentation model on the training dataset.

* Monitored training and validation loss.

* Visualized predicted segmentation masks against ground truth.

* Adjusted hyperparameters such as learning rate and batch size for stable training.

### **🔹 Learning Outcome:**

* Learned how to detect underfitting and overfitting.

* Gained experience in interpreting segmentation outputs visually.

# Week 4 Yolo check Point 

## Week 4 day 1

### Understanding Object Detection & Dataset Preparation

### Objectives

* To understand the fundamentals of object detection

* To study the YOLOv5 architecture and workflow

* To prepare a YOLO-compatible dataset structure

* To distinguish between semantic segmentation and object detection

### Learning Outcomes

* Gained an understanding of how YOLO performs real-time object detection

* Learned the difference between pixel-level segmentation and bounding-box-based detection

* Understood the YOLO annotation format and dataset requirements

### Concepts Covered

* Object Detection vs Semantic Segmentation

* YOLOv5 pipeline

* Bounding Box parameters:

  * `x_center`

  * `y_center`

  * `width`

  * `height` (all normalized)

Defined object detection classes:

 `0 → Structure`  
`1 → Wall`

Created YOLO dataset directory structure:

 `yolo_data/`  
`├── images/`  
`│   ├── train/`  
`│   └── val/`  
`├── labels/`  
`│   ├── train/`  
`│   └── val/`

Identified that existing .png masks are segmentation labels  
 and YOLO requires bounding box annotations (.txt).

**Week 4 day 2 : Annotation & YOLO Configuration**

### **Objectives**

* Create YOLO-compatible annotation files

* Prepare configuration files for training

* Set up YOLOv5 environment in Google Colab

### **Learning Outcomes**

* Manual bounding box annotation process

* Writing YOLO label files

* Creating and validating a dataset YAML file

### **Concepts Covered**

YOLO annotation format:

 `class_id x_center y_center width height`

* Importance of normalized coordinates

* Train/Validation split strategy

# Week 4 Day 3

##  YOLO Training, Evaluation, and Results

### **Objectives**

* To train the YOLOv5 model on annotated satellite imagery

* To evaluate model performance using standard metrics

* To visualize detection results

### **Learning Outcomes**

* Gained experience in training a YOLOv5 object detection model

* Understood how to interpret loss values and evaluation metrics

* Learned to analyze detection results visually

Trained the YOLOv5 model using the prepared dataset with the following parameters:

* Image size: 640 × 640

* Batch size: 2

* Epochs: 10–20

Monitored training progress through loss values and mAP scores.

Evaluated the trained model on validation images.

Performed inference to visualize bounding boxes and confidence scores on detected archaeological structures.  
 [https://drive.google.com/drive/folders/1gBeCxVyhRdQPSs7sygmdZ89h\_oD23nxH?usp=sharing](https://drive.google.com/drive/folders/1gBeCxVyhRdQPSs7sygmdZ89h_oD23nxH?usp=sharing)

# 

# Week 4 Day 4

### **Objective**

The main objective of Week 4 was to implement an object detection model for identifying and classifying archaeological artifacts from images, and to evaluate its performance using standard evaluation metrics such as **mAP**, **precision**, and **recall**.

---

### **Work Done**

#### **1\. Model Selection**

For artifact detection and classification, I implemented **YOLOv5**, a state-of-the-art real-time object detection model.  
 YOLOv5 was selected due to:

* High detection speed

* Good accuracy on small and medium objects

* Easy integration with custom datasets

* Built-in support for evaluation metrics like mAP

*(Alternatively, Faster R-CNN was studied conceptually for comparison, but YOLOv5 was used for implementation.)*

---

#### **2\. Dataset Preparation**

Images were annotated in **YOLO format**, where each object is represented by:

 `class_id x_center y_center width height`

*   
* The dataset was split into:

  * **Training set**

  * **Validation set**

* A custom `data.yaml` file was created specifying:

  * Number of classes

  * Class names

  * Paths to training and validation images

---

#### **3\. Model Training**

* YOLOv5 was trained on the custom archaeological artifact dataset.

* Training involved:

  * Loading pretrained weights for faster convergence

  * Fine-tuning on artifact images

  * Monitoring training loss and validation loss

* The model learned to detect and classify different artifact categories present in the dataset.

---

#### **4\. Model Evaluation**

After training, the model was evaluated using standard object detection metrics:

##### **a) Mean Average Precision (mAP)**

* **mAP@0.5** was used to measure overall detection accuracy.

* This metric evaluates how well the predicted bounding boxes overlap with ground truth boxes.

##### **b) Precision**

* Precision measures how many detected artifacts were actually correct.

* High precision indicates fewer false positives.

##### **c) Recall**

* Recall measures how many actual artifacts were successfully detected.

* High recall indicates fewer missed detections.

##### **d) Class-wise Evaluation**

* Precision and recall were analyzed **for each artifact class separately**.

* This helped identify which artifact categories were detected well and which required more data or tuning.

---

#### **5\. Results and Observations**

* The model achieved satisfactory mAP on the validation dataset.

* Certain artifact classes showed higher precision and recall due to:

  * Better visual features

  * More training samples

* Some classes required further data balancing and annotation refinement.

# Milestone 3: Terrain Erosion Prediction (Weeks 5–6) 

# Week 5 Day 1

## **Week 5 – Day 1: Theoretical Study of Terrain Erosion**

### **Objective**

The objective of Day 1 was to study the basic concept of terrain erosion and understand its relevance in archaeological site mapping and terrain analysis.

---

### **Work Done**

* Studied the fundamentals of **terrain erosion** and its causes.

* Understood how natural factors such as:

  * Rainfall

  * Wind

  * Gravity

  * Surface runoff  
     contribute to soil erosion.

* Learned the difference between **erosion-prone areas** and **stable terrain**.

* Studied the impact of erosion on:

  * Surface artifacts

  * Archaeological site visibility

  * Long-term site preservation.

* Gained theoretical knowledge about why erosion prediction is important before applying machine learning models.

---

# Week 5 Day 2

### **Objective**

The objective of Day 2 was to study and understand the terrain features that are commonly used to analyze and predict erosion-prone and stable areas.

---

### **Work Done**

* Studied the role of **terrain features** in erosion analysis.

* Gained theoretical understanding of the following important features:

#### **1\. Slope**

* Learned how slope affects soil stability.

* Steeper slopes are generally more prone to erosion due to increased gravitational force and surface runoff.

#### **2\. Elevation**

* Studied how elevation influences water flow and erosion patterns.

* Higher elevation areas may experience faster runoff, increasing erosion risk.

#### **3\. Vegetation Cover**

* Learned the importance of vegetation in holding soil together.

* Areas with dense vegetation are generally more stable and less prone to erosion.

#### **4\. Surface Runoff**

* Studied how water movement across terrain contributes to erosion.

* Increased runoff leads to higher soil displacement.

* Understood that combining multiple terrain features gives better erosion prediction than using a single feature.

---

### **Outcome**

By the end of Day 2, a clear theoretical understanding of key terrain features influencing erosion was developed, forming the foundation for future data preparation and model implementation.

# 

# Week 5 day 3

### **Objective**

The objective of Day 3 was to study the theoretical approach for preparing labeled datasets to distinguish between erosion-prone and stable terrain areas.

---

### **Work Done**

* Studied the concept of **data labeling** in the context of terrain erosion prediction.

* Learned how terrain regions can be categorized into:

  * **Erosion-prone areas**

  * **Stable areas**

* Understood that labeling is based on terrain characteristics such as:

  * Slope steepness

  * Vegetation density

  * Elevation variation

  * Historical erosion patterns (theoretical study)

* Studied how labeled data is essential for training supervised machine learning models.

* Learned the importance of **accurate and consistent labeling** to avoid biased or incorrect predictions.

* Understood that expert knowledge or reference maps are often used for labeling erosion-prone regions.

---

### **Outcome**

By the end of Day 3, a clear theoretical understanding of how labeled datasets are prepared for erosion prediction was developed, which will support practical implementation in later stages of the project.

# Week 5 Day 4

### **Objective**

The objective of Day 4 was to study, at a theoretical level, how terrain features are extracted from geospatial and remote sensing data for erosion prediction.

---

### **Work Done**

* Studied the concept of **feature extraction** in terrain analysis.

* Learned how terrain features can be derived from:

  * Digital Elevation Models (DEM)

  * Satellite imagery

* Theoretically studied extraction of the following features:

  * **Slope and aspect** from elevation data

  * **Elevation gradients** for understanding terrain variation

  * **Vegetation indices** (such as NDVI) from satellite images

* Understood the role of feature extraction in converting raw terrain data into machine learning–ready inputs.

---

### **Outcome**

By the end of Day 4, a theoretical understanding of terrain feature extraction methods used for erosion prediction was developed.

# Week 5 Day 5

### **Objective**

The objective of Day 5 was to study different approaches used for predicting terrain erosion using computational and machine learning techniques.

---

### **Work Done**

* Studied traditional and modern approaches for erosion prediction, including:

  * Rule-based and threshold-based methods

  * Machine learning–based classification approaches

* Understood how extracted terrain features are used as inputs to prediction models.

* Studied the concept of **binary classification** for erosion prediction:

  * Erosion-prone areas

  * Stable areas

* Learned about evaluation criteria (theoretical overview) for erosion prediction models such as accuracy, precision, and recall.

* Understood the importance of validating erosion prediction results for real-world terrain analysis.

---

### **Outcome**

By the end of Day 5, a clear theoretical understanding of erosion prediction methodologies and their role in AI-driven terrain analysis was achieved.

# **Week 6: Theoretical Study of Machine Learning for Erosion Prediction**

---

## **Week 6 – Day 1: Study of Machine Learning in Terrain Erosion Prediction**

### **Objective**

The objective of Day 1 was to study the role of machine learning in predicting terrain erosion.

### **Work Done**

* Studied how machine learning can be applied to environmental and terrain analysis.

* Understood the difference between:

  * Classification-based erosion prediction

  * Regression-based erosion prediction

* Learned why erosion prediction is treated as a regression problem in many cases.

### **Outcome**

Developed a basic theoretical understanding of applying machine learning techniques to erosion prediction.

## **Week 6 – Day 2: Study of Random Forest Model**

### **Objective**

The objective of Day 2 was to study the Random Forest algorithm and its suitability for terrain erosion prediction.

### **Work Done**

* Studied the working principle of **Random Forest**.

* Learned how multiple decision trees are combined to improve prediction accuracy.

* Understood why Random Forest performs well with non-linear terrain data.

* Studied advantages such as reduced overfitting and robustness to noise.

### **Outcome**

Gained theoretical understanding of Random Forest as a potential model for erosion prediction.

## **Week 6 – Day 3: Study of XGBoost Model**

### **Objective**

The objective of Day 3 was to study the XGBoost algorithm for terrain erosion prediction.

### **Work Done**

* Studied the concept of **gradient boosting**.

* Learned how XGBoost improves model performance by correcting previous errors.

* Understood why XGBoost is effective for structured terrain feature data.

* Studied theoretical advantages such as high accuracy and scalability.

### **Outcome**

Developed theoretical knowledge of XGBoost for erosion prediction.

