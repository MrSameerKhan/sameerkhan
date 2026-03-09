# Object Detection — Complete Interview Guide

After **image classification**, the next important computer vision task is **object detection**.

Image classification answers:


What object is in the image?


Object detection answers:


What objects are present and where are they located?


Example:


Image → Dog + Cat + Person


Detection result:


Dog → bounding box
Cat → bounding box
Person → bounding box


Object detection predicts **two things simultaneously**:

1. Object class
2. Bounding box location

---

# Learning Structure

We will understand detection in this order:

1. Classification vs Detection  
2. Bounding boxes  
3. IoU (Intersection over Union)  
4. Two-stage detectors  
5. One-stage detectors  
6. Faster R-CNN  
7. YOLO  
8. Interview questions  

---

# 1. Classification vs Detection

## Image Classification

Input:


Image


Output:


Dog


The model only predicts the **object category**.

---

## Object Detection

Input:


Image


Output:


Dog → bounding box
Cat → bounding box
Person → bounding box


Detection outputs both:


class label
bounding box coordinates


---

# 2. Bounding Box

A bounding box is a rectangle surrounding an object.

Typical representation:


(x, y, width, height)


or


(x1, y1, x2, y2)


Example:


Top-left corner: (x1, y1)
Bottom-right corner: (x2, y2)


Bounding boxes allow the model to **localize objects**.

---

# 3. IoU (Intersection over Union)

IoU measures how well the predicted bounding box matches the ground truth.

Formula:


IoU = Area of Overlap / Area of Union


Example:


Prediction box
Ground truth box


Overlap area:


common region


Union area:


total area of both boxes


IoU value ranges:


0 → no overlap
1 → perfect overlap


Typical threshold:


IoU ≥ 0.5 → correct detection


---

# 4. Two-Stage Detectors

Two-stage detectors work in two steps:


Step 1 → propose possible object regions
Step 2 → classify those regions


Pipeline:


Image
↓
Region proposals
↓
CNN feature extraction
↓
Object classification


Examples:


R-CNN
Fast R-CNN
Faster R-CNN


These models are **very accurate but slower**.

---

# 5. One-Stage Detectors

One-stage detectors predict bounding boxes **directly**.

Pipeline:


Image
↓
CNN
↓
Direct bounding box prediction


Examples:


YOLO
SSD
RetinaNet


Advantages:


much faster
real-time detection


---

# 6. Faster R-CNN

Faster R-CNN improved earlier detection models by introducing:


Region Proposal Network (RPN)


Architecture:


Image
↓
CNN backbone (ResNet/VGG)
↓
Feature map
↓
Region Proposal Network
↓
Region proposals
↓
ROI pooling
↓
Classifier + bounding box regressor


Steps:

1. CNN extracts feature maps
2. RPN proposes candidate regions
3. Regions are classified and refined

Advantages:


high accuracy


Disadvantages:


slower than one-stage detectors


---

# 7. YOLO (You Only Look Once)

YOLO is a **one-stage detector**.

Key idea:


Detection as a single regression problem


Pipeline:


Image
↓
CNN
↓
Grid cells
↓
Predict bounding boxes + class probabilities


The image is divided into a grid:


S × S grid


Each cell predicts:


bounding boxes
confidence score
class probabilities


Advantages:


real-time detection
very fast


YOLO is widely used in:


autonomous driving
surveillance
robotics


---

# 8. Faster R-CNN vs YOLO

| Model | Type | Speed | Accuracy |
|------|------|------|------|
| Faster R-CNN | Two-stage | slower | very high |
| YOLO | One-stage | very fast | slightly lower |

---

# 9. Visual Pipeline Summary

Object detection pipeline:


Image
↓
CNN feature extractor
↓
Bounding box prediction
↓
Class prediction


Final output:


class + location


---

# 10. Interview Explanation

**What is object detection?**

Object detection is a computer vision task where the model identifies objects in an image and predicts their locations using bounding boxes along with class labels.

---

# 11. Most Common Interview Questions

### Q1

What is IoU?

Answer:

Intersection over Union measures how much the predicted bounding box overlaps with the ground truth box.

---

### Q2

Difference between classification and detection?

Answer:

Classification predicts the object category, while detection predicts both object category and bounding box location.

---

### Q3

Difference between YOLO and Faster R-CNN?

Answer:

YOLO is a one-stage detector that predicts objects directly, while Faster R-CNN is a two-stage detector that first proposes regions and then classifies them.

---

# 12. Detection Pipeline Summary

You now understand the basics of object detection:


Bounding Boxes
IoU
Two-stage detectors
One-stage detectors
Faster R-CNN
YOLO


---
