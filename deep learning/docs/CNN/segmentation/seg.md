# Image Segmentation — Complete Interview Guide

After **classification** and **object detection**, the next major computer vision task is **image segmentation**.

Segmentation answers:


Which pixels belong to which object?


Instead of predicting a **bounding box**, segmentation predicts a **label for every pixel**.

---

# 1. Classification vs Detection vs Segmentation

## Image Classification

Input:


Image


Output:


Cat


Only predicts **object category**.

---

## Object Detection

Input:


Image


Output:


Cat → bounding box
Dog → bounding box


Predicts:


object class + bounding box


---

## Image Segmentation

Input:


Image


Output:


Pixel-level labels


Example:


Cat pixels → class "cat"
Background pixels → class "background"


Segmentation produces a **mask** over the image.

---

# 2. Types of Segmentation

There are two main types.

---

# Semantic Segmentation

Semantic segmentation classifies **every pixel** into a class.

Example:


Image

Person Person Person
Person Person Person
Car Car Car
Car Car Car


All pixels belonging to a class share the **same label**.

Important detail:


Different objects of the same class are NOT separated


Example:

Two persons → both labeled **person**.

---

# Instance Segmentation

Instance segmentation distinguishes **different objects of the same class**.

Example:


Person1 pixels
Person2 pixels
Person3 pixels


Each object gets its **own mask**.

Example models:


Mask R-CNN


---

# 3. Why Segmentation is Important

Segmentation is used when **precise object boundaries matter**.

Applications:


Medical imaging
Self-driving cars
Satellite imagery
Agriculture
Robotics


Example:


Tumor segmentation
Road segmentation
Building segmentation


Bounding boxes are not precise enough.

---

# 4. Segmentation Output

Segmentation produces a **mask**.

Example image:


Cat image


Output mask:


1 1 1 1
1 1 1 1
0 0 0 0


Where:


1 → object
0 → background


For multi-class segmentation:


0 → background
1 → road
2 → car
3 → pedestrian


---

# 5. U-Net Architecture (Very Important)

U-Net is the **most famous segmentation architecture**, widely used in **medical imaging**.

It has a **U-shaped architecture**.

---

# U-Net Structure


Input
↓
Encoder (downsampling)
↓
Bottleneck
↓
Decoder (upsampling)
↓
Segmentation mask


---

# Encoder (Contracting Path)

Encoder extracts features while reducing spatial size.

Typical operations:


Conv
ReLU
Conv
ReLU
Max Pool


Example:


256×256 → 128×128 → 64×64 → 32×32


Feature depth increases.

---

# Decoder (Expanding Path)

Decoder restores spatial resolution.

Operations:


Upsampling
Convolution


Example:


32×32 → 64×64 → 128×128 → 256×256


---

# Skip Connections in U-Net

U-Net uses skip connections between encoder and decoder.

Diagram:


Encoder feature map
↓
Skip connection
↓
Decoder layer


Purpose:


preserve spatial details
recover fine boundaries


Without skip connections:


segmentation becomes blurry


---

# 6. Visual U-Net Architecture


Input
↓
Conv
↓
Conv
↓
Pool
↓
Conv
↓
Pool
↓
Bottleneck
↓
Upsample
↓
Concat (skip connection)
↓
Conv
↓
Upsample
↓
Concat
↓
Conv
↓
Output mask


Shape resembles the letter:


U


Hence the name **U-Net**.

---

# 7. Loss Functions for Segmentation

Segmentation requires special loss functions.

---

## Cross-Entropy Loss

Used for multi-class segmentation.

---

## Dice Loss

Common in medical imaging.

Formula:


Dice = (2 × intersection) / (prediction + ground truth)


Measures overlap between masks.

---

## IoU Loss

Based on **Intersection over Union**.

Used for segmentation quality.

---

# 8. Evaluation Metrics

Important segmentation metrics:


Pixel accuracy
IoU
Dice score


Example:


IoU = overlap / union


Higher IoU means better segmentation.

---

# 9. Interview Explanation

**What is image segmentation?**

Image segmentation is a computer vision task where each pixel in an image is assigned a class label, allowing precise localization of objects.

---

# 10. Common Interview Questions

## Q1

Difference between semantic and instance segmentation?

Answer:

Semantic segmentation labels each pixel by class, while instance segmentation distinguishes different objects of the same class.

---

## Q2

Why does U-Net use skip connections?

Answer:

Skip connections combine low-level spatial information from the encoder with high-level features in the decoder, improving boundary accuracy.

---

## Q3

Why is segmentation harder than detection?

Answer:

Segmentation requires pixel-level predictions instead of bounding boxes.

---

# 11. Computer Vision Tasks Summary

You now understand the three major CV tasks.


Image Classification
Object Detection
Image Segmentation


---

# 12. Models Covered

You now understand the **most important CNN models for interviews**.


LeNet
AlexNet
ResNet
U-Net