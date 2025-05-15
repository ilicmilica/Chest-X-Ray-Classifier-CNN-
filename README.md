# Chest-X-Ray-Classifier-CNN-
## Overview
The goal of this project is to develop a Convolutional Neural Network (CNN) that can classify chest X-ray images as either Normal or Pneumonia. This deep learning model supports early diagnosis by automating the interpretation of medical imagery, potentially assisting radiologists in clinical settings.

The project includes end-to-end steps: loading and preprocessing image data, building and training a CNN, evaluating its performance on unseen data, and visualizing predictions to understand model behavior.

## Dataset Description
The dataset used is the Chest X-Ray Images (Pneumonia) from Kaggle.

It contains chest X-ray images divided into:

Normal (0) — Clear lungs with no sign of infection

Pneumonia (1) — Lungs infected with pneumonia

Images are resized to 224x224 and normalized for model training.

## Key Insights:
### 1. Data Loading & Preprocessing
Used TensorFlow's image_dataset_from_directory to efficiently load and batch images

Applied real-time shuffling and batching for better generalization

Normalized pixel values using a Rescaling layer

### 2. CNN Architecture
A custom CNN was built with:

3 convolutional layers (with increasing filters: 32 → 64 → 128)

Max-pooling after each conv layer

Dense + dropout layers for final classification

Output layer predicts 2 classes (normal, pneumonia)

### 3. Model Training & Validation

Trained the model with Adam optimizer and Sparse Categorical Crossentropy

Added EarlyStopping to prevent overfitting

### 4. Performance Metrics
Accuracy: Achieved up to XX% accuracy on test set

Balanced performance across both classes with minimal overfitting

## Visual Results
Sample predictions with true and predicted labels:

Confusion Matrix:
![image](https://github.com/user-attachments/assets/2f7dd6ba-615c-4f3c-8621-d9c0ce72aa55)


## Conclusion
This project demonstrates a practical deep learning pipeline for medical image classification using chest X-rays. By using CNNs in TensorFlow, the model reliably distinguishes between normal and pneumonia cases with high accuracy.
