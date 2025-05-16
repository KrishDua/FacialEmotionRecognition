# Facial Emotion Recognition Project

This project focuses on **detecting human emotions from facial expressions** using a **Convolutional Neural Network (CNN)**. It utilizes the **FER2013 dataset**, which contains thousands of labeled facial images annotated with one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The pipeline integrates **image preprocessing**, **model training**, and **performance evaluation** using TensorFlow/Keras with visualizations for insights.

---

## Project Overview

This project allows you to:

- Load and preprocess grayscale facial images (48x48 pixels)
- Normalize image data and encode emotion labels
- Train a CNN model to classify emotions
- Evaluate performance using metrics like accuracy and loss
- Visualize emotion distribution, training curves, and classification report
- Predict emotions on new images

---

## Features

- Data preprocessing from CSV format
- Label encoding for 7 emotions
- CNN model architecture for image classification
- Visualization of:
  - Emotion distribution
  - Training/validation loss and accuracy
  - Confusion matrix and classification heatmap
- Emotion prediction on unseen samples

---

## Technologies Used

### Core Libraries & Tools

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- scikit-learn
- Jupyter Notebook

---

## Model Architecture

- `Conv2D (64 filters)` + `ReLU` + `MaxPooling`
- `Conv2D (128 filters)` + `ReLU` + `MaxPooling`
- `Conv2D (256 filters)` + `ReLU` + `MaxPooling`
- `Flatten` layer
- `Dense (512)` + `Dropout`
- `Dense (7)` with Softmax output

**Loss Function**: Categorical Crossentropy  
**Optimizer**: Adam  
**Epochs**: 50 (with EarlyStopping)  
**Batch Size**: 64  
**Input Shape**: 48x48x1 (grayscale)

---

## Results

| Metric       | Value (Approx.) |
|--------------|------------------|
| Accuracy     | ~65–70%          |
| Loss         | Decreasing steadily with epochs
| Best Class   | Happy / Neutral (based on F1-score)
| Visualization | Included: Confusion matrix, classification report

> Model achieves reasonable emotion detection performance with basic CNN architecture.

---

## Dataset

- **Source**: [FER2013 - Facial Expression Recognition Challenge](https://www.kaggle.com/datasets/msambare/fer2013)
- **Format**: CSV with pixel values as strings
- **Classes**:
  - 0: Angry
  - 1: Disgust
  - 2: Fear
  - 3: Happy
  - 4: Sad
  - 5: Surprise
  - 6: Neutral
- **Image Size**: 48x48 pixels (grayscale)
- **Train/Validation/Test Split**: Predefined in CSV

---

## APIs / Integrations

- TensorFlow/Keras – Model building and training  
- Matplotlib / Seaborn – Plots and heatmaps  
- scikit-learn – Evaluation metrics and classification report

---

## Author

**Krish Dua**  
[Portfolio Website](https://krishdua.vercel.app) | [LinkedIn Profile](https://www.linkedin.com/in/krish-dua-9202a4272/)

---
