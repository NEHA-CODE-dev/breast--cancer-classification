# Breast Cancer Classification using Linear SVM

## Project Overview

This project uses a Machine Learning approach to classify breast cancer tumors as **Benign (B)** or **Malignant (M)** using a **Linear Support Vector Machine (SVM)** classifier.

The goal of this project is to build a model that can learn patterns from tumor features and predict whether a tumor is cancerous or non-cancerous.

---

## Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

The dataset contains:
- **569 samples**
- **30 numerical features**
- **1 target variable (Diagnosis)**

### Target Classes:
- **B** → Benign (Non-cancerous)
- **M** → Malignant (Cancerous)

### Features include:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension
- and other tumor measurements

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Machine Learning Workflow

1. Data Loading
2. Data Exploration and Cleaning
3. Removing unnecessary columns (`id`, `Unnamed` columns)
4. Separating features and target variable
5. Splitting data into training and testing sets
6. Feature Scaling using StandardScaler
7. Training Linear SVM model
8. Model Evaluation
9. Making predictions on new data

---

## Model Used

### Linear Support Vector Machine (SVM)

SVM is a supervised machine learning algorithm that finds the best decision boundary to separate different classes.

Linear SVM was selected because it performs well on high-dimensional numerical datasets after feature scaling.

---

## Model Performance

The Linear SVM model achieved:

- **Accuracy: 96%**

### Classification Report:

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Benign (B) | 0.99 | 0.95 | 0.97 |
| Malignant (M) | 0.91 | 0.97 | 0.94 |

The model achieved a **97% recall for malignant cases**, which is important because detecting cancer cases correctly is a critical part of medical classification.

---

## Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report