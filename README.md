# Rice Leaf Disease Prediction

This project focuses on predicting rice leaf diseases using image data. By leveraging machine learning models and transfer learning, the goal is to classify three types of rice leaf diseases: **Bacterial Leaf Blight**, **Brown Spot**, and **Leaf Smut**.

---

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architectures](#model-architectures)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Transfer Learning](#transfer-learning)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Improvements](#future-improvements)

---

## Installation

Ensure you have Python installed on your system. Install the required libraries using the command below:

```bash
pip install -r requirements.txt
```

### Required Libraries
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `keras-tuner`
- `Pillow`
- `opencv-python`
- `scikit-learn`

---

## Dataset

The dataset contains images of rice leaves classified into three categories:
1. **Bacterial Leaf Blight**
2. **Brown Spot**
3. **Leaf Smut**

Images are organized into subfolders for each class. Ensure the dataset structure follows:

```
Dataset/
├── Bacterial leaf blight/
├── Brown spot/
└── Leaf smut/
```

---

## Model Pipeline

### Data Preprocessing
1. Images are loaded, resized to `(256, 256)`, and converted to NumPy arrays.
2. Data is normalized by dividing pixel values by 255.
3. Labels are encoded using `LabelEncoder`.
4. Data is split into training and testing sets (80:20).

### Model Architectures

#### CNN Model
A custom Convolutional Neural Network (CNN) was built with the following layers:
- 3 Convolutional layers with ReLU activation.
- MaxPooling layers after each convolution.
- Dropout layers to reduce overfitting.
- Fully connected Dense layers with a Softmax output layer for classification.

#### Hyperparameter Tuning with Keras Tuner
- Tunable parameters include:
  - Number of Dense units.
  - Dropout rate.
  - Learning rate.
- Best parameters are chosen based on validation accuracy.

#### Transfer Learning with ResNet50
- Pre-trained ResNet50 model (without top layers) is used as a feature extractor.
- Added a Global Average Pooling layer, Dense layer, and Dropout for final classification.

---

## Results

- **CNN Model**
  - Test Accuracy: ~88%
  - Confusion Matrix: Displayed for evaluation.

- **Transfer Learning with ResNet50**
  - Test Accuracy: ~92%
  - Confusion Matrix: Displayed for evaluation.

- **Best Hyperparameters** (from Keras Tuner):
  - Units: 256
  - Dropout: 0.4
  - Learning Rate: 0.001

---

## Visualizations

### Sample Images
Displayed a grid of sample images from the dataset, including 15 samples per class.

### Accuracy and Loss Curves
- Training vs. Validation Accuracy.
- Training vs. Validation Loss.

### Predictions
Randomly selected test images are displayed with predicted and true labels.

---

## Future Improvements
1. Expand dataset to include more samples.
2. Experiment with other architectures like EfficientNet or MobileNet.
3. Implement real-time predictions using a web interface.
4. Augment data further for better generalization.

