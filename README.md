# 🧠 MNIST Handwritten Digit Recognizer using Deep CNN

## 👥 Contributors
- Sejal Raj (055041)
- Vidushi Rawat (055056)

---

## 📌 Problem Statement
Handwritten digit recognition is essential for automating tasks in various domains, but traditional methods face difficulties due to the *inherent variability of handwriting. This project leverages a **Deep Convolutional Neural Network (CNN)* model to overcome these limitations. By using deep learning, the system accurately classifies handwritten digits, reducing errors and manual effort in real-world applications.

---

## 📁 Project Structure
1. Importing Libraries
2. Preparing the Dataset
3. Model Building
4. Model Fitting
5. Model Analysis
6. Predicting using test data

##Links for dataset
1. Training dataset : https://drive.google.com/file/d/1dqjEk1bza1L12-IQYOCs0Rbc9IHmqr3z/view?usp=sharing
2. Test dataset : https://drive.google.com/file/d/1YiDZfgRO89NWqQWySf2Fq3R779LemviE/view?usp=sharing

---

## 🔍 Data Analysis
### 1. Importing Libraries
- *TensorFlow v2*: Open-source machine learning framework by Google.
- *Keras*: High-level neural network API running on top of TensorFlow.
- *Pandas and Matplotlib*: For data manipulation and visualization.

### 2. Preparing the Dataset
- *Dataset:* MNIST Handwritten Digit Recognition Dataset
- *Data Loading:* read_csv() loads the dataset into a Pandas DataFrame.
- *Target Variable:* Labels stored in Y_srvr41560428train.
- *Input Features:* Pixel values stored in X_srvr4156train.

#### 2.1. Normalization
- Pixel values (0-255) are scaled to the range [0,1] by dividing by 255.
- This improves model convergence speed and stabilizes gradient updates.

#### 2.2. Reshaping
- The dataset is reshaped into (28,28,1) matrices to match the CNN input dimensions.

#### 2.3. Encoding
- Labels are *one-hot encoded* to match the CNN's output vector format, enabling multi-class classification.

#### 2.4. Train-Test Split
- The data is split into *training* and *validation sets*.
- Ensures the model generalizes effectively and prevents overfitting.

---

## 🚀 Model Architecture
- *LeNet-5 Inspired Architecture:*
  - Input → [[Conv2D → ReLU] × 2 → MaxPool2D → Dropout] × 2 → Flatten → Dense → Dropout → Output
- *Data Augmentation:*
  - Expands dataset using transformations (zooming, rotating, flipping, cropping).
- *Optimization:*
  - Uses *RMSProp* optimizer and *ReduceLROnPlateau* for learning rate adjustment.

---

## ⚙ Model Fitting
- The model is trained using Kaggle's *GPU acceleration*.
- *Training and validation losses* are monitored to prevent overfitting.
- Predictions are saved to a CSV file for competition submission.

---

## 📈 Results and Observations
### ✅ Data Preparation
- *Balanced class distribution* confirmed using Countplot.
- No missing values detected.
- Pixel values successfully normalized.

### ✅ Model Performance
- *Learning curve* shows consistent decrease in training and validation losses.
- *Confusion matrix* highlights misclassifications, identifying areas for improvement.

### ✅ Prediction Output
- Model predictions saved to CSV for easy submission.
- Consistent validation accuracy demonstrating model reliability.

---

## 📊 Managerial Insights
### 🔥 Automation Potential
- The CNN model's high accuracy enables *automated data entry* for sectors like banking, postal services, and form digitization.
- Reduces manual errors and accelerates processing times.

### 💡 Cost-Effectiveness
- Deep learning-based recognition cuts down labor costs for digit transcription.
- Data augmentation improves model performance without requiring massive datasets.

### 🔥 Scalability & Adaptability
- The model can be fine-tuned to recognize *diverse handwriting styles* and languages.
- Ideal for *finance, healthcare, and government document processing*.

### 💡 Infrastructure Optimization
- CNNs require significant *computational power* (GPUs).
- Cloud-based AI services can provide a *cost-effective balance* between performance and infrastructure.

### 🔥 Continuous Improvement
- *Active learning* and regular model retraining with new data can enhance accuracy over time.
- Identified misclassifications highlight areas for fine-tuning.

---

## ⚙ Requirements
To run the project, install the following dependencies:

numpy
pandas
matplotlib
tensorflow
keras
scikit-learn


---

## 📊 Usage
1. Clone the repository:

git clone <repository_url>

2. Install the required dependencies:

pip install -r requirements.txt

3. Open the Jupyter Notebook:

jupyter notebook DLM_CNN_055007_Final_.ipynb

4. Run the cells sequentially.
5. Evaluate accuracy, loss, and predictions.

---

## 📚 References
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---
Authors:
Sejal Raj (055041)
Vidushi Rawat (055056)
