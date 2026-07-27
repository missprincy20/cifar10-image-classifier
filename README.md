# 🖼️ CIFAR-10 Image Classification using CNN

A simple image classification web application built with **TensorFlow**, **Keras**, and **Streamlit**. The model is trained on the CIFAR-10 dataset to classify images into one of ten categories.

Users can upload an image to receive the predicted class, confidence score, and top three predictions. The project is designed for learning, academic assignments, and understanding how a trained deep learning model can be deployed as an interactive web application.

---

# Dataset

This project uses the **CIFAR-10** dataset provided by TensorFlow.

### Dataset Information

* Total Images: **60,000**
* Training Images: **50,000**
* Testing Images: **10,000**
* Classes: **10**
* Image Size: **32 × 32**
* Image Type: **RGB**

### Supported Classes

* ✈️ Airplane
* 🚗 Automobile
* 🐦 Bird
* 🐱 Cat
* 🦌 Deer
* 🐶 Dog
* 🐸 Frog
* 🐴 Horse
* 🚢 Ship
* 🚚 Truck

---

# How It Works

The application follows these steps:

1. The user uploads an image.
2. The image is converted to RGB.
3. The image is resized to **32 × 32** pixels.
4. Pixel values are normalized.
5. The trained CNN model loads automatically.
6. The model predicts probabilities for all 10 classes.
7. The application displays:

   * Predicted class
   * Confidence score
   * Top 3 predictions
   * Probability chart

---

# Demo Link
---
comming soon
---

# Technologies Used

### Programming Language

* Python

### Libraries

* TensorFlow
* Keras
* NumPy
* Pillow (PIL)
* Matplotlib
* Scikit-learn
* JSON

### Framework

* Streamlit

### Development Tools

* Google Colab
* Visual Studio Code
* GitHub

---

# Project Structure

```
CIFAR10-IMAGE-CLASSIFIER/
│
├── app.py
├── cifar10_cnn_model.keras
├── class_names.json
├── requirements.txt
├── README.md
├── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/missprincy20/cifar10-image-classifier.git
```

Move to the project folder

```bash
cd cifar10-image-classifier
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Input Requirements

For better predictions:

* Upload a **clear image**.
* The image should contain **one main object**.
* The object should belong to one of the **10 CIFAR-10 classes**.
* Avoid blurry or heavily edited images.

---

# Importance of the Project

This project helps understand the complete deep learning workflow:

* Image preprocessing
* CNN model building
* Model training
* Model evaluation
* Saving trained models
* Deploying a deep learning model using Streamlit
* Building an interactive image classification application

It is a good beginner project for learning computer vision and deployment.

---

# Limitations

This model is trained only on the **CIFAR-10** dataset.

Because of this:

* It supports only **10 predefined classes**.
* Images very different from the CIFAR-10 dataset may not produce accurate predictions.
* Large, complex, or high-resolution real-world images may reduce prediction accuracy.
* The application cannot recognize objects outside the supported categories.

These limitations are expected for a CNN trained specifically on the CIFAR-10 dataset.
---

# Future Improvements

Some possible improvements include:

* Improve model accuracy using data augmentation.
* Use transfer learning models such as MobileNetV2 or ResNet.
* Add support for more image categories.
* Enable camera-based image prediction.
* Deploy on cloud platforms with automatic updates.
* Improve the user interface with additional visualizations.

---

# Application Preview

* Home Page
* Image Upload
* Prediction Result
* Probability Chart
* Download Result

---

#  Author

Princy Patle
