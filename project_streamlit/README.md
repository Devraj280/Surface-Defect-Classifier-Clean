# 🛠️ Steel Fault Classifier Streamlit App

A local and web-ready Streamlit app to classify steel surface faults using pre-trained CNN, ResNet50, and MobileNetV2 models.

## 🔧 Features

- Upload images (`.bmp`, `.jpg`, `.png`)
- Choose between 3 deep learning models
- View prediction, confidence, and probability chart
- Preview test images
- Sidebar with model metrics: confusion matrix, accuracy vs. loss, precision vs. recall
- Links to GitHub, Colab, and Dataset

## 🧠 Model Info

- `model_m.h5`: MobileNetV2
- `model_r.h5`: ResNet50
- `model_c.h5`: Custom CNN

## 🚀 Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
