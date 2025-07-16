import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
from glob import glob
from utils import load_model, preprocess_image, class_map, get_prediction, image_map

# Page Configuration
st.set_page_config(page_title="Steel Fault Classifier", layout="wide")

# Load Configuration
with open("config.json") as f:
    config = json.load(f)

# Sidebar: Navigation
st.sidebar.title("🔗 Navigation")
st.sidebar.markdown(f"[GitHub]({config['github']})")
st.sidebar.markdown(f"[Open in Colab]({config['colab']})")
st.sidebar.markdown(f"[Test Dataset]({config['dataset']})")

# Sidebar: Model Selection
st.sidebar.title("🧠 Select Model")
model_option = st.sidebar.radio("Choose a Model", ("MobileNetV2", "ResNet50", "Custom CNN"))

# Model Paths
model_path_map = {
    "MobileNetV2": "models/MobileNetV2_tflite.tflite" if os.path.exists("models/MobileNetV2_tflite.tflite") else "models/model_m.h5",
    "ResNet50": "models/ResNet50_tflite.tflite" if os.path.exists("models/ResNet50_tflite.tflite") else "models/model_r.h5",
    "Custom CNN": "models/Custom CNN_tflite.tflite" if os.path.exists("models/Custom CNN_tflite.tflite") else "models/model_c.h5"
}

@st.cache_resource(show_spinner="Loading model...")
def get_cached_model(path):
    return load_model(path)

model = get_cached_model(model_path_map[model_option])

# Sidebar: Confusion Matrix
st.sidebar.title("📊 Model Visuals")
with st.sidebar.expander("📌 Confusion Matrix"):
    try:
        image_path = f"graphs/{image_map[model_option]}"
        st.image(image_path, caption="Confusion Matrix")  # Removed use_container_width
    except KeyError:
        st.warning(f"No confusion matrix available for: {model_option}")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Main Area
st.title("🛠️ Steel Surface Fault Classifier")
st.write("Upload a steel image (`.bmp`, `.jpg`, `.png`) to detect fault class using your selected model.")

uploaded_file = st.file_uploader("Upload an image", type=["bmp", "jpg", "png"])

def run_prediction(image, label="Image"):
    st.image(image, caption=label)  # Removed use_container_width
    img_tensor = preprocess_image(image, model_option)

    try:
        pred_index, pred_confidence, raw_pred = get_prediction(model, img_tensor)
        st.subheader("🔎 Prediction Result")
        st.success(f"**Class:** {class_map[pred_index]} | **Confidence:** {pred_confidence:.2f}%")
        st.bar_chart(raw_pred)

        # Download Prediction
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# If user uploads an image
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        run_prediction(image, label="Uploaded Image")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")

# Sample Test Images
st.markdown("---")
st.subheader("🧪 Or Try Sample Test Images")

test_images = glob("test_images/*")
cols = st.columns(4)

for i, img_path in enumerate(test_images):
    with cols[i % 4]:
        if st.button(f"Use {os.path.basename(img_path)}", key=img_path):
            try:
                test_image = Image.open(img_path).convert("RGB")
                run_prediction(test_image, label=os.path.basename(img_path))
            except Exception as e:
                st.error(f"Error loading test image: {e}")
