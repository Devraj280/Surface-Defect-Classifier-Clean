import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
from glob import glob
from utils import load_model, preprocess_image, class_map, get_prediction, image_map

# ---------------------- #
# 🎯 Page Configuration
# ---------------------- #
st.set_page_config(page_title="Steel Fault Classifier", layout="wide")

# ---------------------- #
# 📥 Load Configuration
# ---------------------- #
with open("config.json") as f:
    config = json.load(f)

# ---------------------- #
# 📌 Sidebar: Navigation
# ---------------------- #
st.sidebar.title("🔗 Navigation")
st.sidebar.markdown(f"[GitHub]({config['github']})")
st.sidebar.markdown(f"[Open in Colab]({config['colab']})")
st.sidebar.markdown(f"[Test Dataset]({config['dataset']})")

# ---------------------- #
# 🧠 Sidebar: Model Selection
# ---------------------- #
st.sidebar.title("🧠 Select Model")
model_option = st.sidebar.radio("Choose a Model", ("MobileNetV2", "ResNet50", "Custom CNN"))

model_path_map = {
    "MobileNetV2": "models/model_m.h5",
    "ResNet50": "models/model_r.h5",
    "Custom CNN": "models/model_c.h5"
}

# ---------------------- #
# 📦 Load Selected Model
# ---------------------- #
model = load_model(model_path_map[model_option])

# ---------------------- #
# 📊 Sidebar: Confusion Matrix
# ---------------------- #
st.sidebar.title("📊 Model Visuals")
with st.sidebar.expander("📌 Confusion Matrix"):
    try:
        image_path = f"graphs/{image_map[model_option]}"
        st.image(image_path, use_container_width=True, caption="Confusion Matrix")
    except KeyError:
        st.warning(f"No confusion matrix available for: {model_option}")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# ---------------------- #
# 🖼️ Main Area
# ---------------------- #
st.title("🛠️ Steel Surface Fault Classifier")
st.write("Upload a steel image (`.bmp`, `.jpg`, `.png`) to detect fault class using your selected model.")

uploaded_file = st.file_uploader("Upload an image", type=["bmp", "jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = preprocess_image(test_image, model_option)

    pred_index, pred_confidence, raw_pred = get_prediction(model, img_tensor)

    st.subheader("🔎 Prediction Result")
    st.success(f"**Class:** {class_map[pred_index]} | **Confidence:** {pred_confidence:.2f}%")
    st.bar_chart(raw_pred)

    # Optional: Download as JSON
    if st.button("📥 Download Prediction"):
        st.download_button(
            label="Download JSON",
            data=json.dumps({
                "class": class_map[pred_index],
                "confidence": f"{pred_confidence:.2f}%",
                "raw_output": raw_pred.tolist()
            }),
            file_name="prediction_result.json",
            mime="application/json"
        )

# ---------------------- #
# 🧪 Sample Test Images
# ---------------------- #
st.markdown("---")
st.subheader("🧪 Or Try Sample Test Images")

test_images = glob("test_images/*")
cols = st.columns(4)

for i, img_path in enumerate(test_images):
    with cols[i % 4]:
        if st.button(f"Use {os.path.basename(img_path)}", key=img_path):
            test_image = Image.open(img_path).convert("RGB")
            st.image(test_image, caption="Test Image", use_container_width=True)
            img_tensor = preprocess_image(test_image)
            pred_index, pred_confidence, raw_pred = get_prediction(model, img_tensor)
            st.success(f"**Class:** {class_map[pred_index]} | **Confidence:** {pred_confidence:.2f}%")
            st.bar_chart(raw_pred)
