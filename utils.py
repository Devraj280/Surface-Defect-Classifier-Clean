import tensorflow as tf
import numpy as np
from PIL import Image

class_map = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted",
    4: "rolled",
    5: "scratches"
}

image_map = {
    "MobileNetV2": "confusion_m_model_m.png",
    "ResNet50": "confusion_m_model_r.png",
    "Custom CNN": "confusion_m_model_c.png"
}
def preprocess_image(image, model_type="MobileNetV2"):
    if model_type == "Custom CNN":
        target_size = (200, 200)
    else:
        target_size = (224, 224)

    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)
    return image



def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor

def get_prediction(model, img_tensor):
    pred = model.predict(img_tensor)[0]
    index = np.argmax(pred)
    confidence = np.max(pred) * 100
    return index, confidence, pred
