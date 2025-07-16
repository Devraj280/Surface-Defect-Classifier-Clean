import tensorflow as tf
import numpy as np

# Steel fault class map
class_map = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted",
    4: "rolled",
    5: "scratches"
}

# Confusion matrix image paths
image_map = {
    "MobileNetV2": "confusion_m_model_m.png",
    "ResNet50": "confusion_m_model_r.png",
    "Custom CNN": "confusion_m_model_c.png"
}

def load_model(model_path):
    """Load a TensorFlow .h5 or TFLite model."""
    if model_path.endswith(".tflite"):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        return tf.keras.models.load_model(model_path)

def preprocess_image(image, model_type="MobileNetV2"):
    """Resize and normalize image for model input."""
    target_size = (224, 224)
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def get_prediction(model, img_tensor):
    """Run inference and return predicted index, confidence, and raw scores."""
    if isinstance(model, tf.lite.Interpreter):
        input_index = model.get_input_details()[0]['index']
        output_index = model.get_output_details()[0]['index']
        model.set_tensor(input_index, img_tensor)
        model.invoke()
        raw_pred = model.get_tensor(output_index)[0]
    else:
        raw_pred = model.predict(img_tensor)[0]

    pred_index = int(np.argmax(raw_pred))
    pred_confidence = float(np.max(raw_pred)) * 100
    return pred_index, pred_confidence, raw_pred
