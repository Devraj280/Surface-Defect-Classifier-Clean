import tensorflow as tf

def convert_to_tflite(h5_path, tflite_path):
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Converted: {h5_path} -> {tflite_path}")

# Example usage
convert_to_tflite("models/model_m.h5", "models/MobileNetV2_tflite.tflite")
convert_to_tflite("models/model_r.h5", "models/ResNet50_tflite.tflite")
convert_to_tflite("models/model_c.h5", "models/Custom CNN_tflite.tflite")
