import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/model.h5')

# Export model to different formats
# Save as TensorFlow Lite
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tflite_converter.convert()
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save as ONNX
import tf2onnx
onnx_model = tf2onnx.convert.from_keras(model)
onnx.save_model(onnx_model, 'models/model.onnx')

print("Model exported to TFLite and ONNX formats.")
