# File: training/convert_to_tflite.py

import tensorflow as tf
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("hand_detector_model.h5")

# Quantization with representative dataset
def representative_dataset():
    for _ in range(100):
        dummy = np.random.rand(1, 96, 96, 1).astype(np.float32)
        yield [dummy]

# Convert to TFLite (int8 quantized)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save TFLite model
with open("hand_detector_model_quant.tflite", "wb") as f:
    f.write(tflite_model)
print(" Saved quantized model: hand_detector_model_quant.tflite")

# Convert to C++ header for Arduino
os.system("xxd -i hand_detector_model_quant.tflite > hand_model_data.h")
print(" Saved header file: hand_model_data.h")
