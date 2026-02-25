import tensorflow as tf
from pathlib import Path
import os

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
# If the script is in test/models/, then keys are in test/models/
MODEL_PATH = SCRIPT_DIR / "best_model.keras"
OUTPUT_PATH = SCRIPT_DIR / "sign_language_model.tflite"

print(f"Loading model from: {MODEL_PATH}")

# Load the trained Keras model
model = tf.keras.models.load_model(str(MODEL_PATH))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# -------------------------------------------------------------------------
# CRITICAL FOR ACCURACY:
# We REMOVED predictions optimizations (quantization) to keep the model 
# in float32 (full precision). This makes the model larger but preserves 
# the exact accuracy of the Keras model.
# -------------------------------------------------------------------------
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  <-- REMOVED

# Required for Bidirectional LSTM - use TF Select ops for unsupported operations
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Native TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS      # TensorFlow ops fallback
]

# Disable tensor list lowering (required for LSTM with dynamic shapes)
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model converted successfully!")
print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
print(f"Saved to: {OUTPUT_PATH}")