import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model(r'models\best_model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations for smaller model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

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
with open('models/sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model converted successfully!")
print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
print(f"Saved to: models/sign_language_model.tflite")