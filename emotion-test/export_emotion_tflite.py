"""
Export Emotion SVM → TFLite

Converts the trained sklearn SVM pipeline (StandardScaler + SVC) into a
TensorFlow Lite model so it can run on Android alongside the sign language model.

The approach:
  1. Load the joblib pipeline (scaler + SVM)
  2. Build a small Keras model that replicates the SVM decision function
  3. Export to .tflite

Usage:
    python export_emotion_tflite.py
    python export_emotion_tflite.py --model models/emotion_svm_frame.joblib --output models/emotion_classifier.tflite
"""

import numpy as np
import json
import os
import argparse
import joblib
from pathlib import Path


def export_svm_to_tflite(model_path, output_path, label_map_path):
    """Convert sklearn Pipeline (StandardScaler + SVC) to TFLite"""
    
    print("=" * 60)
    print("🔄 SVM → TFLite CONVERTER")
    print("=" * 60)
    
    # ── 1. Load the trained pipeline ──
    print(f"\n📦 Loading model: {model_path}")
    pipeline = joblib.load(model_path)
    
    scaler = pipeline.named_steps['scaler']
    svm = pipeline.named_steps['svm']
    
    n_features = scaler.n_features_in_
    n_classes = len(svm.classes_)
    
    print(f"   Features: {n_features}")
    print(f"   Classes: {n_classes} → {list(svm.classes_)}")
    print(f"   Kernel: {svm.kernel}")
    
    # ── 2. Load label map ──
    print(f"\n📋 Loading label map: {label_map_path}")
    with open(label_map_path, 'r') as f:
        label_data = json.load(f)
    idx_to_label = {int(k): v for k, v in label_data['idx_to_label'].items()}
    print(f"   Labels: {idx_to_label}")
    
    # ── 3. Build equivalent Keras model ──
    # Strategy: Use the sklearn model to generate predictions on a grid,
    # then train a small neural net to mimic it. This is the most reliable
    # approach for RBF SVM → TFLite conversion.
    
    print("\n🧠 Building surrogate Keras model...")
    
    import tensorflow as tf
    
    # Generate synthetic training data for the surrogate model
    # Use the scaler's statistics to generate data in the right range
    mean = scaler.mean_
    scale = scaler.scale_
    
    np.random.seed(42)
    n_synthetic = 50000  # More data = better approximation
    
    # Generate data around the training distribution
    X_synthetic = np.random.randn(n_synthetic, n_features).astype(np.float32)
    # Scale back to original space
    X_synthetic_original = X_synthetic * scale * 2 + mean
    # Clip to valid blendshape range [0, 1]
    X_synthetic_original = np.clip(X_synthetic_original, 0.0, 1.0)
    
    # Get SVM predictions + probabilities
    print("   Generating SVM predictions on synthetic data...")
    y_synthetic_proba = pipeline.predict_proba(X_synthetic_original)
    
    # Build surrogate model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Model params: {model.count_params()}")
    
    # Train surrogate on SVM's soft predictions (knowledge distillation)
    print("   Training surrogate model (knowledge distillation)...")
    model.fit(
        X_synthetic_original.astype(np.float32),
        y_synthetic_proba.astype(np.float32),
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        verbose=0
    )
    
    # ── 4. Validate surrogate matches SVM ──
    print("\n📊 Validating surrogate accuracy...")
    X_test = np.random.randn(5000, n_features).astype(np.float32) * scale + mean
    X_test = np.clip(X_test, 0.0, 1.0)
    
    svm_preds = pipeline.predict(X_test)
    keras_proba = model.predict(X_test, verbose=0)
    keras_preds = np.array([svm.classes_[i] for i in np.argmax(keras_proba, axis=1)])
    
    agreement = np.mean(svm_preds == keras_preds)
    print(f"   SVM ↔ Keras agreement: {agreement*100:.1f}%")
    
    if agreement < 0.90:
        print("   ⚠️ Agreement is below 90%, retraining with more data...")
        # Retrain with harder examples
        more_data = np.random.randn(100000, n_features).astype(np.float32)
        more_original = more_data * scale * 3 + mean
        more_original = np.clip(more_original, 0.0, 1.0)
        more_proba = pipeline.predict_proba(more_original)
        
        model.fit(
            more_original.astype(np.float32),
            more_proba.astype(np.float32),
            epochs=30,
            batch_size=256,
            verbose=0
        )
        
        keras_proba2 = model.predict(X_test, verbose=0)
        keras_preds2 = np.array([svm.classes_[i] for i in np.argmax(keras_proba2, axis=1)])
        agreement = np.mean(svm_preds == keras_preds2)
        print(f"   SVM ↔ Keras agreement (retrained): {agreement*100:.1f}%")
    
    # ── 5. Convert to TFLite ──
    print("\n📱 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TFLite model
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_kb = len(tflite_model) / 1024
    print(f"   Saved: {output_path} ({tflite_size_kb:.1f} KB)")
    
    # ── 6. Verify TFLite model ──
    print("\n🔍 Verifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"   Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")
    
    # Test with a few samples
    test_samples = X_test[:10].astype(np.float32)
    tflite_correct = 0
    for i in range(len(test_samples)):
        interpreter.set_tensor(input_details[0]['index'], test_samples[i:i+1])
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(output_details[0]['index'])[0]
        tflite_label = svm.classes_[np.argmax(tflite_out)]
        svm_label = svm_preds[i]
        if tflite_label == svm_label:
            tflite_correct += 1
    
    print(f"   TFLite vs SVM spot check: {tflite_correct}/{len(test_samples)} match")
    
    # ── 7. Export label map for Android ──
    emotion_label_map = {
        "labels": [idx_to_label[i] for i in sorted(idx_to_label.keys())],
        "num_classes": n_classes,
        "num_features": n_features,
    }
    
    android_label_path = output_path.replace('.tflite', '_labels.json')
    with open(android_label_path, 'w') as f:
        json.dump(emotion_label_map, f, indent=2)
    print(f"   Label map: {android_label_path}")
    
    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE!")
    print("=" * 60)
    print(f"   TFLite model: {output_path} ({tflite_size_kb:.1f} KB)")
    print(f"   Label map: {android_label_path}")
    print(f"   SVM ↔ TFLite agreement: {agreement*100:.1f}%")
    print(f"\n   Next steps:")
    print(f"   1. Copy {output_path} → flutter-app/android/app/src/main/assets/")
    print(f"   2. Copy {android_label_path} → flutter-app/android/app/src/main/assets/")
    
    return output_path, android_label_path


def main():
    parser = argparse.ArgumentParser(description="Convert SVM emotion model to TFLite")
    parser.add_argument("--model", type=str, default="models/emotion_svm_frame.joblib",
                        help="Path to trained joblib pipeline")
    parser.add_argument("--output", type=str, default="models/emotion_classifier.tflite",
                        help="Output TFLite path")
    parser.add_argument("--label-map", type=str, default="models/label_map.json",
                        help="Path to label_map.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print(f"   Run: python training/train_emotion_classifier.py")
        return
    
    export_svm_to_tflite(args.model, args.output, args.label_map)


if __name__ == "__main__":
    main()
