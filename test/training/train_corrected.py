"""
Main training script for sign language recognition
Works with preprocessed .npy data from data_preprocessing.py
"""
import tensorflow as tf
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🎮 Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"   - {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"   Warning: {e}")
else:
    print("⚠️  No GPU found. Training will use CPU (this will be slower).")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from training.lstm_model import create_lstm_model, train_model, evaluate_model

def load_preprocessed_data(data_dir="preprocessed"):
    """
    Load preprocessed data from disk
    
    Args:
        data_dir: Directory containing preprocessed .npy files
    
    Returns:
        X_train, X_val, y_train, y_val, metadata
    """
    print(f"\n📂 Loading preprocessed data from: {data_dir}/")
    
    # Check if files exist
    required_files = ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy", "label_mappings.json"]
    for file in required_files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}\n"
                                   f"Please run: python training/data_preprocessing.py first")
    
    # Load data
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    
    # Load metadata
    with open(os.path.join(data_dir, "label_mappings.json"), "r") as f:
        metadata = json.load(f)
    
    print(f"   ✅ X_train: {X_train.shape}")
    print(f"   ✅ X_val: {X_val.shape}")
    print(f"   ✅ y_train: {y_train.shape}")
    print(f"   ✅ y_val: {y_val.shape}")
    print(f"   ✅ Number of classes: {metadata['num_classes']}")
    
    # Convert idx_to_label keys from string to int
    metadata['idx_to_label'] = {int(k): v for k, v in metadata['idx_to_label'].items()}
    
    return X_train, X_val, y_train, y_val, metadata

def main(data_dir="preprocessed", model_dir="models", epochs=20, batch_size=16):
    """
    Main training function
    
    Args:
        data_dir: Directory with preprocessed data
        model_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    print("="*60)
    print("🎯 SIGN LANGUAGE RECOGNITION - TRAINING")
    print("="*60)
    
    # Load data
    X_train, X_val, y_train, y_val, metadata = load_preprocessed_data(data_dir)
    
    num_classes = metadata['num_classes']
    idx_to_label = metadata['idx_to_label']
    
    print(f"\n🏷️  Classes:")
    for idx, label in idx_to_label.items():
        print(f"   {idx}: {label}")
    
    # Train model
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=model_dir
    )
    
    # Evaluate model
    results = evaluate_model(model, X_val, y_val, idx_to_label)
    
    # Save training history
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        history_dict = {
            key: [float(val) for val in values] 
            for key, values in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"\n📊 Training history saved to: {history_path}")
    
    # Save label mapping for inference
    label_map_path = os.path.join(model_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({
            "label_to_idx": metadata['label_to_idx'],
            "idx_to_label": idx_to_label,
            "num_classes": num_classes
        }, f, indent=2)
    print(f"📝 Label map saved to: {label_map_path}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✅ Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"✅ Final validation accuracy: {results['accuracy']*100:.2f}%")
    print(f"\n📁 Models saved in: {model_dir}/")
    print("   - best_model.keras (use this for inference)")
    print("   - final_model.keras")
    print("   - label_map.json")
    print("   - training_history.json")
    
    return model, history, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train sign language recognition model")
    parser.add_argument("--data-dir", type=str, default="preprocessed",
                       help="Directory with preprocessed data")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    
    args = parser.parse_args()
    
    # Check if preprocessed data exists
    if not os.path.exists(args.data_dir):
        print(f"\n❌ Error: Preprocessed data directory '{args.data_dir}' not found!")
        print("\n📋 To preprocess your data, run:")
        print("   python training/data_preprocessing.py")
        print("\nThis will:")
        print("   1. Load all .npy files from sign_language_dataset/")
        print("   2. Split into training and validation sets")
        print("   3. Apply data augmentation")
        print("   4. Save preprocessed data to preprocessed/")
        sys.exit(1)
    
    # Run training
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
