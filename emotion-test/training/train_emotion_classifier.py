"""
Emotion Classifier Training (Phase 2 — SVM + optional MLP)

Loads .npy blendshape data collected by collect_emotion_data.py, preprocesses it,
and trains an SVM classifier on the 52 blendshape features.

Pipeline mirrors sign-test/training/ but uses frame-level classification
instead of sequence-level LSTM since emotions are expressed continuously.

Usage:
    python training/train_emotion_classifier.py
    python training/train_emotion_classifier.py --dataset-dir emotion_dataset --model-type svm
    python training/train_emotion_classifier.py --model-type mlp
"""
import numpy as np
import os
import json
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────────────
# 1. DATA LOADING (mirrors sign-test/training/data_preprocessing.py)
# ────────────────────────────────────────────────────────────────────

def load_dataset_from_npy(dataset_dir="emotion_dataset", expected_frames=60, num_features=52):
    """
    Load all .npy blendshape files from the dataset directory.
    
    Args:
        dataset_dir: Root directory containing emotion class folders
        expected_frames: Expected frames per sample (60 for 2s at 30FPS)
        num_features: Number of blendshape features per frame (52)
    
    Returns:
        X_sequences: array of shape (n_samples, expected_frames, num_features) — full sequences
        X_frames: array of shape (n_samples * expected_frames, num_features) — individual frames
        y_sequences: labels for sequences (n_samples,)
        y_frames: labels for individual frames (n_samples * expected_frames,)
        label_to_idx, idx_to_label: label mappings
    """
    print(f"📂 Loading emotion dataset from: {dataset_dir}")
    
    class_dirs = sorted([d for d in Path(dataset_dir).iterdir() 
                         if d.is_dir() and not d.name.startswith('.')])
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {dataset_dir}")
    
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    label_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}
    idx_to_label = {idx: d.name for idx, d in enumerate(class_dirs)}
    
    X_seq_list = []
    y_seq_list = []
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = label_to_idx[class_name]
        
        npy_files = sorted(list(class_dir.glob("*.npy")))
        print(f"  {class_name}: {len(npy_files)} samples")
        
        for npy_file in npy_files:
            try:
                data = np.load(npy_file)
                
                # Validate / pad / truncate to expected shape
                if data.shape[0] != expected_frames:
                    if data.shape[0] < expected_frames:
                        padding = np.zeros((expected_frames - data.shape[0], data.shape[1]))
                        data = np.vstack([data, padding])
                    else:
                        data = data[:expected_frames]
                
                if data.shape[1] != num_features:
                    print(f"    ⚠️ {npy_file.name}: {data.shape[1]} features, expected {num_features}. Skipping.")
                    continue
                
                X_seq_list.append(data)
                y_seq_list.append(class_idx)
                
            except Exception as e:
                print(f"    ❌ Error loading {npy_file.name}: {e}")
                continue
    
    if len(X_seq_list) == 0:
        raise ValueError("No valid samples loaded! Run collect_emotion_data.py first.")
    
    X_sequences = np.array(X_seq_list, dtype=np.float32)
    y_sequences = np.array(y_seq_list, dtype=np.int32)
    
    # Also create frame-level dataset (flatten sequences)
    # This is key: SVM works on individual frames, not sequences
    X_frames = X_sequences.reshape(-1, num_features)  # (n_samples*60, 52)
    y_frames = np.repeat(y_sequences, expected_frames)  # repeat each label 60 times
    
    # Remove zero-only frames (no face detected)
    nonzero_mask = np.any(X_frames != 0, axis=1)
    X_frames_filtered = X_frames[nonzero_mask]
    y_frames_filtered = y_frames[nonzero_mask]
    
    print(f"\n✅ Loaded {len(X_sequences)} samples ({len(X_frames_filtered)} valid frames)")
    print(f"   Sequence shape: {X_sequences.shape}")
    print(f"   Frame shape: {X_frames_filtered.shape}")
    
    return X_sequences, X_frames_filtered, y_sequences, y_frames_filtered, label_to_idx, idx_to_label


# ────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────

def compute_statistical_features(X_sequences):
    """
    Compute statistical features from blendshape sequences.
    For each sequence of (60, 52), compute per-blendshape: mean, std, max, min, range
    Resulting feature vector: 52 * 5 = 260 features per sample.
    
    This is used for sequence-level classification (one label per 2-second clip).
    """
    features = []
    for seq in X_sequences:
        # Filter out zero frames
        nonzero = seq[np.any(seq != 0, axis=1)]
        if len(nonzero) == 0:
            nonzero = seq  # fallback
        
        mean_feat = np.mean(nonzero, axis=0)
        std_feat = np.std(nonzero, axis=0)
        max_feat = np.max(nonzero, axis=0)
        min_feat = np.min(nonzero, axis=0)
        range_feat = max_feat - min_feat
        
        combined = np.concatenate([mean_feat, std_feat, max_feat, min_feat, range_feat])
        features.append(combined)
    
    return np.array(features, dtype=np.float32)


# ────────────────────────────────────────────────────────────────────
# 3. DATA AUGMENTATION
# ────────────────────────────────────────────────────────────────────

def augment_frame_data(X_train, y_train, augmentation_factor=3):
    """
    Augment training data with noise and scaling.
    
    Args:
        X_train: Training frames (n_frames, 52)
        y_train: Labels
        augmentation_factor: Multiplier (3 = triple the dataset)
    
    Returns:
        X_augmented, y_augmented
    """
    print(f"\n🪄 Augmenting frame data (factor: {augmentation_factor})...")
    
    X_aug = [X_train]
    y_aug = [y_train]
    
    for _ in range(augmentation_factor - 1):
        # Random scaling (0.9 to 1.1)
        scale = np.random.uniform(0.9, 1.1, X_train.shape)
        noisy = X_train * scale
        
        # Small Gaussian noise
        noise = np.random.normal(0, 0.01, X_train.shape)
        noisy = noisy + noise
        
        # Clip to valid range [0, 1]
        noisy = np.clip(noisy, 0.0, 1.0)
        
        X_aug.append(noisy.astype(np.float32))
        y_aug.append(y_train)
    
    X_augmented = np.concatenate(X_aug, axis=0)
    y_augmented = np.concatenate(y_aug, axis=0)
    
    # Shuffle
    indices = np.arange(len(X_augmented))
    np.random.shuffle(indices)
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]
    
    print(f"   Original: {len(X_train)} frames → Augmented: {len(X_augmented)} frames")
    return X_augmented, y_augmented


# ────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ────────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train, X_val, y_val, idx_to_label):
    """Train an SVM classifier on frame-level blendshape data"""
    print("\n" + "=" * 60)
    print("🚀 TRAINING SVM CLASSIFIER")
    print("=" * 60)
    
    # Build pipeline with scaling + SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,  # Enable probability estimates for confidence
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42
        ))
    ])
    
    print(f"\n📊 Training data: {X_train.shape}")
    print(f"📊 Validation data: {X_val.shape}")
    
    # Cross-validation on training data
    print("\n🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    # Train on full training set
    print("\n🏋️ Training final model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n✅ Validation Accuracy: {accuracy*100:.2f}%")
    print(f"\n📈 Classification Report:")
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    print("📊 Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(f"{'':>12}", end="")
    for name in target_names:
        print(f"{name:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>12}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()
    
    return pipeline, accuracy


def train_mlp(X_train, y_train, X_val, y_val, idx_to_label):
    """Train an MLP classifier as alternative to SVM"""
    print("\n" + "=" * 60)
    print("🚀 TRAINING MLP CLASSIFIER")
    print("=" * 60)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=False
        ))
    ])
    
    print(f"\n📊 Training data: {X_train.shape}")
    print(f"📊 Validation data: {X_val.shape}")
    
    # Cross-validation
    print("\n🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    # Train
    print("\n🏋️ Training final model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n✅ Validation Accuracy: {accuracy*100:.2f}%")
    print(f"\n📈 Classification Report:")
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    return pipeline, accuracy


# ────────────────────────────────────────────────────────────────────
# 5. SEQUENCE-LEVEL TRAINING (SVM on statistical features)
# ────────────────────────────────────────────────────────────────────

def train_sequence_svm(X_sequences, y_sequences, idx_to_label):
    """
    Train an SVM on sequence-level statistical features.
    This gives one prediction per 2-second recording.
    """
    print("\n" + "=" * 60)
    print("🚀 TRAINING SEQUENCE-LEVEL SVM")
    print("=" * 60)
    
    # Compute statistical features
    print("\n📐 Computing statistical features from sequences...")
    X_features = compute_statistical_features(X_sequences)
    print(f"   Feature shape: {X_features.shape} (mean + std + max + min + range of 52 blendshapes)")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_features, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
    )
    
    # Train
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                     class_weight='balanced', random_state=42))
    ])
    
    print(f"\n📊 Training: {X_train.shape}, Validation: {X_val.shape}")
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=min(5, len(X_train)), scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n✅ Sequence-level Accuracy: {accuracy*100:.2f}%")
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    return pipeline, accuracy


# ────────────────────────────────────────────────────────────────────
# 6. MODEL SAVING
# ────────────────────────────────────────────────────────────────────

def save_model(model, label_to_idx, idx_to_label, output_dir="models", model_name="emotion_svm"):
    """Save trained model and label mappings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Save label map
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
            "num_classes": len(label_to_idx),
            "num_features": 52,
            "model_type": model_name,
        }, f, indent=2)
    print(f"💾 Label map saved: {label_map_path}")
    
    return model_path, label_map_path


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier on blendshape data")
    parser.add_argument("--dataset-dir", type=str, default="emotion_dataset",
                        help="Directory containing emotion class folders with .npy files")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for trained models")
    parser.add_argument("--model-type", type=str, default="svm", choices=["svm", "mlp", "both"],
                        help="Model type to train (default: svm)")
    parser.add_argument("--augment", type=int, default=3,
                        help="Data augmentation factor (1=none, 3=triple)")
    parser.add_argument("--mode", type=str, default="both", choices=["frame", "sequence", "both"],
                        help="Training mode: frame-level, sequence-level, or both")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎭 EMOTION CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Load data
    X_seq, X_frames, y_seq, y_frames, label_to_idx, idx_to_label = load_dataset_from_npy(
        dataset_dir=args.dataset_dir
    )
    
    # Print dataset stats
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {len(X_seq)}")
    print(f"   Total frames: {len(X_frames)}")
    print(f"   Classes: {label_to_idx}")
    for idx, name in idx_to_label.items():
        count = np.sum(y_seq == idx)
        print(f"   {name}: {count} sequences ({count * 60} frames)")
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    # ── Frame-level training ──
    if args.mode in ["frame", "both"]:
        # Split frame-level data
        X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
            X_frames, y_frames, test_size=0.2, random_state=42, stratify=y_frames
        )
        
        # Augment
        if args.augment > 1:
            X_train_f, y_train_f = augment_frame_data(X_train_f, y_train_f, args.augment)
        
        if args.model_type in ["svm", "both"]:
            svm_model, svm_acc = train_svm(X_train_f, y_train_f, X_val_f, y_val_f, idx_to_label)
            if svm_acc > best_accuracy:
                best_model, best_accuracy, best_name = svm_model, svm_acc, "emotion_svm_frame"
        
        if args.model_type in ["mlp", "both"]:
            mlp_model, mlp_acc = train_mlp(X_train_f, y_train_f, X_val_f, y_val_f, idx_to_label)
            if mlp_acc > best_accuracy:
                best_model, best_accuracy, best_name = mlp_model, mlp_acc, "emotion_mlp_frame"
    
    # ── Sequence-level training ──
    if args.mode in ["sequence", "both"] and len(X_seq) >= 10:
        seq_model, seq_acc = train_sequence_svm(X_seq, y_seq, idx_to_label)
        save_model(seq_model, label_to_idx, idx_to_label, args.output_dir, "emotion_svm_sequence")
    elif args.mode in ["sequence", "both"]:
        print("\n⚠️ Not enough sequences for sequence-level training (need ≥10)")
    
    # Save best frame-level model
    if best_model is not None:
        save_model(best_model, label_to_idx, idx_to_label, args.output_dir, best_name)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    if best_model:
        print(f"   Best model: {best_name} ({best_accuracy*100:.2f}%)")
    print(f"   Models saved in: {args.output_dir}/")
    print(f"\n   Next: python validation/realtime_emotion.py")


if __name__ == "__main__":
    main()
