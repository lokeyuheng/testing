"""
Preprocessing script for sign language .npy files
Loads normalized hand landmark data from collect_sign_data.py output
"""
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_dataset_from_npy(dataset_dir="sign_language_dataset", expected_frames=60):
    """
    Load all .npy files from the dataset directory
    
    Args:
        dataset_dir: Root directory containing class folders
        expected_frames: Expected number of frames per sample (60 for 2-second recordings at 30 FPS)
    
    Returns:
        X: numpy array of shape (n_samples, expected_frames, 63)
        y: numpy array of labels (n_samples,)
        label_to_idx: dictionary mapping class names to indices
        idx_to_label: dictionary mapping indices to class names
    """
    print(f"📂 Loading dataset from: {dataset_dir}")
    
    # Get all class directories
    class_dirs = sorted([d for d in Path(dataset_dir).iterdir() if d.is_dir()])
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {dataset_dir}")
    
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    # Create label mappings
    label_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}
    idx_to_label = {idx: d.name for idx, d in enumerate(class_dirs)}
    
    X_list = []
    y_list = []
    
    # Load all .npy files
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = label_to_idx[class_name]
        
        npy_files = sorted(list(class_dir.glob("*.npy")))
        print(f"  {class_name}: {len(npy_files)} samples")
        
        for npy_file in npy_files:
            try:
                # Load landmarks
                landmarks = np.load(npy_file)
                
                # Validate shape
                if landmarks.shape[0] != expected_frames:
                    print(f"    ⚠️ Warning: {npy_file.name} has {landmarks.shape[0]} frames, expected {expected_frames}. Padding/truncating...")
                    
                    # Pad or truncate to expected_frames
                    if landmarks.shape[0] < expected_frames:
                        # Pad with zeros
                        padding = np.zeros((expected_frames - landmarks.shape[0], landmarks.shape[1]))
                        landmarks = np.vstack([landmarks, padding])
                    else:
                        # Truncate
                        landmarks = landmarks[:expected_frames]
                
                # Validate feature dimension (should be 63 for normalized hand landmarks)
                if landmarks.shape[1] != 63:
                    print(f"    ❌ Error: {npy_file.name} has {landmarks.shape[1]} features, expected 63. Skipping...")
                    continue
                
                X_list.append(landmarks)
                y_list.append(class_idx)
                
            except Exception as e:
                print(f"    ❌ Error loading {npy_file.name}: {e}")
                continue
    
    if len(X_list) == 0:
        raise ValueError("No valid samples loaded!")
    
    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"\n✅ Loaded {len(X)} samples successfully")
    print(f"   Shape: X={X.shape}, y={y.shape}")
    print(f"   Classes: {label_to_idx}")
    
    return X, y, label_to_idx, idx_to_label

def prepare_train_val_split(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and validation sets
    
    Args:
        X: Features (n_samples, n_frames, n_features)
        y: Labels (n_samples,)
        test_size: Proportion of validation data
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    print(f"\n📊 Splitting dataset: {int((1-test_size)*100)}% train, {int(test_size*100)}% validation")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensure balanced split across classes
    )
    
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n   Training class distribution:")
    for label_idx, count in zip(unique, counts):
        print(f"     Class {label_idx}: {count} samples")
    
    return X_train, X_val, y_train, y_val

def augment_data(X_train, y_train, augmentation_factor=2):
    """
    Augment training data with simple transformations
    
    Args:
        X_train: Training features
        y_train: Training labels
        augmentation_factor: How many times to augment (2 = double the dataset)
    
    Returns:
        X_augmented, y_augmented (original + augmented data)
    """
    print(f"\n🪄 Augmenting training data (factor: {augmentation_factor})...")
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    
    for _ in range(augmentation_factor - 1):
        X_aug = []
        for sample in X_train:
            # Random scaling (0.95 to 1.05)
            scale = np.random.uniform(0.95, 1.05)
            augmented = sample * scale
            
            # Small Gaussian noise
            noise = np.random.normal(0, 0.002, sample.shape)
            mask = (sample != 0)  # Only add noise to non-zero values
            augmented = augmented + (noise * mask)
            
            # Temporal shift (small)
            if np.random.random() > 0.5:
                shift = np.random.randint(-2, 3)
                if shift > 0:
                    augmented = np.pad(augmented, ((shift, 0), (0, 0)), mode='edge')[:-shift]
                elif shift < 0:
                    augmented = np.pad(augmented, ((0, -shift), (0, 0)), mode='edge')[-shift:]
            
            X_aug.append(augmented.astype(np.float32))
        
        X_aug_list.append(np.array(X_aug))
        y_aug_list.append(y_train)
    
    X_augmented = np.concatenate(X_aug_list, axis=0)
    y_augmented = np.concatenate(y_aug_list, axis=0)
    
    # Shuffle
    indices = np.arange(len(X_augmented))
    np.random.shuffle(indices)
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]
    
    print(f"   Original: {len(X_train)} samples")
    print(f"   Augmented: {len(X_augmented)} samples")
    
    return X_augmented, y_augmented

def save_preprocessed_data(X_train, X_val, y_train, y_val, label_to_idx, idx_to_label, output_dir="preprocessed"):
    """Save preprocessed data for training"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving preprocessed data to: {output_dir}/")
    
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    
    # Save label mappings as JSON
    import json
    with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
        json.dump({
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "num_classes": len(label_to_idx),
            "num_frames": int(X_train.shape[1]),
            "num_features": int(X_train.shape[2])
        }, f, indent=2)
    
    print(f"   ✅ X_train.npy: {X_train.shape}")
    print(f"   ✅ X_val.npy: {X_val.shape}")
    print(f"   ✅ y_train.npy: {y_train.shape}")
    print(f"   ✅ y_val.npy: {y_val.shape}")
    print(f"   ✅ label_mappings.json")

def visualize_dataset_stats(X, y, idx_to_label):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"  - Frames per sample: {X.shape[1]}")
    print(f"  - Features per frame: {X.shape[2]}")
    
    print(f"\nClasses: {len(idx_to_label)}")
    for idx, label in idx_to_label.items():
        count = np.sum(y == idx)
        print(f"  {idx}: '{label}' - {count} samples")
    
    print(f"\nData statistics:")
    print(f"  Mean: {X.mean():.4f}")
    print(f"  Std: {X.std():.4f}")
    print(f"  Min: {X.min():.4f}")
    print(f"  Max: {X.max():.4f}")
    
    # Check for zero frames (masked frames)
    zero_frames = np.sum(np.all(X == 0, axis=2))
    total_frames = X.shape[0] * X.shape[1]
    print(f"\nZero (masked) frames: {zero_frames}/{total_frames} ({100*zero_frames/total_frames:.1f}%)")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess sign language dataset")
    parser.add_argument("--dataset-dir", type=str, default="sign_language_dataset", 
                       help="Directory containing class folders with .npy files")
    parser.add_argument("--output-dir", type=str, default="preprocessed",
                       help="Output directory for preprocessed data")
    parser.add_argument("--validation-split", type=float, default=0.2,
                       help="Proportion of data for validation (0.0-1.0)")
    parser.add_argument("--augment", type=int, default=2,
                       help="Data augmentation factor (1=no augmentation, 2=double dataset)")
    parser.add_argument("--expected-frames", type=int, default=60,
                       help="Expected number of frames per sample")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save preprocessed data (just show statistics)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 SIGN LANGUAGE DATA PREPROCESSING")
    print("="*60)
    
    # Load dataset
    X, y, label_to_idx, idx_to_label = load_dataset_from_npy(
        dataset_dir=args.dataset_dir,
        expected_frames=args.expected_frames
    )
    
    # Show statistics
    visualize_dataset_stats(X, y, idx_to_label)
    
    # Split data
    X_train, X_val, y_train, y_val = prepare_train_val_split(
        X, y, 
        test_size=args.validation_split
    )
    
    # Augment training data
    if args.augment > 1:
        X_train, y_train = augment_data(X_train, y_train, augmentation_factor=args.augment)
    
    # Save preprocessed data
    if not args.no_save:
        save_preprocessed_data(X_train, X_val, y_train, y_val, 
                              label_to_idx, idx_to_label, 
                              output_dir=args.output_dir)
        
        print("\n✅ Preprocessing complete!")
        print(f"\nTo train the model, run:")
        print(f"  python training/train.py --data-dir {args.output_dir}")
    else:
        print("\n✅ Statistics computed (data not saved due to --no-save flag)")
