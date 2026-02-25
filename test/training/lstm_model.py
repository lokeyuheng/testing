"""
LSTM Model for Sign Language Recognition
Simplified and corrected for .npy landmark data
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, 
    BatchNormalization, Masking
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import tensorflow as tf

def create_lstm_model(num_frames=60, num_features=63, num_classes=5):
    """
    Create Bidirectional LSTM model for sign language recognition
    
    Args:
        num_frames: Number of frames per sample (60 for 2-second videos at 30 FPS)
        num_features: Number of features per frame (63 for normalized hand landmarks: 21 landmarks × 3 coords)
        num_classes: Number of sign classes to predict
    
    Returns:
        Compiled Keras model
    """
    print(f"\n🏗️ Building LSTM model...")
    print(f"   Input: ({num_frames}, {num_features})")
    print(f"   Output: {num_classes} classes")
    
    reg = regularizers.l2(0.001)
    
    model = Sequential([
        # Input layer
        Input(shape=(num_frames, num_features), name="input"),
        
        # Masking layer (handles zero-padded frames)
        Masking(mask_value=0.0, name="masking"),
        
        # First Bidirectional LSTM layer
        Bidirectional(
            LSTM(128, return_sequences=True, kernel_regularizer=reg),
            name="bidirectional_lstm_1"
        ),
        BatchNormalization(name="batch_norm_1"),
        Dropout(0.3, name="dropout_1"),
        
        # Second Bidirectional LSTM layer
        Bidirectional(
            LSTM(64, return_sequences=False, kernel_regularizer=reg),
            name="bidirectional_lstm_2"
        ),
        BatchNormalization(name="batch_norm_2"),
        Dropout(0.3, name="dropout_2"),
        
        # Dense classification layers
        Dense(128, activation="relu", kernel_regularizer=reg, name="dense_1"),
        BatchNormalization(name="batch_norm_3"),
        Dropout(0.3, name="dropout_3"),
        
        Dense(64, activation="relu", kernel_regularizer=reg, name="dense_2"),
        Dropout(0.2, name="dropout_4"),
        
        # Output layer
        Dense(num_classes, activation="softmax", name="output")
    ], name="SignLanguage_LSTM")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    return model

def train_model(X_train, y_train, X_val, y_val, num_classes, 
                epochs=100, batch_size=16, output_dir="models"):
    """
    Train the LSTM model
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        num_classes: Number of classes
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save models
    
    Returns:
        model: Trained model
        history: Training history
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    
    # Create model
    num_frames = X_train.shape[1]
    num_features = X_train.shape[2]
    
    model = create_lstm_model(
        num_frames=num_frames,
        num_features=num_features,
        num_classes=num_classes
    )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🏋️ Training model...\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.keras")
    model.save(final_model_path)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\n📁 Models saved to: {output_dir}/")
    print(f"   - best_model.keras (best validation accuracy)")
    print(f"   - final_model.keras (final epoch)")
    
    return model, history

def evaluate_model(model, X_val, y_val, idx_to_label):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        idx_to_label: Dictionary mapping indices to label names
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # Overall accuracy
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    
    # Per-class accuracy
    predictions = model.predict(X_val, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    print("\n📈 Per-Class Accuracy:")
    for idx, label in idx_to_label.items():
        mask = y_val == idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_val[mask])
            print(f"   {label}: {class_acc*100:.2f}% ({np.sum(mask)} samples)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    print("\n🔍 Classification Report:")
    print(classification_report(y_val, y_pred, 
                                target_names=[idx_to_label[i] for i in range(len(idx_to_label))]))
    
    cm = confusion_matrix(y_val, y_pred)
    print("\n📊 Confusion Matrix:")
    print("   Predicted →")
    print("   True ↓")
    print("   ", end="")
    for idx in range(len(idx_to_label)):
        print(f"{idx_to_label[idx]:>10}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{idx_to_label[i]:>8}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()
    
    return {
        "loss": loss,
        "accuracy": accuracy,
        "predictions": y_pred,
        "confusion_matrix": cm
    }

if __name__ == "__main__":
    print("🧪 Testing LSTM model with dummy data...")
    
    # Create dummy data
    num_samples = 50
    num_frames = 60
    num_features = 63
    num_classes = 5
    
    X_train = np.random.randn(num_samples, num_frames, num_features).astype(np.float32)
    y_train = np.random.randint(0, num_classes, num_samples)
    
    X_val = np.random.randn(10, num_frames, num_features).astype(np.float32)
    y_val = np.random.randint(0, num_classes, 10)
    
    # Test model creation
    model = create_lstm_model(num_frames, num_features, num_classes)
    
    print("\n✅ Model created successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    
    # Quick training test
    print("\n🧪 Running 2 epochs as smoke test...")
    history = model.fit(X_train, y_train, 
                       validation_data=(X_val, y_val),
                       epochs=2, batch_size=8, verbose=1)
    
    print("\n✅ LSTM model test passed!")
