"""
Visualize training results and model performance
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_training_history(history_path="models/training_history.json"):
    """Plot training and validation metrics"""
    
    if not Path(history_path).exists():
        print(f"❌ Error: {history_path} not found!")
        print("Train the model first: python training/train_corrected.py")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Mark best epoch
    best_epoch = np.argmax(history['val_accuracy']) + 1
    best_acc = max(history['val_accuracy'])
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax1.plot(best_epoch, best_acc, 'g*', markersize=15)
    ax1.text(best_epoch, best_acc + 0.02, f'{best_acc*100:.1f}%', ha='center', fontsize=10)
    
    # Plot loss
    ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to: training_curves.png")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"\nTotal epochs trained: {len(epochs)}")
    print(f"Best validation accuracy: {max(history['val_accuracy'])*100:.2f}% (epoch {best_epoch})")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]*100:.2f}%")
    print(f"Final training accuracy: {history['accuracy'][-1]*100:.2f}%")
    
    # Check for overfitting
    gap = history['accuracy'][-1] - history['val_accuracy'][-1]
    if gap > 0.2:
        print(f"\n⚠️  Possible overfitting detected!")
        print(f"   Training-Validation gap: {gap*100:.1f}%")
        print(f"   Consider: more data, more regularization, or simpler model")
    elif gap > 0.1:
        print(f"\n✅ Slight overfitting (gap: {gap*100:.1f}%)")
        print(f"   This is normal for small datasets")
    else:
        print(f"\n✅ Good generalization (gap: {gap*100:.1f}%)")
    
    print("="*60)

def analyze_confusion_matrix(model_dir="models"):
    """Analyze model predictions from confusion matrix"""
    # This would require loading the model and running predictions
    # For now, just show where to find the evaluation results
    
    print("\n" + "="*60)
    print("📈 MODEL EVALUATION")
    print("="*60)
    
    print("\nTo see detailed evaluation metrics:")
    print("1. The confusion matrix is printed during training")
    print("2. Check the terminal output from train_corrected.py")
    print("3. Or re-evaluate with:")
    print("   python -c \"from training.train_corrected import *; ...\"")
    
    label_map_path = Path(model_dir) / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)
        
        print(f"\n🏷️  Classes in model:")
        for idx, label in label_map['idx_to_label'].items():
            print(f"   {idx}: {label}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--history", type=str, default="models/training_history.json",
                       help="Path to training history JSON")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory containing trained models")
    
    args = parser.parse_args()
    
    print("="*60)
    print("📊 TRAINING VISUALIZATION")
    print("="*60)
    
    plot_training_history(args.history)
    analyze_confusion_matrix(args.model_dir)
    
    print("\n✅ Visualization complete!")
