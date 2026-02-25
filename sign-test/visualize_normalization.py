import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_normalization(npy_file):
    """
    Visualize the effect of normalization on hand landmarks
    Shows the first frame from a recording
    """
    # Load landmarks
    landmarks = np.load(npy_file)
    
    print(f"Dataset shape: {landmarks.shape}")
    print(f"  - Frames: {landmarks.shape[0]}")
    print(f"  - Features (21 landmarks × 3): {landmarks.shape[1]}")
    
    # Get first frame
    first_frame = landmarks[0].reshape(21, 3)
    
    # Hand landmark connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 5))
    
    # Plot normalized landmarks
    ax = fig.add_subplot(121, projection='3d')
    
    # Plot landmarks
    ax.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], 
               c='red', s=50, label='Landmarks')
    
    # Plot connections
    for connection in HAND_CONNECTIONS:
        start, end = connection
        xs = [first_frame[start, 0], first_frame[end, 0]]
        ys = [first_frame[start, 1], first_frame[end, 1]]
        zs = [first_frame[start, 2], first_frame[end, 2]]
        ax.plot(xs, ys, zs, 'g-', linewidth=1)
    
    # Highlight wrist (should be at origin after normalization)
    ax.scatter([first_frame[0, 0]], [first_frame[0, 1]], [first_frame[0, 2]], 
               c='blue', s=100, marker='*', label='Wrist (origin)')
    
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized)')
    ax.set_title('Normalized Hand Landmarks\n(Scale & Position Invariant)')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = 1.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Print statistics
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    stats_text = f"""
    NORMALIZATION STATISTICS
    ========================
    
    File: {npy_file}
    
    Shape: {landmarks.shape}
    
    First Frame Stats:
    -----------------
    Wrist position: {first_frame[0]}
    (Should be ~[0, 0, 0] after normalization)
    
    Hand size (max distance from wrist):
    {np.max(np.linalg.norm(first_frame, axis=1)):.4f}
    (Should be ~1.0 after normalization)
    
    X range: [{first_frame[:, 0].min():.3f}, {first_frame[:, 0].max():.3f}]
    Y range: [{first_frame[:, 1].min():.3f}, {first_frame[:, 1].max():.3f}]
    Z range: [{first_frame[:, 2].min():.3f}, {first_frame[:, 2].max():.3f}]
    
    KEY BENEFITS:
    ✓ Same scale regardless of distance
    ✓ Centered at wrist (translation invariant)
    ✓ Normalized size (scale invariant)
    ✓ Better model generalization
    """
    
    ax2.text(0.1, 0.5, stats_text, fontfamily='monospace', 
             fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('normalization_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: normalization_visualization.png")
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_normalization.py <path_to_npy_file>")
        print("\nExample:")
        print("  python visualize_normalization.py sign_language_dataset/Teacher/Teacher_000.npy")
    else:
        visualize_normalization(sys.argv[1])
