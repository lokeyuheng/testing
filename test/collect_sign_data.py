import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import numpy as np

# Configuration
CLASSES = ["a", "have", "I", "question", "Teacher"]
SAMPLES_PER_CLASS = 30
RECORDING_DURATION = 2  # seconds
FPS = 30
OUTPUT_DIR = "sign_language_dataset"
MODEL_PATH = "hand_landmarker.task"  # You'll need to download this model

# Create output directory structure
for class_name in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

def normalize_landmarks(hand_landmarks):
    """
    Normalize hand landmarks to be scale and position invariant
    - Centers landmarks on wrist (landmark 0)
    - Scales by hand size (max distance from wrist)
    Returns normalized x, y, z coordinates as flattened list
    """
    # Extract all landmarks as numpy array
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # Get wrist position (landmark 0)
    wrist = landmarks_array[0]
    
    # Center on wrist (translate so wrist is at origin)
    centered = landmarks_array - wrist
    
    # Calculate hand size (max distance from wrist to any landmark)
    # This makes the normalization scale-invariant
    distances = np.linalg.norm(centered, axis=1)
    hand_size = np.max(distances)
    
    # Avoid division by zero
    if hand_size > 0:
        normalized = centered / hand_size
    else:
        normalized = centered
    
    # Flatten to 1D array [x0, y0, z0, x1, y1, z1, ...]
    return normalized.flatten().tolist()

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks on the image using cv2"""
    annotated_image = np.copy(rgb_image)
    
    # Hand landmark connections (21 landmarks total)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    hand_landmarks_list = detection_result.hand_landmarks
    h, w, _ = annotated_image.shape
    
    # Loop through detected hands
    for hand_landmarks in hand_landmarks_list:
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            start_x, start_y = int(start_point.x * w), int(start_point.y * h)
            end_x, end_y = int(end_point.x * w), int(end_point.y * h)
            
            cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
    
    return annotated_image

def record_sample(cap, landmarker, class_name, sample_num):
    """Record a 3-second sample for a specific class with exactly RECORDING_DURATION * FPS frames"""
    print(f"\nRecording: {class_name} - Sample {sample_num + 1}/{SAMPLES_PER_CLASS}")
    print("Get ready in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("RECORDING NOW!")
    
    # Calculate exact number of frames needed
    EXPECTED_FRAMES = RECORDING_DURATION * FPS  # 60 frames
    
    # Store frames and landmarks
    frames = []
    landmarks_data = []
    frame_count = 0
    start_time = time.time()
    
    # Record exactly EXPECTED_FRAMES
    while frame_count < EXPECTED_FRAMES:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = landmarker.detect(mp_image)
        
        # Extract landmark coordinates
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Normalize landmarks (scale and position invariant)
                normalized_landmarks = normalize_landmarks(hand_landmarks)
                landmarks_data.append(normalized_landmarks)
            
            # Draw landmarks on the frame
            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            # If no hand detected, append zeros
            landmarks_data.append([0.0] * 63)  # 21 landmarks * 3 coordinates
        
        # Add overlay with frame progress
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Recording: {frame_count + 1}/{EXPECTED_FRAMES} frames", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"{class_name} - Sample {sample_num + 1}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store frame
        frames.append(frame)
        frame_count += 1
        
        # Display
        cv2.imshow('Sign Language Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    if len(frames) == 0:
        print("Error: No frames recorded")
        return False
    
    # Ensure exactly EXPECTED_FRAMES by padding or truncating
    if len(frames) < EXPECTED_FRAMES:
        # Pad with last frame if needed
        print(f"Warning: Only {len(frames)} frames captured, padding to {EXPECTED_FRAMES}")
        while len(frames) < EXPECTED_FRAMES:
            frames.append(frames[-1])
            landmarks_data.append(landmarks_data[-1])
    elif len(frames) > EXPECTED_FRAMES:
        # Truncate if somehow we got too many
        print(f"Warning: {len(frames)} frames captured, truncating to {EXPECTED_FRAMES}")
        frames = frames[:EXPECTED_FRAMES]
        landmarks_data = landmarks_data[:EXPECTED_FRAMES]
    
    # Save video
    video_filename = os.path.join(OUTPUT_DIR, class_name, f"{class_name}_{sample_num:03d}.mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    # Save landmarks as numpy array
    landmarks_filename = os.path.join(OUTPUT_DIR, class_name, f"{class_name}_{sample_num:03d}.npy")
    np.save(landmarks_filename, np.array(landmarks_data))
    
    print(f"✓ Saved: {video_filename}")
    print(f"✓ Saved landmarks: {landmarks_filename}")
    print(f"✓ Frame count: {len(frames)} frames (standardized)")
    
    return True

def main():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("\nPlease download the hand landmarker model:")
        print("1. Visit: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models")
        print("2. Download 'hand_landmarker.task'")
        print(f"3. Place it in: {os.path.abspath(MODEL_PATH)}")
        print("\nOr run this command:")
        print("wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("=" * 60)
    print("SIGN LANGUAGE DATA COLLECTION")
    print("=" * 60)
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Recording duration: {RECORDING_DURATION} seconds")
    print(f"Total samples to collect: {len(CLASSES) * SAMPLES_PER_CLASS}")
    print("=" * 60)
    print("\nPress 'q' to quit at any time")
    print("Press 's' to skip current class")
    print("\n")
    
    # Initialize MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for class_name in CLASSES:
            print(f"\n{'='*60}")
            print(f"CLASS: {class_name}")
            print(f"{'='*60}")
            
            for sample_num in range(SAMPLES_PER_CLASS):
                # Wait for user to be ready
                print(f"\nPress SPACE when ready to record sample {sample_num + 1}/{SAMPLES_PER_CLASS}...")
                print("Press 's' to skip this class, 'q' to quit")
                
                while True:
                    success, frame = cap.read()
                    if success:
                        frame = cv2.flip(frame, 1)
                        
                        # Convert BGR to RGB for MediaPipe
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        
                        # Detect hand landmarks
                        detection_result = landmarker.detect(mp_image)
                        
                        # Draw hand landmarks
                        if detection_result.hand_landmarks:
                            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
                            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                        
                        cv2.putText(frame, f"Class: {class_name}", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Sample: {sample_num + 1}/{SAMPLES_PER_CLASS}", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "Press SPACE to start recording", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow('Sign Language Data Collection', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Space bar
                        break
                    elif key == ord('s'):  # Skip class
                        print(f"\nSkipping class: {class_name}")
                        break
                    elif key == ord('q'):  # Quit
                        print("\n\nQuitting...")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                if key == ord('s'):
                    break
                
                # Record the sample
                success = record_sample(cap, landmarker, class_name, sample_num)
                if not success:
                    print("\n\nRecording interrupted. Quitting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                # Brief pause between recordings
                time.sleep(0.5)
            
            print(f"\n✓ Completed class: {class_name}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"Dataset saved in: {OUTPUT_DIR}/")
    print("\nEach class folder contains:")
    print("  - .mp4 files: Video recordings")
    print("  - .npy files: MediaPipe hand landmark coordinates")

if __name__ == "__main__":
    main()
