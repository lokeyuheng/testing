"""
Emotion Sentiment Data Collection Script
Collects facial blendshape data using MediaPipe FaceLandmarker for emotion classification.
Mirrors the sign-test/collect_sign_data.py pipeline but captures 52 blendshape scores
instead of 63 hand landmark coordinates.

Each sample = 2 seconds at 30 FPS = 60 frames of 52 blendshape scores → .npy shape (60, 52)
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import sys
import numpy as np
import urllib.request

# Configuration
CLASSES = ["happy", "angry", "down", "confused", "questioning", "neutral"]
SAMPLES_PER_CLASS = 30
RECORDING_DURATION = 2  # seconds
FPS = 30
EXPECTED_FRAMES = RECORDING_DURATION * FPS  # 60 frames
OUTPUT_DIR = "emotion_dataset"
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Number of blendshape scores output by MediaPipe FaceLandmarker
NUM_BLENDSHAPES = 52

# Emotion recording instructions
EMOTION_INSTRUCTIONS = {
    "happy": "😊 SMILE! Show teeth, squint cheeks, look genuinely happy",
    "angry": "😤 FROWN! Furrow brows downward, clench jaw, look mad",
    "down": "😞 LOOK SAD! Droop mouth corners down, lower gaze, look dejected",
    "confused": "😕 LOOK CONFUSED! Squint eyes, furrow brows slightly, pucker lips",
    "questioning": "🤔 RAISE EYEBROWS! Open eyes wide, lift brows high, look curious",
    "neutral": "😐 STAY NEUTRAL! Relax your face completely, no expression",
}

# Create output directory structure
for class_name in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)


def download_model():
    """Download the face landmarker model if not present"""
    if os.path.exists(MODEL_PATH):
        return True

    print(f"📥 Downloading face landmarker model...")
    print(f"   URL: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"   ✅ Saved to: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print(f"\n   Manual download:")
        print(f"   1. Visit: https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models")
        print(f"   2. Download 'face_landmarker.task'")
        print(f"   3. Place it in: {os.path.abspath(MODEL_PATH)}")
        return False


def extract_blendshape_scores(face_blendshapes):
    """
    Extract the 52 blendshape scores from MediaPipe FaceLandmarker output.
    Excludes the '_neutral' blendshape (index 0) which is always present.
    
    Returns a list of 52 float scores in consistent order.
    """
    if not face_blendshapes or len(face_blendshapes) == 0:
        return [0.0] * NUM_BLENDSHAPES

    # Get the first face's blendshapes
    blendshapes = face_blendshapes[0]

    # Extract scores, skipping '_neutral' (the first one, index 0)
    scores = []
    for bs in blendshapes:
        if bs.category_name == "_neutral":
            continue
        scores.append(bs.score)

    # Ensure exactly 52 scores
    if len(scores) < NUM_BLENDSHAPES:
        scores.extend([0.0] * (NUM_BLENDSHAPES - len(scores)))
    elif len(scores) > NUM_BLENDSHAPES:
        scores = scores[:NUM_BLENDSHAPES]

    return scores


def get_blendshape_names(face_blendshapes):
    """Get ordered list of blendshape names for reference"""
    if not face_blendshapes or len(face_blendshapes) == 0:
        return []
    names = []
    for bs in face_blendshapes[0]:
        if bs.category_name == "_neutral":
            continue
        names.append(bs.category_name)
    return names


def draw_face_mesh_on_image(rgb_image, detection_result):
    """Draw face landmarks on the image using cv2"""
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    if not detection_result.face_landmarks:
        return annotated_image

    # Draw face landmarks (using a subset for cleaner visualization)
    # Face oval connections
    FACE_OVAL = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
    ]

    for face_landmarks in detection_result.face_landmarks:
        # Draw face oval
        for connection in FACE_OVAL:
            start_idx, end_idx = connection
            if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                start = face_landmarks[start_idx]
                end = face_landmarks[end_idx]
                start_pt = (int(start.x * w), int(start.y * h))
                end_pt = (int(end.x * w), int(end.y * h))
                cv2.line(annotated_image, start_pt, end_pt, (0, 255, 0), 1)

        # Draw key facial landmarks (eyes, nose, mouth)
        key_indices = [
            # Left eye
            33, 133, 160, 144, 153, 154, 155, 157, 158, 159, 161, 163, 246,
            # Right eye
            362, 263, 387, 373, 380, 381, 382, 384, 385, 386, 388, 390, 466,
            # Nose
            1, 2, 4, 5, 6, 168, 197, 195,
            # Mouth
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
            267, 0, 37, 39, 40, 185
        ]
        for idx in key_indices:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                x, y_coord = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_image, (x, y_coord), 2, (0, 0, 255), -1)

    return annotated_image


def record_sample(cap, face_landmarker, class_name, sample_num):
    """Record a 2-second sample for a specific emotion class with exactly EXPECTED_FRAMES frames"""
    instruction = EMOTION_INSTRUCTIONS.get(class_name, f"Show: {class_name}")
    print(f"\nRecording: {class_name} - Sample {sample_num + 1}/{SAMPLES_PER_CLASS}")
    print(f"   {instruction}")
    print("Get ready in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("RECORDING NOW!")

    # Store frames and blendshape data
    frames = []
    blendshape_data = []
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

        # Detect face landmarks and blendshapes
        detection_result = face_landmarker.detect(mp_image)

        # Extract blendshape scores
        if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
            scores = extract_blendshape_scores(detection_result.face_blendshapes)
            blendshape_data.append(scores)

            # Draw face mesh on the frame
            annotated_frame = draw_face_mesh_on_image(rgb_frame, detection_result)
            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            # If no face detected, append zeros
            blendshape_data.append([0.0] * NUM_BLENDSHAPES)

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
        cv2.imshow('Emotion Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    if len(frames) == 0:
        print("Error: No frames recorded")
        return False

    # Ensure exactly EXPECTED_FRAMES by padding or truncating
    if len(blendshape_data) < EXPECTED_FRAMES:
        print(f"Warning: Only {len(blendshape_data)} frames captured, padding to {EXPECTED_FRAMES}")
        while len(blendshape_data) < EXPECTED_FRAMES:
            blendshape_data.append(blendshape_data[-1])
            if len(frames) < EXPECTED_FRAMES:
                frames.append(frames[-1])
    elif len(blendshape_data) > EXPECTED_FRAMES:
        print(f"Warning: {len(blendshape_data)} frames captured, truncating to {EXPECTED_FRAMES}")
        blendshape_data = blendshape_data[:EXPECTED_FRAMES]
        frames = frames[:EXPECTED_FRAMES]

    # Save video
    video_filename = os.path.join(OUTPUT_DIR, class_name, f"{class_name}_{sample_num:03d}.mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

    # Save blendshape data as numpy array — shape (60, 52)
    blendshape_filename = os.path.join(OUTPUT_DIR, class_name, f"{class_name}_{sample_num:03d}.npy")
    np.save(blendshape_filename, np.array(blendshape_data, dtype=np.float32))

    print(f"✓ Saved: {video_filename}")
    print(f"✓ Saved blendshapes: {blendshape_filename}")
    print(f"✓ Shape: {np.array(blendshape_data).shape} (frames × blendshapes)")

    return True


def main():
    # Handle --dry-run flag
    dry_run = "--dry-run" in sys.argv

    # Download model if not exists
    if not download_model():
        return

    if dry_run:
        print("\n🧪 Dry run: Testing MediaPipe FaceLandmarker initialization...")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
            print("✅ FaceLandmarker initialized successfully!")
            print(f"   Classes: {CLASSES}")
            print(f"   Samples per class: {SAMPLES_PER_CLASS}")
            print(f"   Frames per sample: {EXPECTED_FRAMES}")
            print(f"   Blendshapes per frame: {NUM_BLENDSHAPES}")
            print(f"   Output shape per sample: ({EXPECTED_FRAMES}, {NUM_BLENDSHAPES})")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("=" * 60)
    print("EMOTION SENTIMENT DATA COLLECTION")
    print("=" * 60)
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Recording duration: {RECORDING_DURATION} seconds ({EXPECTED_FRAMES} frames)")
    print(f"Blendshape features: {NUM_BLENDSHAPES}")
    print(f"Total samples to collect: {len(CLASSES) * SAMPLES_PER_CLASS}")
    print("=" * 60)
    print("\nPress 'q' to quit at any time")
    print("Press 's' to skip current class")
    print("\n")

    # Initialize MediaPipe FaceLandmarker with blendshapes enabled
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    blendshape_names_saved = False

    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        for class_name in CLASSES:
            print(f"\n{'='*60}")
            print(f"EMOTION: {class_name.upper()}")
            print(f"{'='*60}")
            print(f"   {EMOTION_INSTRUCTIONS.get(class_name, '')}")

            key = None
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

                        # Detect face
                        detection_result = face_landmarker.detect(mp_image)

                        # Save blendshape names once for reference
                        if not blendshape_names_saved and detection_result.face_blendshapes:
                            names = get_blendshape_names(detection_result.face_blendshapes)
                            if names:
                                import json
                                with open(os.path.join(OUTPUT_DIR, "blendshape_names.json"), "w") as f:
                                    json.dump(names, f, indent=2)
                                print(f"   ✓ Saved blendshape names ({len(names)} blendshapes)")
                                blendshape_names_saved = True

                        # Draw face mesh
                        if detection_result.face_landmarks:
                            annotated_frame = draw_face_mesh_on_image(rgb_frame, detection_result)
                            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                        cv2.putText(frame, f"Emotion: {class_name}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Sample: {sample_num + 1}/{SAMPLES_PER_CLASS}", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "Press SPACE to start recording", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        cv2.imshow('Emotion Data Collection', frame)

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
                success = record_sample(cap, face_landmarker, class_name, sample_num)
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
    print("  - .npy files: MediaPipe face blendshape scores (shape: 60×52)")
    print(f"\nNext steps:")
    print(f"  1. Run rule-based detector: python blendshape_emotion_detector.py")
    print(f"  2. Train SVM classifier:    python training/train_emotion_classifier.py")
    print(f"  3. Real-time validation:    python validation/realtime_emotion.py")


if __name__ == "__main__":
    main()
