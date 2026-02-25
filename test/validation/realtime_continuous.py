"""
Continuous Real-time Sign Language Recognition
Uses sliding window approach for seamless classification like object detection demos
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import time
import tensorflow as tf
from pathlib import Path
from collections import deque

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent  # Assumes script is in test/validation/ and resources are in test/

MODEL_PATH = str(ROOT_DIR / "models/best_model.keras")
LABEL_MAP_PATH = str(ROOT_DIR / "models/label_map.json")
HAND_LANDMARKER_PATH = str(ROOT_DIR / "hand_landmarker.task")

# Sliding window settings
WINDOW_SIZE = 60  # frames (2 seconds at 30 FPS)
PREDICTION_INTERVAL = 5  # Predict every N frames (for smoother performance)
CONFIDENCE_THRESHOLD = 0.0  # Show all predictions regardless of confidence
SMOOTHING_FRAMES = 3  # Number of predictions to average for stability

def normalize_landmarks(hand_landmarks):
    """Normalize hand landmarks - centers on wrist and scales by hand size"""
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = landmarks_array[0]
    centered = landmarks_array - wrist
    distances = np.linalg.norm(centered, axis=1)
    hand_size = np.max(distances)
    
    if hand_size > 0:
        normalized = centered / hand_size
    else:
        normalized = centered
    
    return normalized.flatten().tolist()

def draw_landmarks_on_image(frame, detection_result):
    """Draw hand landmarks on the frame"""
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    h, w, _ = frame.shape
    
    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            cv2.line(frame, 
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in hand_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)
    
    return frame

class ContinuousSignRecognizer:
    def __init__(self):
        """Initialize the continuous recognizer"""
        
        # Load model
        print(f"📦 Loading model: {MODEL_PATH}")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load labels
        with open(LABEL_MAP_PATH, 'r') as f:
            data = json.load(f)
            self.idx_to_label = {int(k): v for k, v in data['idx_to_label'].items()}
        print(f"🏷️  Classes: {list(self.idx_to_label.values())}")
        
        # Initialize MediaPipe
        print(f"🤚 Initializing MediaPipe...")
        base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Sliding window buffer
        self.landmark_buffer = deque(maxlen=WINDOW_SIZE)
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=SMOOTHING_FRAMES)
        
        # State
        self.current_prediction = None
        self.current_confidence = 0.0
        self.hand_detected = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
    def predict(self):
        """Run prediction on current buffer"""
        if len(self.landmark_buffer) == 0:
            return None, 0.0, None
        
        # Convert buffer to list
        current_data = list(self.landmark_buffer)
        
        # Pad with zeros if we don't have enough frames yet
        if len(current_data) < WINDOW_SIZE:
            padding_needed = WINDOW_SIZE - len(current_data)
            # Pad at the beginning (pre-padding)
            padding = [[0.0] * 63] * padding_needed
            current_data = padding + current_data
            
        # Convert to numpy array
        landmarks_array = np.array(current_data).astype(np.float32)
        
        # Add batch dimension
        input_data = np.expand_dims(landmarks_array, axis=0)
        
        # Predict
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        label = self.idx_to_label[predicted_idx]
        
        return label, confidence, predictions
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction from history"""
        if not self.prediction_history:
            return None, 0.0
        
        # Count votes for each label
        votes = {}
        confidences = {}
        
        for label, conf in self.prediction_history:
            if label not in votes:
                votes[label] = 0
                confidences[label] = []
            votes[label] += 1
            confidences[label].append(conf)
        
        # Get most common prediction
        best_label = max(votes, key=votes.get)
        avg_confidence = np.mean(confidences[best_label])
        
        return best_label, avg_confidence
    
    def process_frame(self, frame):
        """Process a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.landmarker.detect(mp_image)
        
        self.hand_detected = len(detection_result.hand_landmarks) > 0
        
        if self.hand_detected:
            # Extract and normalize landmarks
            normalized = normalize_landmarks(detection_result.hand_landmarks[0])
            self.landmark_buffer.append(normalized)
            
            # Draw landmarks
            frame = draw_landmarks_on_image(frame, detection_result)
        else:
            # Add zeros when no hand detected
            self.landmark_buffer.append([0.0] * 63)
        
        return frame
    
    def draw_ui(self, frame):
        """Draw the UI overlay"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for prediction box
        overlay = frame.copy()
        
        # Top prediction bar
        if self.current_prediction and self.current_confidence >= CONFIDENCE_THRESHOLD:
            # Color based on confidence
            if self.current_confidence >= 0.8:
                color = (0, 200, 0)  # Green - high confidence
            elif self.current_confidence >= 0.6:
                color = (0, 200, 200)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence
            
            cv2.rectangle(overlay, (0, 0), (w, 100), color, -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Prediction text
            cv2.putText(frame, f"{self.current_prediction}", (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            cv2.putText(frame, f"{self.current_confidence*100:.0f}%", (w - 120, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        else:
            # No valid prediction
            cv2.rectangle(overlay, (0, 0), (w, 80), (100, 100, 100), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv2.putText(frame, "Show a sign...", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Bottom info bar
        cv2.rectangle(frame, (0, h - 40), (w, h), (50, 50, 50), -1)
        
        # Hand detection status
        hand_text = "Hand: YES" if self.hand_detected else "Hand: NO"
        hand_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
        cv2.putText(frame, hand_text, (10, h - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        # Buffer status
        buffer_pct = len(self.landmark_buffer) / WINDOW_SIZE * 100
        cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%", (w // 2 - 50, h - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 100, h - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press Q to quit | R to reset", (w // 2 - 120, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop - continuous recognition"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("🎥 CONTINUOUS SIGN LANGUAGE RECOGNITION")
        print("="*60)
        print("\n✨ Just perform signs in front of the camera!")
        print("   The system will continuously recognize them.\n")
        print("Controls:")
        print("   Q - Quit")
        print("   R - Reset buffer")
        print("="*60 + "\n")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Run prediction periodically
            self.frame_count += 1
            # Predict every frame or interval, regardless of buffer size (will be padded)
            if self.frame_count % PREDICTION_INTERVAL == 0:
                result = self.predict()
                if result and result[0] is not None:
                    label, confidence, _ = result
                    print(f"Prediction: {label} ({confidence:.2f})")
                    self.prediction_history.append((label, confidence))
                    
                    # Get smoothed prediction
                    self.current_prediction, self.current_confidence = self.get_smoothed_prediction()
            
            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.landmark_buffer.clear()
                self.prediction_history.clear()
                self.current_prediction = None
                self.current_confidence = 0.0
                print("🔄 Reset")
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        print("\n✅ Session ended")

def main():
    # Check files exist
    for path, name in [(MODEL_PATH, "Model"), (LABEL_MAP_PATH, "Label map"), (HAND_LANDMARKER_PATH, "Hand landmarker")]:
        if not Path(path).exists():
            print(f"❌ {name} not found: {path}")
            return
    
    recognizer = ContinuousSignRecognizer()
    recognizer.run()

if __name__ == "__main__":
    main()
