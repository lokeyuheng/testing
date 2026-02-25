"""
Real-time Continuous Emotion Recognition
Mirrors sign-test/validation/realtime_continuous.py but for facial emotion detection.

Supports two modes:
  1. Rule-based (Phase 1): Uses blendshape_emotion_detector.py — no model files needed
  2. SVM/MLP   (Phase 2): Loads trained .joblib model from models/ directory

Usage:
    python validation/realtime_emotion.py                    # Rule-based (default)
    python validation/realtime_emotion.py --mode ml          # Use trained SVM/MLP model
    python validation/realtime_emotion.py --mode both        # Show side-by-side comparison
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import time
import sys
import os
import argparse
import urllib.request
from collections import deque
from pathlib import Path

# Add parent directory to path so we can import the detector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blendshape_emotion_detector import EmotionDetector

# Configuration
FACE_LANDMARKER_PATH = "face_landmarker.task"
MODEL_DIR = "models"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Sliding window settings
SMOOTHING_FRAMES = 8  # Number of predictions to average


# Emotion display config
EMOTION_COLORS = {
    "happy":       (0, 200, 0),      # Green
    "angry":       (0, 0, 200),      # Red
    "down":        (200, 100, 0),    # Blue-ish
    "confused":    (0, 165, 255),    # Orange
    "questioning": (255, 200, 0),    # Cyan-ish
    "neutral":     (150, 150, 150),  # Gray
}

EMOTION_EMOJIS = {
    "happy": "Happy", "angry": "Angry", "down": "Down/Sad",
    "confused": "Confused", "questioning": "Questioning", "neutral": "Neutral"
}


class ContinuousEmotionRecognizer:
    def __init__(self, mode="ml"):
        """
        Initialize the continuous emotion recognizer.
        
        Args:
            mode: "rules" (Phase 1), "ml" (Phase 2), or "both" (side-by-side)
        """
        self.mode = mode
        
        # Download face landmarker model if needed
        if not os.path.exists(FACE_LANDMARKER_PATH):
            print("📥 Downloading face landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, FACE_LANDMARKER_PATH)
        
        # Initialize rule-based detector
        self.rule_detector = EmotionDetector(smoothing_window=SMOOTHING_FRAMES)
        print("✅ Rule-based detector initialized")
        
        # Initialize ML model if needed
        self.ml_model = None
        self.ml_label_map = None
        self.ml_prediction_history = deque(maxlen=SMOOTHING_FRAMES)
        
        if mode in ["ml", "both"]:
            self._load_ml_model()
        
        # Initialize MediaPipe
        print(f"🔍 Initializing MediaPipe FaceLandmarker...")
        base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # State
        self.current_rule_emotion = "neutral"
        self.current_rule_confidence = 0.0
        self.current_ml_emotion = "neutral"
        self.current_ml_confidence = 0.0
        self.face_detected = False
        self.fps = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.all_emotion_scores = {}
    
    def _load_ml_model(self):
        """Load the trained SVM/MLP model"""
        import joblib
        
        label_map_path = os.path.join(MODEL_DIR, "label_map.json")
        
        if not os.path.exists(label_map_path):
            print(f"⚠️ No label map found at {label_map_path}")
            print(f"   ML mode disabled. Train a model first:")
            print(f"   python training/train_emotion_classifier.py")
            return
        
        # Load label map
        with open(label_map_path, 'r') as f:
            data = json.load(f)
            self.ml_label_map = {int(k): v for k, v in data['idx_to_label'].items()}
        
        # Find the best model file
        model_files = list(Path(MODEL_DIR).glob("*.joblib"))
        if not model_files:
            print(f"⚠️ No .joblib model files found in {MODEL_DIR}/")
            return
        
        # Prefer frame-level model over sequence-level
        frame_models = [f for f in model_files if "frame" in f.name]
        model_path = frame_models[0] if frame_models else model_files[0]
        
        print(f"📦 Loading ML model: {model_path.name}")
        self.ml_model = joblib.load(model_path)
        print(f"🏷️  Classes: {list(self.ml_label_map.values())}")
    
    def extract_blendshapes(self, detection_result):
        """Extract 52 blendshape scores from detection result"""
        if not detection_result.face_blendshapes or len(detection_result.face_blendshapes) == 0:
            return None
        
        scores = []
        for bs in detection_result.face_blendshapes[0]:
            if bs.category_name == "_neutral":
                continue
            scores.append(bs.score)
        
        while len(scores) < 52:
            scores.append(0.0)
        return scores[:52]
    
    def predict_ml(self, blendshape_scores):
        """Run ML model prediction on a single frame"""
        if self.ml_model is None or self.ml_label_map is None:
            return "neutral", 0.0
        
        X = np.array([blendshape_scores], dtype=np.float32)
        
        # Predict with probabilities
        pred_idx = self.ml_model.predict(X)[0]
        label = self.ml_label_map.get(pred_idx, "neutral")
        
        # Get confidence
        if hasattr(self.ml_model, 'predict_proba'):
            proba = self.ml_model.predict_proba(X)[0]
            confidence = float(proba[pred_idx])
        else:
            confidence = 0.8  # default if no probability
        
        # Add to smoothing history
        self.ml_prediction_history.append((label, confidence))
        
        return label, confidence
    
    def get_smoothed_ml(self):
        """Get smoothed ML prediction"""
        if not self.ml_prediction_history:
            return "neutral", 0.0
        
        votes = {}
        confs = {}
        for label, conf in self.ml_prediction_history:
            if label not in votes:
                votes[label] = 0
                confs[label] = []
            votes[label] += 1
            confs[label].append(conf)
        
        best = max(votes, key=votes.get)
        avg_conf = float(np.mean(confs[best]))
        return best, avg_conf
    
    def process_frame(self, frame):
        """Process a single video frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.face_landmarker.detect(mp_image)
        
        self.face_detected = bool(detection_result.face_landmarks and len(detection_result.face_landmarks) > 0)
        
        if self.face_detected:
            # Extract blendshapes
            scores = self.extract_blendshapes(detection_result)
            
            if scores is not None:
                # Rule-based detection
                raw_emotion, raw_conf = self.rule_detector.detect(scores)
                smooth_emotion, smooth_conf = self.rule_detector.get_smoothed()
                self.current_rule_emotion = smooth_emotion
                self.current_rule_confidence = smooth_conf
                self.all_emotion_scores = self.rule_detector.compute_emotion_scores(scores)
                
                # ML detection
                if self.mode in ["ml", "both"] and self.ml_model is not None:
                    self.predict_ml(scores)
                    self.current_ml_emotion, self.current_ml_confidence = self.get_smoothed_ml()
            
            # Draw face landmarks (simplified)
            h, w, _ = frame.shape
            for face_landmarks in detection_result.face_landmarks:
                # Draw key facial points
                key_indices = [33, 133, 362, 263, 1, 61, 291, 199]
                for idx in key_indices:
                    if idx < len(face_landmarks):
                        lm = face_landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame
    
    def draw_ui(self, frame):
        """Draw the UI overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # ── TOP BAR: Primary emotion ──
        if self.mode == "both":
            # Side-by-side comparison
            # Left: Rule-based
            color_rule = EMOTION_COLORS.get(self.current_rule_emotion, (150, 150, 150))
            cv2.rectangle(overlay, (0, 0), (w // 2, 95), color_rule, -1)
            # Right: ML
            color_ml = EMOTION_COLORS.get(self.current_ml_emotion, (150, 150, 150))
            cv2.rectangle(overlay, (w // 2, 0), (w, 95), color_ml, -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Labels
            cv2.putText(frame, "RULES", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"{self.current_rule_emotion.upper()}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(frame, f"{self.current_rule_confidence*100:.0f}%", (w // 2 - 80, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.putText(frame, "SVM/ML", (w // 2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"{self.current_ml_emotion.upper()}", (w // 2 + 10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(frame, f"{self.current_ml_confidence*100:.0f}%", (w - 90, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            # Single mode
            if self.mode == "ml" and self.ml_model is not None:
                emotion = self.current_ml_emotion
                confidence = self.current_ml_confidence
                mode_label = "ML"
            else:
                emotion = self.current_rule_emotion
                confidence = self.current_rule_confidence
                mode_label = "RULES"
            
            color = EMOTION_COLORS.get(emotion, (150, 150, 150))
            
            if self.face_detected:
                cv2.rectangle(overlay, (0, 0), (w, 95), color, -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                
                cv2.putText(frame, f"{emotion.upper()}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                cv2.putText(frame, f"{confidence*100:.0f}%", (w - 130, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frame, f"[{mode_label}]", (w - 130, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            else:
                cv2.rectangle(overlay, (0, 0), (w, 70), (80, 80, 80), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                cv2.putText(frame, "No face detected...", (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # ── RIGHT SIDEBAR: Emotion score bars ──
        if self.face_detected and self.all_emotion_scores:
            bar_x = w - 260
            bar_y_start = 115
            bar_height = 22
            bar_max_width = 210
            
            cv2.putText(frame, "Emotion Scores:", (bar_x, bar_y_start - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for i, (ename, score) in enumerate(sorted(self.all_emotion_scores.items())):
                y = bar_y_start + 5 + i * (bar_height + 10)
                bar_w = int(min(score, 1.0) * bar_max_width)
                color = EMOTION_COLORS.get(ename, (150, 150, 150))
                
                # Background bar
                cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                             (40, 40, 40), -1)
                # Score bar
                if bar_w > 0:
                    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_height),
                                 color, -1)
                # Label
                cv2.putText(frame, f"{ename}: {score:.2f}", (bar_x + 5, y + bar_height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # ── BOTTOM BAR ──
        cv2.rectangle(frame, (0, h - 40), (w, h), (40, 40, 40), -1)
        
        # Face detection status
        face_text = "Face: YES" if self.face_detected else "Face: NO"
        face_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
        cv2.putText(frame, face_text, (10, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        # Mode
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(frame, mode_text, (w // 2 - 60, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 100, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Controls
        cv2.putText(frame, "Q=Quit | R=Reset | M=Switch Mode", (w // 2 - 160, 135 if self.mode != "both" else h - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        return frame
    
    def run(self):
        """Main loop — continuous emotion recognition"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        modes_available = ["rules"]
        if self.ml_model is not None:
            modes_available.extend(["ml", "both"])
        
        current_mode_idx = modes_available.index(self.mode) if self.mode in modes_available else 0
        
        print("\n" + "=" * 60)
        print("🎭 CONTINUOUS EMOTION RECOGNITION")
        print("=" * 60)
        print(f"\n   Mode: {self.mode}")
        print(f"   Available modes: {', '.join(modes_available)}")
        print("\n   Controls:")
        print("     Q — Quit")
        print("     R — Reset detectors")
        print("     M — Switch mode (rules / ml / both)")
        print("=" * 60 + "\n")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process
            frame = self.process_frame(frame)
            
            # FPS
            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
            
            # UI
            frame = self.draw_ui(frame)
            
            cv2.imshow('Emotion Recognition', frame)
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rule_detector.reset()
                self.ml_prediction_history.clear()
                self.current_rule_emotion = "neutral"
                self.current_rule_confidence = 0.0
                self.current_ml_emotion = "neutral"
                self.current_ml_confidence = 0.0
                self.all_emotion_scores = {}
                print("🔄 Reset")
            elif key == ord('m'):
                current_mode_idx = (current_mode_idx + 1) % len(modes_available)
                self.mode = modes_available[current_mode_idx]
                print(f"🔀 Switched to mode: {self.mode}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.face_landmarker.close()
        print("\n✅ Session ended")


def main():
    parser = argparse.ArgumentParser(description="Real-time emotion recognition")
    parser.add_argument("--mode", type=str, default="ml", choices=["rules", "ml", "both"],
                        help="Detection mode: rules (Phase 1), ml (Phase 2 SVM), both (comparison)")
    args = parser.parse_args()
    
    # Check face landmarker exists
    if not os.path.exists(FACE_LANDMARKER_PATH):
        print(f"📥 Face landmarker not found, will download on first run...")
    
    recognizer = ContinuousEmotionRecognizer(mode=args.mode)
    recognizer.run()


if __name__ == "__main__":
    main()
