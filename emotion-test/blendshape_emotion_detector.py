"""
Blendshape-Based Emotion Detector (Phase 1 — Rule-Based)

Uses MediaPipe FaceLandmarker blendshape scores to classify emotions
via weighted sums of specific facial action units. No training required.

6 Emotions: happy, angry, down, confused, questioning, neutral

Can be used:
  - Standalone: python blendshape_emotion_detector.py (quick webcam demo)
  - Imported:   from blendshape_emotion_detector import EmotionDetector
"""
import numpy as np
from collections import deque

# ────────────────────────────────────────────────────────────────────
# Blendshape name → index mapping (MediaPipe 52 blendshapes)
# These are in the order MediaPipe returns them (after skipping _neutral)
# ────────────────────────────────────────────────────────────────────
BLENDSHAPE_NAMES = [
    "browDownLeft",       # 0
    "browDownRight",      # 1
    "browInnerUp",        # 2
    "browOuterUpLeft",    # 3
    "browOuterUpRight",   # 4
    "cheekPuff",          # 5
    "cheekSquintLeft",    # 6
    "cheekSquintRight",   # 7
    "eyeBlinkLeft",       # 8
    "eyeBlinkRight",      # 9
    "eyeLookDownLeft",    # 10
    "eyeLookDownRight",   # 11
    "eyeLookInLeft",      # 12
    "eyeLookInRight",     # 13
    "eyeLookOutLeft",     # 14
    "eyeLookOutRight",    # 15
    "eyeLookUpLeft",      # 16
    "eyeLookUpRight",     # 17
    "eyeSquintLeft",      # 18
    "eyeSquintRight",     # 19
    "eyeWideLeft",        # 20
    "eyeWideRight",       # 21
    "jawForward",         # 22
    "jawLeft",            # 23
    "jawOpen",            # 24
    "jawRight",           # 25
    "mouthClose",         # 26
    "mouthDimpleLeft",    # 27
    "mouthDimpleRight",   # 28
    "mouthFrownLeft",     # 29
    "mouthFrownRight",    # 30
    "mouthFunnel",        # 31
    "mouthLeft",          # 32
    "mouthLowerDownLeft", # 33
    "mouthLowerDownRight",# 34
    "mouthPressLeft",     # 35
    "mouthPressRight",    # 36
    "mouthPucker",        # 37
    "mouthRight",         # 38
    "mouthRollLower",     # 39
    "mouthRollUpper",     # 40
    "mouthShrugLower",    # 41
    "mouthShrugUpper",    # 42
    "mouthSmileLeft",     # 43
    "mouthSmileRight",    # 44
    "mouthStretchLeft",   # 45
    "mouthStretchRight",  # 46
    "mouthUpperUpLeft",   # 47
    "mouthUpperUpRight",  # 48
    "noseSneerLeft",      # 49
    "noseSneerRight",     # 50
    "_padding",           # 51  (padding slot — MediaPipe outputs 51 non-neutral blendshapes)
]

# Build name → index lookup
BS_IDX = {name: idx for idx, name in enumerate(BLENDSHAPE_NAMES)}


class EmotionDetector:
    """
    Rule-based emotion detector using MediaPipe blendshape scores.
    
    Usage:
        detector = EmotionDetector()
        emotion, confidence = detector.detect(blendshape_scores_array_52)
        smoothed_emotion, smoothed_conf = detector.get_smoothed()
    """
    
    def __init__(self, smoothing_window=5, confidence_threshold=0.25):
        """
        Args:
            smoothing_window: Number of past frames to average for temporal smoothing
            confidence_threshold: Minimum score to classify as non-neutral
        """
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.history = deque(maxlen=smoothing_window)
        
        # Define emotion rules: each maps specific blendshape indices to weights
        # Scores are computed as weighted average of relevant blendshapes
        self.emotion_rules = self._build_rules()
    
    def _build_rules(self):
        """
        Define blendshape-to-emotion mapping rules.
        Each rule is a dict with 'indices' (blendshape indices) and 'weights'.
        """
        rules = {}
        
        # ── HAPPY: smiling, squinting cheeks ──
        rules["happy"] = {
            "components": [
                ("mouthSmileLeft", 1.5),
                ("mouthSmileRight", 1.5),
                ("cheekSquintLeft", 1.0),
                ("cheekSquintRight", 1.0),
                ("mouthDimpleLeft", 0.5),
                ("mouthDimpleRight", 0.5),
            ],
            "threshold": 0.30,
        }
        
        # ── ANGRY: brows down, frown, jaw tension ──
        rules["angry"] = {
            "components": [
                ("browDownLeft", 1.5),
                ("browDownRight", 1.5),
                ("mouthFrownLeft", 1.0),
                ("mouthFrownRight", 1.0),
                ("jawForward", 0.8),
                ("noseSneerLeft", 0.7),
                ("noseSneerRight", 0.7),
            ],
            "threshold": 0.25,
        }
        
        # ── DOWN (sad): mouth frown, lowered gaze, drooping ──
        rules["down"] = {
            "components": [
                ("mouthFrownLeft", 1.5),
                ("mouthFrownRight", 1.5),
                ("browInnerUp", 1.0),     # inner brow raise (sadness indicator)
                ("eyeLookDownLeft", 0.8),
                ("eyeLookDownRight", 0.8),
                ("mouthPressLeft", 0.5),
                ("mouthPressRight", 0.5),
            ],
            "threshold": 0.25,
        }
        
        # ── CONFUSED: squinting, furrowed brow, puckered lips ──
        rules["confused"] = {
            "components": [
                ("eyeSquintLeft", 1.2),
                ("eyeSquintRight", 1.2),
                ("browDownLeft", 1.0),
                ("browDownRight", 1.0),
                ("mouthPucker", 0.8),
                ("mouthFunnel", 0.5),
                ("mouthLeft", 0.5),       # asymmetric mouth = confusion
            ],
            "threshold": 0.25,
        }
        
        # ── QUESTIONING: raised eyebrows, wide eyes ──
        rules["questioning"] = {
            "components": [
                ("browInnerUp", 1.5),
                ("browOuterUpLeft", 1.2),
                ("browOuterUpRight", 1.2),
                ("eyeWideLeft", 1.0),
                ("eyeWideRight", 1.0),
                ("jawOpen", 0.3),         # slight jaw open (surprise element)
            ],
            "threshold": 0.30,
        }
        
        # Convert component names to indices and normalize weights
        for emotion_name, rule in rules.items():
            indices = []
            weights = []
            for bs_name, weight in rule["components"]:
                if bs_name in BS_IDX:
                    indices.append(BS_IDX[bs_name])
                    weights.append(weight)
            
            total_weight = sum(weights) if weights else 1.0
            rule["indices"] = indices
            rule["weights"] = [w / total_weight for w in weights]  # normalize to sum=1
        
        return rules
    
    def compute_emotion_scores(self, blendshape_scores):
        """
        Compute raw scores for each emotion from a single frame's blendshape vector.
        
        Args:
            blendshape_scores: numpy array or list of 52 blendshape scores
            
        Returns:
            dict mapping emotion name → weighted score (0.0 to 1.0)
        """
        scores = np.array(blendshape_scores, dtype=np.float32)
        
        emotion_scores = {}
        for emotion_name, rule in self.emotion_rules.items():
            weighted_sum = 0.0
            for idx, weight in zip(rule["indices"], rule["weights"]):
                if idx < len(scores):
                    weighted_sum += scores[idx] * weight
            emotion_scores[emotion_name] = float(weighted_sum)
        
        return emotion_scores
    
    def detect(self, blendshape_scores):
        """
        Detect emotion from a single frame's blendshape scores.
        
        Args:
            blendshape_scores: numpy array or list of 52 blendshape scores
            
        Returns:
            (emotion_label, confidence): tuple of string and float
        """
        emotion_scores = self.compute_emotion_scores(blendshape_scores)
        
        # Find the highest scoring emotion
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion]
        
        # Check if it exceeds the emotion-specific threshold
        threshold = self.emotion_rules[best_emotion]["threshold"]
        
        if best_score < threshold:
            result = ("neutral", 1.0 - max(emotion_scores.values()))
        else:
            result = (best_emotion, min(best_score, 1.0))
        
        # Add to history for smoothing
        self.history.append(result)
        
        return result
    
    def detect_batch(self, blendshape_sequence):
        """
        Detect emotions for a sequence of frames (e.g., 60 frames).
        
        Args:
            blendshape_sequence: numpy array of shape (n_frames, 52)
            
        Returns:
            list of (emotion_label, confidence) tuples
        """
        results = []
        for frame_scores in blendshape_sequence:
            results.append(self.detect(frame_scores))
        return results
    
    def get_smoothed(self):
        """
        Get temporally smoothed prediction using majority vote over recent history.
        
        Returns:
            (emotion_label, average_confidence): tuple
        """
        if not self.history:
            return ("neutral", 0.0)
        
        # Count votes
        votes = {}
        confidences = {}
        for emotion, conf in self.history:
            if emotion not in votes:
                votes[emotion] = 0
                confidences[emotion] = []
            votes[emotion] += 1
            confidences[emotion].append(conf)
        
        # Get most common emotion
        best = max(votes, key=votes.get)
        avg_conf = float(np.mean(confidences[best]))
        
        return (best, avg_conf)
    
    def get_dominant_emotion(self, blendshape_sequence):
        """
        Get the single dominant emotion for an entire recording sequence.
        Useful for labeling a 2-second sample.
        
        Args:
            blendshape_sequence: numpy array of shape (n_frames, 52)
            
        Returns:
            (emotion_label, confidence, per_emotion_percentages)
        """
        detections = self.detect_batch(blendshape_sequence)
        
        # Count occurrences of each emotion
        counts = {}
        confs = {}
        for emotion, conf in detections:
            if emotion not in counts:
                counts[emotion] = 0
                confs[emotion] = []
            counts[emotion] += 1
            confs[emotion].append(conf)
        
        total = len(detections)
        percentages = {e: (c / total * 100) for e, c in counts.items()}
        
        dominant = max(counts, key=counts.get)
        avg_conf = float(np.mean(confs[dominant]))
        
        return (dominant, avg_conf, percentages)
    
    def reset(self):
        """Clear the smoothing history"""
        self.history.clear()


def demo_standalone():
    """Quick standalone webcam demo of the rule-based detector"""
    import cv2
    import mediapipe as mp_lib
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import time
    import os
    import urllib.request
    
    MODEL_PATH = "face_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    # Download model if needed
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # Init detector
    detector = EmotionDetector(smoothing_window=8)
    
    # Init MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n" + "=" * 60)
    print("🎭 EMOTION DETECTOR — Rule-Based Demo")
    print("=" * 60)
    print("Press Q to quit\n")
    
    # Emotion colors (BGR)
    EMOTION_COLORS = {
        "happy":       (0, 200, 0),      # Green
        "angry":       (0, 0, 200),       # Red
        "down":        (200, 100, 0),     # Blue-ish
        "confused":    (0, 165, 255),     # Orange
        "questioning": (255, 200, 0),     # Cyan
        "neutral":     (150, 150, 150),   # Gray
    }
    
    EMOTION_EMOJIS = {
        "happy": "😊", "angry": "😤", "down": "😞",
        "confused": "😕", "questioning": "🤔", "neutral": "😐"
    }
    
    fps_counter = 0
    fps = 0
    last_fps_time = time.time()
    
    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_frame)
            
            result = face_landmarker.detect(mp_image)
            
            # Extract blendshapes and detect emotion
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                scores = []
                for bs in result.face_blendshapes[0]:
                    if bs.category_name == "_neutral":
                        continue
                    scores.append(bs.score)
                
                # Pad to 52 if needed
                while len(scores) < 52:
                    scores.append(0.0)
                scores = scores[:52]
                
                # Detect
                raw_emotion, raw_conf = detector.detect(scores)
                smooth_emotion, smooth_conf = detector.get_smoothed()
                
                # Get all scores for debug bar
                all_scores = detector.compute_emotion_scores(scores)
                
                # Draw UI
                h, w = frame.shape[:2]
                
                # Top bar with smoothed prediction
                color = EMOTION_COLORS.get(smooth_emotion, (150, 150, 150))
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 90), color, -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                
                cv2.putText(frame, f"{smooth_emotion.upper()}", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                cv2.putText(frame, f"{smooth_conf*100:.0f}%", (w - 130, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Score bars on the right side
                bar_x = w - 250
                bar_y_start = 110
                bar_height = 25
                bar_max_width = 200
                
                for i, (emotion_name, score) in enumerate(sorted(all_scores.items())):
                    y = bar_y_start + i * (bar_height + 8)
                    bar_width = int(score * bar_max_width)
                    bar_color = EMOTION_COLORS.get(emotion_name, (150, 150, 150))
                    
                    # Background
                    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                                 (50, 50, 50), -1)
                    # Score bar
                    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height),
                                 bar_color, -1)
                    # Label
                    cv2.putText(frame, f"{emotion_name}: {score:.2f}", (bar_x - 5, y + bar_height - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 70), (100, 100, 100), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                cv2.putText(frame, "No face detected", (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # FPS
            fps_counter += 1
            if time.time() - last_fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_fps_time = time.time()
            
            cv2.putText(frame, f"FPS: {fps}", (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Press Q to quit", (w - 180, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Emotion Detector - Rule Based", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo ended")


if __name__ == "__main__":
    demo_standalone()
