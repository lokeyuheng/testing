# AI-Powered Real-Time Bidirectional Voice-To-Hand-Sign System

A mobile Flutter application that enables bidirectional communication between sign language and voice using on-device ML (MediaPipe, LSTM) and cloud-based AI services (Google Gemini, Vertex AI).

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the Sign Language Model](#training-the-sign-language-model)
- [Running the Flutter App](#running-the-flutter-app)
- [Google Cloud Setup for Team](#google-cloud-setup-for-team)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## 🚀 Features

- **Sign-to-Voice**: Real-time hand sign recognition using MediaPipe + LSTM
- **Voice-to-Text**: Speech-to-text conversion (Phase 4 - Coming Soon)
- **Multi-language Support**: 11 languages (English, Chinese, Malay, Tamil, German, Italian, Spanish, Russian, French, Portuguese, Japanese)
- **Cloud AI Integration**: Google Gemini for natural language processing
- **Firebase Authentication**: Google Sign-In and Email/Password
- **On-Device ML**: MediaPipe hand tracking + TensorFlow Lite LSTM model

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│          Flutter Mobile App             │
├─────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────────┐ │
│  │ Sign-to-Voice│  │ Voice-to-Text   │ │
│  │   (Phase 2)  │  │   (Phase 4)     │ │
│  └──────────────┘  └─────────────────┘ │
├─────────────────────────────────────────┤
│         On-Device Processing            │
│  • MediaPipe Hand Landmarker            │
│  • TFLite LSTM Model (29 classes)       │
│  • CPU with XNNPACK optimization        │
├─────────────────────────────────────────┤
│          Cloud Services (Phase 3)       │
│  • Google Gemini Pro API                │
│  • Vertex AI TTS/STT                    │
│  • Cloud Translation API                │
│  • Firebase Realtime Database           │
└─────────────────────────────────────────┘
```

**Current Status**: Phase 2 Complete (Sign-to-Voice detection working)

---

## 📦 Prerequisites

### For Running the App

- **Flutter SDK**: 3.0 or higher
  - [Installation Guide](https://docs.flutter.dev/get-started/install)
- **Android Studio** or **VS Code** with Flutter extension
- **Physical Android Device** (ARM architecture required for MediaPipe)
  - ⚠️ **Emulators are NOT supported** (MediaPipe only provides ARM native libraries)
- **Git**: For cloning the repository

### For Training the AI Model

- **Python**: 3.10
- **Webcam**: For data collection
- **GPU** (optional but recommended): For faster training
  - CUDA-compatible GPU for TensorFlow
  - Or use Google Colab (free GPU)

### For Google Cloud Integration (Phase 3+, already configured)

- **Google Cloud Project** with billing enabled
- **gcloud CLI**: [Installation Guide](https://cloud.google.com/sdk/docs/install)

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai_voice_to_hand_signs_project
```

### 2. Install Flutter Dependencies

```bash
cd flutter-app
flutter pub get
```

### 3. Firebase Configuration (For Teammates)

**Get Firebase config files from the project owner:**

The following files are **not in Git** for security reasons. Request them from the project owner:

1. **`google-services.json`** → Place in `flutter-app/android/app/`
2. **`firebase_options.dart`** → Place in `flutter-app/lib/`

**Note**: The app connects to **production Firebase** by default. You do NOT need to run Firebase emulators.

### 4. Download MediaPipe Model

The MediaPipe hand landmarker model should already be in `flutter-app/android/app/src/main/assets/`. If missing:

```bash
cd flutter-app/android/app/src/main/assets
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## 🤖 Training the Sign Language Model

### Quick Start with Pre-trained Model

The project includes a pre-trained model (`sign_language_model.tflite`) with 29 classes:
- Letters: a-z (excluding i)
- Words: "have", "I", "question", "Teacher"

**Model Performance**: ~96% validation accuracy, ~58% real-time accuracy on mobile phone with tflite

### Training Your Own Model

#### Step 1: Set Up Python Environment

```bash
cd sign-test
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Download MediaPipe Model (If not found)

```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

#### Step 3: Collect Training Data

```bash
python collect_sign_data.py
```

**Instructions**:
1. The script will prompt you for each sign class
2. Hold your hand in the sign position
3. Press **SPACE** to start recording (60 frames over 3 seconds)
4. Repeat 30 times per class (Press s for skipping class and q for quitting)
5. Data saved in `sign_language_dataset/` (Both npy file and mp4 video)

**Tips for Better Data**:
- Use good lighting
- Vary hand positions slightly
- Use front-facing camera (matches app usage)
- Keep hand in frame and stable

#### Step 4: Train the Model

**Local Training** (if you have GPU)

```bash
python training/train_corrected.py
```

**Training Output**:
- `best_model.keras`: Trained Keras model
- `training_history.json`: Loss/accuracy metrics
- Training typically takes 10-30 minutes (specifically 1.5 minute per epoch on a RTX 4060 Laptop GPU)

#### Step 5: Convert Model to TFLite

```bash
python convert_to_tflite.py
```

**Output**:
- `sign_language_model.tflite`: Optimized mobile model
- `label_map.json`: Class index mapping


#### Step 6: Deploy to Flutter App

```bash
# Copy model to Flutter assets
Move the tflite model to the android app under `flutter-app\android\app\src\main\assets`


# If you changed the number of classes, update:
# flutter-app/android/app/src/main/kotlin/.../SignLanguageAnalyzer.kt
# Update the labels list to match your new classes
```

### Validating Model Performance

```bash
# Test on validation data
python training/visualize_training.py

# Real-time testing
python realtime_continuous.py
```

**Expected Metrics**:
- **Training accuracy**: 90-96%
- **Validation accuracy**: 85-95%
- **Real-time accuracy**: 50-70% (varies with lighting/positioning)

---

## 📱 Running the Flutter App

### 1. Connect Physical Device

⚠️ **IMPORTANT**: MediaPipe requires ARM architecture. Emulators will crash!

**Connect your Android phone**:
```bash
# Enable USB debugging on your phone:
# Settings → About Phone → Tap "Build Number" 7 times → Developer Options → USB Debugging

# Verify device connection
flutter devices
```

You should see your device listed.

### 2. Run the App

```bash
cd flutter-app
flutter run
```

**Or run in release mode for better performance**:
```bash
flutter run --release
```

### 3. Using the App

1. **Login**: Use Google Sign-In or create an account
2. **Navigate**: Tap "Sign to Voice" card on dashboard
3. **Grant Permissions**: Allow camera access
4. **Start Recording**: Tap the record button
5. **Make Signs**: Hold each sign for ~2 seconds
6. **View Results**: Detected words appear in the list below
7. **Stop Recording**: Tap stop button

### Build APK (Can ignore this for now)

```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

---

## ☁️ Google Cloud Setup for Team

### For Team Members Making API Calls

Even though you've added teammates as IAM members in Google Cloud Console, **they still need to authenticate locally** to make API calls.

#### Step 1: Install gcloud CLI

**Windows**:
```powershell
# Download installer from:
https://cloud.google.com/sdk/docs/install

# Or use PowerShell:
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe
```

**macOS**:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Linux**:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

#### Step 2: Initialize gcloud

```bash
gcloud init
```

Follow the prompts:
1. Login with your Google account (the one added to IAM)
2. Select project: `ai-real-time-voice-to-sign`
3. Choose default region: `us-central1` (or your preferred region)

#### Step 3: Set Up Application Default Credentials (ADC)

This is **required** for making API calls from your code:

```bash
gcloud auth application-default login
```

This creates credentials at:
- **Windows**: `%APPDATA%\gcloud\application_default_credentials.json`
- **macOS/Linux**: `~/.config/gcloud/application_default_credentials.json`

#### Step 4: Verify IAM Permissions

Your account needs these roles (should already be set by project owner):
- `Vertex AI User`
- `Cloud Translation API User`
- `Firebase Admin SDK Administrator Service Agent`

Verify you have access:
```bash
gcloud projects get-iam-policy ai-real-time-voice-to-sign \
  --flatten="bindings[].members" \
  --filter="bindings.members:user:YOUR_EMAIL@gmail.com"
```

#### Step 5: Enable Required APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  translate.googleapis.com \
  generativelanguage.googleapis.com
```

### Testing Google Cloud Connection

1. Create a test file `test_gemini.py`:

```python
import google.generativeai as genai
import os

# Configure with your API key or use ADC
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Say hello!")
print(response.text)
```
2. Get the secret env file from project owner and add it to the root directory (sign-test/.env)

3. Run:
```bash
python test_gemini.py
```

If successful, you're ready for Phase 3!

---

## 🔧 Troubleshooting

### Flutter App Issues

#### **Error: "libmediapipe_tasks_vision_jni.so not found"**

**Cause**: Running on x86/x86_64 emulator

**Solution**: Use a physical Android device. MediaPipe only supports ARM architectures.

```bash
# Verify your device is ARM
flutter devices
# Should show: arm64-v8a or armeabi-v7a
```

---

#### **Error: "java.lang.NoClassDefFoundError: GpuDelegateFactory$Options"**

**Cause**: GPU delegate compatibility issue

**Solution**: Already fixed in the code. Using CPU with XNNPACK instead.

If issue persists:
```bash
# Clean and rebuild
cd flutter-app
flutter clean
flutter pub get
flutter run
```

---

#### **Error: "A network error... has occurred" (Firebase)**

**Cause**: App trying to connect to Firebase emulator on localhost

**Solution**: 
1. Open `flutter-app/lib/main.dart`
2. Ensure this line is commented out:
   ```dart
   // await FirebaseAuth.instance.useAuthEmulator("localhost", 9099);
   ```
3. Hot restart the app (press `R` in terminal)

---

#### **Low Model Accuracy (<50%)**

**Possible Causes**:
1. **Poor lighting** - Use good, even lighting
2. **Hand too far/close** - Keep hand at medium distance
3. **Motion blur** - Hold signs steady for 2 seconds
4. **Mirroring mismatch** - Train data used front camera flip

**Solutions**:
```bash
# Collect more training data with similar conditions to real usage
python collect_sign_data.py

# Retrain with data augmentation
python training/train_corrected.py
```

**Tuning Parameters** (in `SignLanguageAnalyzer.kt`):
```kotlin
// Lower confidence threshold to catch more predictions
if (maxScore > 0.3f) { // Changed from 0.5f

// Increase inference frequency for faster detection
private val INFERENCE_INTERVAL = 5 // Changed from 10
```

---

### Model Training Issues

#### **Error: "No module named 'tensorflow'"**

```bash
pip install tensorflow
# Or for GPU support:
pip install tensorflow-gpu=2.10.0 (Last tensorflow version with GPU compatibility on Windows)
```
#### **Model Overfitting (Validation < Training accuracy)**

**Solutions**:
1. **Collect more data** - 50+ samples per class instead of 30
2. **Add dropout** - In `lstm_model.py`, increase dropout rate
3. **Data augmentation** - Add random rotations, scales
4. **Reduce model complexity** - Decrease LSTM units

```python
# In lstm_model.py, add more regularization:
model.add(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dropout(0.5))  # Increase from 0.3
```

---

### Google Cloud Issues

#### **Error: "You do not currently have an active account"**

```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### **Error: "Permission denied" when calling APIs**

**Check IAM permissions**:
1. Go to [Google Cloud Console](https://console.cloud.google.com/iam-admin/iam)
2. Find your email
3. Verify you have required roles
4. Contact project owner if missing permissions

#### **Error: "API not enabled"**

```bash
# Enable all required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  translate.googleapis.com \
  generativelanguage.googleapis.com \
  firestore.googleapis.com \
  firebase.googleapis.com
```

---

### Common Build Errors

#### **Gradle build fails with "Execution failed for task ':app:compileDebugKotlin'"**

```bash
cd flutter-app/android
./gradlew clean
cd ..
flutter clean
flutter pub get
flutter run
```

#### **"SDK location not found"**

Create `android/local.properties`:
```properties
sdk.dir=C:\\Users\\YOUR_USERNAME\\AppData\\Local\\Android\\Sdk
# Or on macOS/Linux:
# sdk.dir=/Users/YOUR_USERNAME/Library/Android/sdk
```

---

## 📁 Project Structure

```
ai_voice_to_hand_signs_project/
├── flutter-app/                          # Flutter mobile application
│   ├── android/
│   │   └── app/
│   │       ├── src/main/
│   │       │   ├── kotlin/.../          # Native MediaPipe + TFLite
│   │       │   │   ├── MainActivity.kt
│   │       │   │   ├── SignLanguageAnalyzer.kt
│   │       │   │   ├── NativeCameraView.kt
│   │       │   │   └── NativeViewFactory.kt
│   │       │   └── assets/
│   │       │       ├── hand_landmarker.task    # MediaPipe model
│   │       │       └── sign_language_model.tflite  # LSTM model
│   │       └── build.gradle.kts         # Android dependencies
│   ├── lib/
│   │   ├── features/
│   │   │   ├── auth/                    # Firebase authentication
│   │   │   ├── dashboard/               # Main app screen
│   │   │   └── sign_to_voice/
│   │   │       ├── controllers/         # GetX state management
│   │   │       ├── screens/             # UI screens
│   │   │       ├── services/            # Camera, LSTM services
│   │   │       └── widgets/             # Reusable components
│   │   ├── util/
│   │   │   └── constants/               # Colors, themes
│   │   └── main.dart                    # App entry point
│   ├── assets/
│   │   └── label_map.json              # Class index mapping
│   └── pubspec.yaml                    # Flutter dependencies
│
├── sign-test/                           # Model training pipeline
│   ├── collect_sign_data.py            # Data collection script
│   ├── hand_landmarker.task            # MediaPipe model
│   ├── convert_to_tflite.py            # Keras → TFLite conversion
│   ├── realtime_continuous.py          # Real-time testing
│   ├── training/
│   │   ├── train_corrected.py          # LSTM training script
│   │   ├── data_preprocessing.py       # Data loaders
│   │   ├── lstm_model.py               # Model architecture
│   │   └── visualize_training.py       # Performance plots
│   └── sign_language_dataset/          # Training data (60 frames/sample)
│       ├── a/
│       ├── b/
│       └── ...
│
└── README.md                            # This file
```

---

## 🎯 Development Roadmap

- [x] **Phase 1**: Model Deployment ✅
- [x] **Phase 2**: Sign-to-Voice Feature ✅
  - [x] Native MediaPipe integration
  - [x] TFLite LSTM inference
  - [x] Real-time detection UI
- [ ] **Phase 3**: Google Cloud Integration (Next)
  - [ ] Gemini Pro API for NLG
  - [ ] Vertex AI TTS/STT
  - [ ] gRPC communication
- [ ] **Phase 4**: Voice-to-Text Feature
- [ ] **Phase 5**: Firebase Realtime Database
- [ ] **Phase 6**: Multi-language Translation

---

## 📊 Model Specifications

**LSTM Model**:
- **Input**: 60 frames × 63 features (21 hand landmarks × 3 coordinates)
- **Architecture**: 2 LSTM layers (128, 64 units) + Dense layers
- **Output**: 29 classes (a-z, have, I, question, Teacher)
- **Preprocessing**: Wrist-centered, scale-normalized
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical crossentropy

**Performance**:
- Training accuracy: ~96%
- Validation accuracy: ~95%
- Real-time accuracy: ~58-70% (varies with conditions)
- Inference time: ~50-80ms on device (CPU)

---

## 👥 Team Collaboration

### Roles

- **ML Engineers**: Model training, data collection, preprocessing
- **Flutter Developers**: UI/UX, platform channels, state management
- **Cloud Engineers**: Google Cloud setup, API integration, gRPC

### Workflow

1. **ML Team**: Train model → Convert to TFLite → Share in `assets/`
2. **Flutter Team**: Integrate model → Test on device → Deploy
3. **Cloud Team**: Set up APIs → Share credentials → Integrate services

### Best Practices

- **Version Control**: Use Git branches for features
- **Code Review**: PR review before merging to main
- **Testing**: Test on physical devices before deploying
- **Documentation**: Update README when changing architecture

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review Flutter/MediaPipe official docs
3. Contact team lead for Google Cloud access issues

---

## 📄 License

This project is for educational/hackathon purposes.

---

## 🙏 Acknowledgments

- **MediaPipe**: Hand landmark detection
- **TensorFlow Lite**: On-device ML inference
- **Google Cloud**: AI/ML APIs
- **Flutter**: Cross-platform mobile framework
- **Firebase**: Authentication and backend services
