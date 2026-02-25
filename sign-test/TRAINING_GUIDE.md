# Sign Language Recognition - Quick Start Guide

## 📁 Your Current Setup

You have recorded sign language data with these classes:
- Teacher
- I
- have
- a
- question

Each class should have 30 samples (2-second videos = 60 frames × 63 features).

## 🚀 Step-by-Step Training Process

### Step 1: Verify Your Data

First, check your dataset structure:
```bash
python training/data_preprocessing.py --no-save
```

This will show you:
- How many samples per class
- Data shape and statistics
- Any issues with the data

### Step 2: Preprocess the Data

Create training-ready dataset:
```bash
python training/data_preprocessing.py --dataset-dir sign_language_dataset --output-dir preprocessed --validation-split 0.2 --augment 3
```

Parameters:
- `--dataset-dir`: Your recorded .npy files (default: sign_language_dataset)
- `--output-dir`: Where to save preprocessed data (default: preprocessed)
- `--validation-split`:  of data for validation (default: 0.2 = 20%)
- `--augment`: Data augmentation factor (3 = triple your dataset)
- `--expected-frames`: Frames per sample (default: 60 for 2-second videos)

This creates:
- `preprocessed/X_train.npy` - Training features
- `preprocessed/X_val.npy` - Validation features
- `preprocessed/y_train.npy` - Training labels
- `preprocessed/y_val.npy` - Validation labels
- `preprocessed/label_mappings.json` - Class mappings

### Step 3: Train the Model

Train the LSTM model:
```bash
python training/train_corrected.py --data-dir preprocessed --model-dir models --epochs 100 --batch-size 16
```

Parameters:
- `--data-dir`: Preprocessed data directory
- `--model-dir`: Where to save trained models
- `--epochs`: Maximum training epochs (early stopping will stop earlier if validation accuracy plateaus)
- `--batch-size`: Batch size (16 is good for small datasets, use 32 for larger datasets)

This creates:
- `models/best_model.keras` - Best model (use for inference)
- `models/final_model.keras` - Final epoch model
- `models/label_map.json` - Class labels for inference
- `models/training_history.json` - Training metrics

### Step 4: Test the Model (Optional)

Test the LSTM model architecture:
```bash
python training/lstm_model.py
```

This runs a smoke test with dummy data to verify the model works.

## 📊 Expected Results

With 5 classes and ~30 samples each (augmented to ~90 each):
- **Training samples**: ~360 (after augmentation and split)
- **Validation samples**: ~90
- **Expected accuracy**: 70-95% (depending on sign quality and consistency)

### Good Practices:
- ✅ Perform signs consistently
- ✅ Good lighting
- ✅ Hand clearly visible
- ✅ Similar distance from camera for all recordings

### If Accuracy is Low (<70%):
- Collect more samples per class (aim for 50+)
- Ensure consistent sign execution
- Check if hands are detected in all frames
- Increase data augmentation factor
- Try different hand positions/angles

## 🐛 Troubleshooting

### Error: "No class directories found"
Make sure `sign_language_dataset/` has subdirectories for each class.

### Error: "has X frames, expected 60"
Your recordings are inconsistent. The script will pad/truncate automatically.

### Error: "has X features, expected 63"
Data format is wrong. Make sure you're using the normalized landmarks from `collect_sign_data.py`.

### Low Training Accuracy
- Increase epochs
- Reduce regularization (change `l2(0.001)` to `l2(0.0001)` in lstm_model.py)
- Increase model capacity (change LSTM units from 128/64 to 256/128)

### Overfitting (train accuracy >> val accuracy)
- Increase dropout (change from 0.3 to 0.4 or 0.5)
- Add more data augmentation
- Reduce model capacity

## 🎯 File Structure

```
sign-test/
├── sign_language_dataset/      # Your recorded data
│   ├── Teacher/
│   │   ├── Teacher_000.npy
│   │   ├── Teacher_000.mp4
│   │   └── ...
│   ├── I/
│   ├── have/
│   ├── a/
│   └── question/
├── preprocessed/               # Preprocessed data (created by step 2)
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   └── label_mappings.json
├── models/                     # Trained models (created by step 3)
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── label_map.json
│   └── training_history.json
└── training/
    ├── data_preprocessing.py   # Step 2
    ├── lstm_model.py          # Model architecture
    └── train_corrected.py     # Step 3
```

## 📈 Next Steps After Training

1. **Real-time Inference**: Create a script to use the trained model with your webcam
2. **Mobile Deployment**: Convert model to TensorFlow Lite for your Flutter app
3. **Expand Dataset**: Add more signs and samples
4. **Improve Model**: Try attention mechanisms or transformer architectures

## 🔍 Understanding the Data Flow

1. **collect_sign_data.py** → Records videos and extracts normalized landmarks
2. **data_preprocessing.py** → Loads .npy files, splits train/val, augments
3. **train_corrected.py** → Trains LSTM model on preprocessed data
4. **Inference** → Uses trained model to predict new signs
