# Assistive Pickup System v2.0 (Blind-Accessible Conveyor Counter)

## 🎯 Project Overview
The **Assistive Pickup System** is an AI-powered, voice-controlled industrial counter designed specifically for **blind or visually impaired operators**. It leverages computer vision to identify, track, and count mechanical parts (Nuts, Bolts, Bearings) on a conveyor belt, providing real-time audio guidance and batch management.

---

## 🛠️ Technical Architecture

### 1. Vision Stack
*   **Object Detection:** [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) using a custom-trained model (`best.pt`).
*   **Hand Tracking:** [MediaPipe](https://github.com/google/mediapipe) for real-time hand landmark detection.
*   **Interaction Logic:**
    *   **Pinch Detection:** Identifies "pick" actions via hand landmarks.
    *   **Proximity Analysis:** Links objects to the operator's hand.
    *   **ROI & Line Crossing:** Ensures only relevant objects are processed.

### 2. Interaction Stack
*   **Voice Control (STT):** Hands-free operation using Web Speech API.
*   **Audio Feedback (TTS):** Real-time guidance and count confirmations.
*   **Web Interface:** High-contrast, assistive UI for mobile and desktop.

---

## 🚀 Setup & Execution

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python app.py
```
*Default port: `5005`*

---

## 🧠 Training Your Own Model
If you need to re-train the model with new data:

1.  Prepare your dataset in YOLOv8 format (e.g., from Roboflow).
2.  Place the dataset in a directory (e.g., `datasets/MechanicalParts`).
3.  Run the training script:
```bash
python train.py --data_path datasets/MechanicalParts --output_dir Models --epochs 150
```
This will generate a new `best.pt` in the `Models` directory.

---

## 📂 Project Structure
```text
.
├── app.py                # Flask server, YOLO/MediaPipe integration
├── train.py              # Local training script for YOLOv8
├── best.pt               # Custom YOLOv8 weights (Required for app.py)
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Frontend: Voice UI & Camera stream
└── snapshots/            # verification images (auto-generated)
```

---

## 🎙️ Voice Commands
*   **"Start batch [N] nuts [M] bolts [P] bearings"**: Initializes a target-based session.
*   **"Status"**: Reports total and batch progress.
*   **"Pause" / "Resume"**: Toggles activity.
*   **"Undo"**: Reverts the last count.
*   **"Reset"**: Clears all session data.

---

## 🛡️ License
This project is released under the [MIT License](LICENSE).
# PickAssist_Guide
