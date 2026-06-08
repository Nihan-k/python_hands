# python_hands

Hand tracking projects using OpenCV and MediaPipe.

## Projects

### 1. Hand Landmarks (`hand_landmarks.py`)
Displays 21 hand landmarks and connections in real-time from webcam feed.

### 2. Finger Counter (`finger_counter.py`)
Counts raised fingers for each detected hand and displays the count.

### 3. Volume Control (`volume_control.py`)
Controls system volume via gestures:
- **Thumbs up** — volume up
- **Thumbs down** — volume down

## Setup

```bash
pip install opencv-python mediapipe pycaw comtypes
```

The `hand_landmarker.task` model file is required and included in this repo.

## Usage

```bash
python hand_landmarks.py
python finger_counter.py
python volume_control.py
```

Press **q** to quit any program.
