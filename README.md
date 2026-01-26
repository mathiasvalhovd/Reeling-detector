# Reeling Detector

Detects when you're looking at your phone and plays a sound to snap you out of it.

## How it Works

1. **Phone Detection**: Uses YOLOv8 with COCO pre-trained weights to detect phones in the webcam feed
2. **Gaze Detection**: Uses MediaPipe Face Mesh to estimate head pose/gaze direction
3. **Trigger Logic**: When both a phone is detected AND you're looking down at it, audio plays
4. **Auto-stop**: Audio stops when either condition breaks (phone out of frame or looking up)

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your audio file at `assets/reeling.mp3` (or update `AUDIO_PATH` in `config.py`)
2. Run the detector:

```bash
python src/main.py
```

3. Press `q` to quit

## Configuration

Edit `config.py` to tune detection parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PHONE_CONFIDENCE_THRESHOLD` | 0.5 | Minimum confidence for phone detection |
| `HEAD_PITCH_THRESHOLD` | -15 | Head pitch angle (degrees) to consider "looking down" |
| `CAMERA_INDEX` | 0 | Webcam device index |
| `AUDIO_PATH` | `assets/reeling.mp3` | Path to audio file to play |
| `AUDIO_LOOP` | true | Loop audio while conditions are met |
| `DETECTION_COOLDOWN` | 0.5 | Seconds before re-triggering after stopping |

## Project Structure

```
├── src/
│   ├── main.py          # Main application loop
│   ├── detector.py      # YOLO phone detection
│   ├── gaze.py          # MediaPipe gaze detection
│   └── audio_player.py  # Audio playback handling
├── assets/              # Place video files here
├── config.py            # Configuration and thresholds
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- Webcam
- CPU (GPU optional but improves performance)
