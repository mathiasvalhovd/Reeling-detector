# Detection thresholds
PHONE_CONFIDENCE_THRESHOLD = 0.5
PHONE_CLASS_ID = 67  # COCO class ID for "cell phone"

# Gaze/head pose thresholds
HEAD_YAW_THRESHOLD = 25      # degrees, looking left/right
HEAD_PITCH_THRESHOLD = -20   # degrees, negative = looking down
EYE_GAZE_THRESHOLD = 0.3     # normalized iris offset from center

# Webcam settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# YOLO model settings
YOLO_MODEL = "yolov8n.pt"  # nano model for CPU performance

# Audio playback settings - phone detection
AUDIO_PATH = "assets/reeling.mp3"  # path to audio file for phone detection
AUDIO_LOOP = True  # loop audio while conditions are met

# Audio playback settings - distraction alerts
DISTRACTION_ALERT_1 = "assets/alert1.mp3"  # "Focus here brother"
DISTRACTION_ALERT_2 = "assets/alert2.mp3"  # "Bro, get back to work"

# Distraction timing
DISTRACTION_THRESHOLD = 5.0    # seconds before first alert
SECOND_ALERT_DELAY = 5.0       # seconds after first alert before second alert

# Detection timing
DETECTION_COOLDOWN = 0.5  # seconds before triggering again after stopping
