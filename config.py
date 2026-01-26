# Detection thresholds
PHONE_CONFIDENCE_THRESHOLD = 0.5
PHONE_CLASS_ID = 67  # COCO class ID for "cell phone"

# Gaze/head pose thresholds (to be tuned)
HEAD_PITCH_THRESHOLD = -15  # degrees, negative = looking down
GAZE_DOWN_THRESHOLD = 0.3   # normalized value

# Webcam settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLO model settings
YOLO_MODEL = "yolov8n.pt"  # nano model for CPU performance

# Video playback settings
VIDEO_PATH = "assets/reeling.mp4"  # path to video to play

# Detection timing
DETECTION_COOLDOWN = 0.5  # seconds before triggering again after stopping
