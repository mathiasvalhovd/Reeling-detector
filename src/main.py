"""Main application loop for phone + gaze detection."""

import cv2
import time
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from detector import PhoneDetector
from gaze import GazeDetector
from audio_player import AudioPlayer


class ReelingDetector:
    """Main application class that orchestrates detection and video playback."""

    def __init__(self):
        print("Initializing phone detector...")
        self.phone_detector = PhoneDetector(
            model_path=config.YOLO_MODEL,
            confidence_threshold=config.PHONE_CONFIDENCE_THRESHOLD,
            phone_class_id=config.PHONE_CLASS_ID
        )

        print("Initializing gaze detector...")
        self.gaze_detector = GazeDetector(
            pitch_threshold=config.HEAD_PITCH_THRESHOLD
        )

        print("Initializing audio player...")
        self.audio_player = AudioPlayer(config.AUDIO_PATH, loop=config.AUDIO_LOOP)

        self.last_trigger_time = 0
        self._running = False

    def should_trigger(self, phone_detected: bool, looking_down: bool) -> bool:
        """Determine if video should be playing based on detection state."""
        return phone_detected and looking_down

    def run(self):
        """Main detection loop."""
        print(f"Opening camera {config.CAMERA_INDEX}...")
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        print("Detection started. Press 'q' to quit.")
        self._running = True

        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Detect phones
                phone_detections = self.phone_detector.detect(frame)
                phone_detected = len(phone_detections) > 0

                # Detect gaze
                gaze_result = self.gaze_detector.detect(frame)
                looking_down = gaze_result.looking_down if gaze_result else False

                # Determine if we should trigger
                should_play = self.should_trigger(phone_detected, looking_down)

                # Handle audio playback
                current_time = time.time()
                if should_play and not self.audio_player.is_playing:
                    if current_time - self.last_trigger_time > config.DETECTION_COOLDOWN:
                        print("Phone + looking down detected! Playing audio...")
                        self.audio_player.play()
                elif not should_play and self.audio_player.is_playing:
                    print("Conditions no longer met. Stopping audio...")
                    self.audio_player.stop()
                    self.last_trigger_time = current_time

                # Draw visualizations
                frame = self.phone_detector.draw_detections(frame, phone_detections)
                if gaze_result:
                    frame = self.gaze_detector.draw_gaze(frame, gaze_result)

                # Draw status
                status_color = (0, 0, 255) if should_play else (0, 255, 0)
                status_text = "TRIGGERED" if should_play else "Monitoring..."
                cv2.putText(frame, status_text, (10, frame.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Draw phone detection status
                phone_status = f"Phone: {'YES' if phone_detected else 'NO'}"
                cv2.putText(frame, phone_status, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Calculate FPS
                fps_counter += 1
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = current_time

                cv2.putText(frame, f"FPS: {current_fps}", (frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show frame
                cv2.imshow("Reeling Detector", frame)

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            self._running = False
            self.audio_player.stop()
            self.audio_player.cleanup()
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped.")


def main():
    detector = ReelingDetector()
    detector.run()


if __name__ == "__main__":
    main()
