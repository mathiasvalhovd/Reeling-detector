"""Main application loop for phone + gaze detection."""

import cv2
import time
import sys
from pathlib import Path
from enum import Enum

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from detector import PhoneDetector
from gaze import GazeDetector
from audio_player import AudioPlayer, AlertPlayer


class DistractionState(Enum):
    """Tracks the current distraction alert state."""
    FOCUSED = 0
    DISTRACTED = 1      # Looking away, timer running
    ALERT_1_PLAYED = 2  # First alert played, waiting for focus or second alert
    ALERT_2_PLAYED = 3  # Second alert played


class ReelingDetector:
    """Main application class that orchestrates detection and audio playback."""

    def __init__(self):
        print("Initializing phone detector...")
        self.phone_detector = PhoneDetector(
            model_path=config.YOLO_MODEL,
            confidence_threshold=config.PHONE_CONFIDENCE_THRESHOLD,
            phone_class_id=config.PHONE_CLASS_ID
        )

        print("Initializing gaze detector...")
        self.gaze_detector = GazeDetector(
            yaw_threshold=config.HEAD_YAW_THRESHOLD,
            pitch_threshold=config.HEAD_PITCH_THRESHOLD,
            eye_gaze_threshold=config.EYE_GAZE_THRESHOLD
        )

        print("Initializing audio player...")
        self.audio_player = AudioPlayer(config.AUDIO_PATH, loop=config.AUDIO_LOOP)

        # Distraction tracking
        self.distraction_state = DistractionState.FOCUSED
        self.distraction_start_time = None
        self.alert_1_time = None

        self.last_trigger_time = 0
        self._running = False

    def _handle_distraction(self, looking_at_screen: bool, face_detected: bool):
        """Handle distraction detection and tiered alerts."""
        current_time = time.time()

        # If looking at screen or no face, reset distraction state
        if looking_at_screen or not face_detected:
            if self.distraction_state != DistractionState.FOCUSED:
                print("Back to focus! Resetting distraction state.")
            self.distraction_state = DistractionState.FOCUSED
            self.distraction_start_time = None
            self.alert_1_time = None
            return

        # User is distracted (not looking at screen)
        if self.distraction_state == DistractionState.FOCUSED:
            # Start tracking distraction
            self.distraction_state = DistractionState.DISTRACTED
            self.distraction_start_time = current_time
            print("Distraction detected, starting timer...")

        elif self.distraction_state == DistractionState.DISTRACTED:
            # Check if threshold reached for first alert
            elapsed = current_time - self.distraction_start_time
            if elapsed >= config.DISTRACTION_THRESHOLD:
                print(f"Distracted for {elapsed:.1f}s - Playing Alert 1: 'Focus here brother'")
                AlertPlayer.play_alert(config.DISTRACTION_ALERT_1)
                self.distraction_state = DistractionState.ALERT_1_PLAYED
                self.alert_1_time = current_time

        elif self.distraction_state == DistractionState.ALERT_1_PLAYED:
            # Check if threshold reached for second alert
            elapsed_since_alert1 = current_time - self.alert_1_time
            if elapsed_since_alert1 >= config.SECOND_ALERT_DELAY:
                print(f"Still distracted after alert 1 - Playing Alert 2: 'Bro, get back to work'")
                AlertPlayer.play_alert(config.DISTRACTION_ALERT_2)
                self.distraction_state = DistractionState.ALERT_2_PLAYED

        # ALERT_2_PLAYED: Just wait for user to focus again

    def should_trigger_phone_alert(self, phone_detected: bool, looking_at_phone: bool) -> bool:
        """Determine if phone audio should be playing based on detection state."""
        return phone_detected and looking_at_phone

    def run(self):
        """Main detection loop."""
        print(f"Opening camera {config.CAMERA_INDEX}...")
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        # Try to zoom out / reset zoom to minimum
        cap.set(cv2.CAP_PROP_ZOOM, 0)

        print("Detection started. Press 'q' to quit.")
        print(f"Distraction alert after {config.DISTRACTION_THRESHOLD}s of looking away")
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

                # Pass phone position to gaze detector
                phone_bbox = phone_detections[0].bbox if phone_detected else None
                self.gaze_detector.set_phone_position(phone_bbox)

                # Detect gaze
                gaze_result = self.gaze_detector.detect(frame)

                # Handle distraction alerts (independent of phone)
                self._handle_distraction(
                    gaze_result.looking_at_screen,
                    gaze_result.face_detected
                )

                # Handle phone detection alert
                looking_at_phone = gaze_result.looking_at_phone if gaze_result else False
                should_play_phone_audio = self.should_trigger_phone_alert(phone_detected, looking_at_phone)

                current_time = time.time()
                if should_play_phone_audio and not self.audio_player.is_playing:
                    if current_time - self.last_trigger_time > config.DETECTION_COOLDOWN:
                        print("Phone + eyes on phone detected! Playing audio...")
                        self.audio_player.play()
                elif not should_play_phone_audio and self.audio_player.is_playing:
                    print("Phone conditions no longer met. Stopping audio...")
                    self.audio_player.stop()
                    self.last_trigger_time = current_time

                # Draw visualizations
                frame = self.phone_detector.draw_detections(frame, phone_detections)
                if gaze_result:
                    frame = self.gaze_detector.draw_gaze(frame, gaze_result)

                # Draw distraction timer
                if self.distraction_start_time and self.distraction_state != DistractionState.FOCUSED:
                    elapsed = current_time - self.distraction_start_time
                    timer_text = f"Distracted: {elapsed:.1f}s"
                    cv2.putText(frame, timer_text, (frame.shape[1] - 200, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw alert state
                if self.distraction_state == DistractionState.ALERT_1_PLAYED:
                    cv2.putText(frame, "ALERT 1 PLAYED", (frame.shape[1] - 200, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                elif self.distraction_state == DistractionState.ALERT_2_PLAYED:
                    cv2.putText(frame, "ALERT 2 PLAYED", (frame.shape[1] - 200, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw status
                phone_playing = self.audio_player.is_playing
                status_color = (0, 0, 255) if phone_playing else (0, 255, 0)
                status_text = "PHONE TRIGGERED" if phone_playing else "Monitoring..."
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
            AlertPlayer.cleanup()
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped.")


def main():
    detector = ReelingDetector()
    detector.run()


if __name__ == "__main__":
    main()
