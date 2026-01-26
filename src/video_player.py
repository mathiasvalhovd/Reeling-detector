"""Video playback module using OpenCV."""

import cv2
import threading
from pathlib import Path


class VideoPlayer:
    """Plays a video file in a separate window."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self._playing = False
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self) -> bool:
        """
        Start playing the video. Returns False if video file doesn't exist.
        """
        if self._playing:
            return True

        if not self.video_path.exists():
            print(f"Warning: Video file not found: {self.video_path}")
            return False

        self._stop_flag.clear()
        self._playing = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop the video playback."""
        if not self._playing:
            return

        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._playing = False

    def _playback_loop(self):
        """Internal playback loop running in a separate thread."""
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            self._playing = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = int(1000 / fps)

        window_name = "Reeling Video"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while not self._stop_flag.is_set():
            ret, frame = cap.read()

            if not ret:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow(window_name, frame)

            # Check for window close or 'q' key
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)
        self._playing = False
