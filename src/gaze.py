"""Gaze detection using MediaPipe Face Mesh (to be implemented)."""

from dataclasses import dataclass
import numpy as np


@dataclass
class GazeResult:
    """Represents gaze/head pose estimation result."""
    pitch: float  # head tilt up/down in degrees
    yaw: float    # head rotation left/right in degrees
    looking_down: bool


class GazeDetector:
    """Detects gaze direction using MediaPipe Face Mesh."""

    def __init__(self, pitch_threshold: float):
        self.pitch_threshold = pitch_threshold
        # TODO: Initialize MediaPipe Face Mesh
        self._initialized = False

    def detect(self, frame: np.ndarray) -> GazeResult | None:
        """
        Estimate head pose / gaze direction.

        Args:
            frame: BGR image from OpenCV

        Returns:
            GazeResult or None if no face detected
        """
        # TODO: Implement MediaPipe Face Mesh detection
        # For now, return a placeholder that always says "looking down"
        # This will be replaced with actual implementation
        return GazeResult(pitch=-20.0, yaw=0.0, looking_down=True)

    def draw_gaze(self, frame: np.ndarray, gaze: GazeResult) -> np.ndarray:
        """Draw gaze indicators on frame for visualization."""
        import cv2

        status = "Looking DOWN" if gaze.looking_down else "Looking UP"
        color = (0, 0, 255) if gaze.looking_down else (0, 255, 0)
        cv2.putText(frame, f"{status} (pitch: {gaze.pitch:.1f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame
