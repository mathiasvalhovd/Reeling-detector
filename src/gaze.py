"""Gaze detection placeholder - to be implemented later."""

from dataclasses import dataclass
import numpy as np


@dataclass
class GazeResult:
    """Represents gaze estimation result."""
    looking_at_phone: bool


class GazeDetector:
    """Placeholder gaze detector - always returns True."""

    def __init__(self, pitch_threshold: float = -15):
        pass

    def set_phone_position(self, bbox: tuple[int, int, int, int] | None):
        """Placeholder for phone position."""
        pass

    def detect(self, frame: np.ndarray) -> GazeResult:
        """Always returns looking_at_phone=True (placeholder)."""
        return GazeResult(looking_at_phone=True)

    def draw_gaze(self, frame: np.ndarray, gaze: GazeResult) -> np.ndarray:
        """No-op for placeholder."""
        return frame
