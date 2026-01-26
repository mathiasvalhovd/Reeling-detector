"""Phone detection using YOLO with COCO pre-trained weights."""

from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np


@dataclass
class Detection:
    """Represents a detected phone."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: tuple[int, int]


class PhoneDetector:
    """Detects phones in frames using YOLO."""

    def __init__(self, model_path: str, confidence_threshold: float, phone_class_id: int):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.phone_class_id = phone_class_id

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect phones in the given frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of Detection objects for phones found
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == self.phone_class_id and confidence >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    center=(center_x, center_y)
                ))

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame for visualization."""
        import cv2

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Phone: {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame
