"""Gaze detection using MediaPipe Face Mesh with iris tracking."""

from dataclasses import dataclass
import cv2
import numpy as np
import mediapipe as mp


@dataclass
class GazeResult:
    """Represents gaze estimation result."""
    looking_at_screen: bool
    looking_at_phone: bool
    head_yaw: float    # left/right rotation in degrees
    head_pitch: float  # up/down rotation in degrees
    eye_gaze_x: float  # normalized horizontal gaze offset (-1 to 1)
    eye_gaze_y: float  # normalized vertical gaze offset (-1 to 1)
    face_detected: bool


class GazeDetector:
    """Detects head pose and eye gaze using MediaPipe Face Mesh."""

    # Key landmark indices
    NOSE_TIP = 1
    CHIN = 199
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # Iris landmarks (from MediaPipe face mesh with refine_landmarks=True)
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263

    def __init__(
        self,
        yaw_threshold: float = 25,
        pitch_threshold: float = -20,
        eye_gaze_threshold: float = 0.3
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.eye_gaze_threshold = eye_gaze_threshold
        self.phone_bbox = None

        # Initialize MediaPipe Face Mesh with iris refinement
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer
            (225.0, 170.0, -135.0),      # Right eye outer
            (-150.0, -150.0, -125.0),    # Left mouth
            (150.0, -150.0, -125.0)      # Right mouth
        ], dtype=np.float64)

    def set_phone_position(self, bbox: tuple[int, int, int, int] | None):
        """Set the phone bounding box for gaze-to-phone detection."""
        self.phone_bbox = bbox

    def _get_head_pose(self, landmarks, frame_shape) -> tuple[float, float, float]:
        """Calculate head pose (yaw, pitch, roll) from landmarks."""
        h, w = frame_shape[:2]

        # Extract 2D points for pose estimation
        image_points = np.array([
            (landmarks[self.NOSE_TIP].x * w, landmarks[self.NOSE_TIP].y * h),
            (landmarks[self.CHIN].x * w, landmarks[self.CHIN].y * h),
            (landmarks[self.LEFT_EYE_OUTER].x * w, landmarks[self.LEFT_EYE_OUTER].y * h),
            (landmarks[self.RIGHT_EYE_OUTER].x * w, landmarks[self.RIGHT_EYE_OUTER].y * h),
            (landmarks[self.LEFT_MOUTH].x * w, landmarks[self.LEFT_MOUTH].y * h),
            (landmarks[self.RIGHT_MOUTH].x * w, landmarks[self.RIGHT_MOUTH].y * h),
        ], dtype=np.float64)

        # Camera matrix approximation
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Get Euler angles
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return yaw, pitch, roll

    def _get_eye_gaze(self, landmarks, frame_shape) -> tuple[float, float]:
        """Calculate eye gaze direction from iris position relative to eye corners."""
        h, w = frame_shape[:2]

        # Get iris and eye corner positions for both eyes
        left_iris = np.array([landmarks[self.LEFT_IRIS_CENTER].x * w,
                              landmarks[self.LEFT_IRIS_CENTER].y * h])
        right_iris = np.array([landmarks[self.RIGHT_IRIS_CENTER].x * w,
                               landmarks[self.RIGHT_IRIS_CENTER].y * h])

        # Left eye corners
        left_eye_left = np.array([landmarks[self.LEFT_EYE_LEFT].x * w,
                                   landmarks[self.LEFT_EYE_LEFT].y * h])
        left_eye_right = np.array([landmarks[self.LEFT_EYE_RIGHT].x * w,
                                    landmarks[self.LEFT_EYE_RIGHT].y * h])

        # Right eye corners
        right_eye_left = np.array([landmarks[self.RIGHT_EYE_LEFT].x * w,
                                    landmarks[self.RIGHT_EYE_LEFT].y * h])
        right_eye_right = np.array([landmarks[self.RIGHT_EYE_RIGHT].x * w,
                                     landmarks[self.RIGHT_EYE_RIGHT].y * h])

        # Calculate iris position as ratio within eye (0 = left corner, 1 = right corner)
        left_eye_width = np.linalg.norm(left_eye_right - left_eye_left)
        right_eye_width = np.linalg.norm(right_eye_right - right_eye_left)

        if left_eye_width < 1 or right_eye_width < 1:
            return 0.0, 0.0

        # Horizontal gaze (average of both eyes)
        left_gaze_x = (left_iris[0] - left_eye_left[0]) / left_eye_width
        right_gaze_x = (right_iris[0] - right_eye_left[0]) / right_eye_width
        gaze_x = (left_gaze_x + right_gaze_x) / 2 - 0.5  # Center at 0

        # Vertical gaze (using eye center as reference)
        left_eye_center_y = (left_eye_left[1] + left_eye_right[1]) / 2
        right_eye_center_y = (right_eye_left[1] + right_eye_right[1]) / 2

        left_gaze_y = (left_iris[1] - left_eye_center_y) / (left_eye_width * 0.5)
        right_gaze_y = (right_iris[1] - right_eye_center_y) / (right_eye_width * 0.5)
        gaze_y = (left_gaze_y + right_gaze_y) / 2

        # Normalize to -1 to 1 range (roughly)
        gaze_x = np.clip(gaze_x * 2, -1, 1)
        gaze_y = np.clip(gaze_y * 2, -1, 1)

        return float(gaze_x), float(gaze_y)

    def _is_looking_at_screen(self, yaw: float, pitch: float, gaze_x: float, gaze_y: float) -> bool:
        """Determine if user is looking at the screen based on head pose and eye gaze."""
        # Check head pose - looking too far left/right or down
        head_looking_away = (
            abs(yaw) > self.yaw_threshold or
            pitch < self.pitch_threshold
        )

        # Check eye gaze - eyes looking too far off center
        eyes_looking_away = (
            abs(gaze_x) > self.eye_gaze_threshold or
            gaze_y > self.eye_gaze_threshold  # Looking down
        )

        # User is looking at screen if BOTH head and eyes are reasonably centered
        # Being lenient: only flag as "not looking" if head OR eyes are significantly off
        return not (head_looking_away or eyes_looking_away)

    def _is_looking_at_phone(self, landmarks, frame_shape, pitch: float, gaze_y: float) -> bool:
        """Determine if user is looking at a detected phone."""
        if self.phone_bbox is None:
            return False

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.phone_bbox

        # Get nose position as approximate gaze point
        nose_x = landmarks[self.NOSE_TIP].x * w
        nose_y = landmarks[self.NOSE_TIP].y * h

        # Check if looking down (head pitch or eye gaze)
        looking_down = pitch < self.pitch_threshold or gaze_y > 0.2

        # Check if phone is in lower portion of frame (typical phone position)
        phone_center_y = (y1 + y2) / 2
        phone_below_face = phone_center_y > nose_y

        return looking_down and phone_below_face

    def detect(self, frame: np.ndarray) -> GazeResult:
        """Detect head pose and eye gaze from frame."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return GazeResult(
                looking_at_screen=False,
                looking_at_phone=False,
                head_yaw=0.0,
                head_pitch=0.0,
                eye_gaze_x=0.0,
                eye_gaze_y=0.0,
                face_detected=False
            )

        landmarks = results.multi_face_landmarks[0].landmark

        # Get head pose
        yaw, pitch, roll = self._get_head_pose(landmarks, frame.shape)

        # Get eye gaze
        gaze_x, gaze_y = self._get_eye_gaze(landmarks, frame.shape)

        # Determine attention state
        looking_at_screen = self._is_looking_at_screen(yaw, pitch, gaze_x, gaze_y)
        looking_at_phone = self._is_looking_at_phone(landmarks, frame.shape, pitch, gaze_y)

        return GazeResult(
            looking_at_screen=looking_at_screen,
            looking_at_phone=looking_at_phone,
            head_yaw=yaw,
            head_pitch=pitch,
            eye_gaze_x=gaze_x,
            eye_gaze_y=gaze_y,
            face_detected=True
        )

    def draw_gaze(self, frame: np.ndarray, gaze: GazeResult) -> np.ndarray:
        """Draw gaze visualization on frame."""
        if not gaze.face_detected:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        # Draw head pose info
        cv2.putText(frame, f"Yaw: {gaze.head_yaw:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {gaze.head_pitch:.1f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw eye gaze info
        cv2.putText(frame, f"Eye X: {gaze.eye_gaze_x:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Eye Y: {gaze.eye_gaze_y:.2f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw attention status
        attention_color = (0, 255, 0) if gaze.looking_at_screen else (0, 0, 255)
        attention_text = "FOCUSED" if gaze.looking_at_screen else "DISTRACTED"
        cv2.putText(frame, attention_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, attention_color, 2)

        return frame
