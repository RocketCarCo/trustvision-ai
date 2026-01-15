"""
Liveness Detection Module for TrustVision AI

Detects if a person in video is live (not a photo, video playback, or mask)

Techniques used:
1. Blink Detection - Natural blink patterns indicate live person
2. Head Movement Analysis - Micro-movements suggest live presence
3. Texture Analysis - Real skin vs printed/screen display

Uses MediaPipe Face Mesh for facial landmark detection.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import mediapipe as mp

from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)


@dataclass
class BlinkEvent:
    """Detected blink event"""
    timestamp: float
    duration_ms: float
    ear_value: float  # Eye Aspect Ratio at blink


@dataclass
class HeadPose:
    """Head pose estimation"""
    timestamp: float
    pitch: float  # Up/down rotation
    yaw: float    # Left/right rotation
    roll: float   # Tilt


@dataclass
class LivenessResult:
    """Complete liveness analysis result"""
    is_live: bool
    confidence: float
    blink_detected: bool
    blink_count: int
    blink_pattern: str  # "natural", "none", "suspicious"
    head_movement_detected: bool
    movement_magnitude: float
    texture_score: float
    checks: Dict[str, any] = field(default_factory=dict)


class LivenessDetector:
    """
    Liveness Detection using facial landmark analysis

    This detector analyzes video frames to determine if a real person
    is present (vs. photo replay attack or video playback attack).

    Key Detection Methods:
    1. Eye Aspect Ratio (EAR) for blink detection
    2. Head pose estimation for movement detection
    3. Facial texture analysis for photo detection
    """

    # MediaPipe Face Mesh landmark indices for eyes
    # Left eye landmarks
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Right eye landmarks
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    # Nose tip and face contour for head pose
    NOSE_TIP_INDEX = 1
    CHIN_INDEX = 199
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # EAR threshold for blink detection
    EAR_THRESHOLD = 0.21
    EAR_CONSECUTIVE_FRAMES = 2

    def __init__(self):
        """Initialize the liveness detector with MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._reset_state()
        logger.info("LivenessDetector initialized with MediaPipe Face Mesh")

    def _reset_state(self):
        """Reset internal state for new video analysis"""
        self.blink_events: List[BlinkEvent] = []
        self.head_poses: List[HeadPose] = []
        self.ear_history: List[Tuple[float, float]] = []  # (timestamp, ear)
        self._blink_counter = 0
        self._consecutive_low_ear = 0

    def analyze_video_frames(
        self,
        frames: List[Tuple[np.ndarray, float]]  # (frame, timestamp)
    ) -> LivenessResult:
        """
        Analyze a sequence of video frames for liveness

        Args:
            frames: List of (frame, timestamp) tuples

        Returns:
            LivenessResult with comprehensive analysis
        """
        self._reset_state()

        if len(frames) < 5:
            logger.warning("Too few frames for reliable liveness detection")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                blink_detected=False,
                blink_count=0,
                blink_pattern="insufficient_data",
                head_movement_detected=False,
                movement_magnitude=0.0,
                texture_score=0.0,
                checks={"error": "Insufficient frames"}
            )

        texture_scores = []

        for frame, timestamp in frames:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # Analyze blinks
                self._analyze_blink(face_landmarks, timestamp, frame.shape)

                # Analyze head pose
                self._analyze_head_pose(face_landmarks, timestamp, frame.shape)

                # Analyze texture
                texture_score = self._analyze_texture(frame, face_landmarks)
                texture_scores.append(texture_score)

        # Compile results
        return self._compile_results(texture_scores)

    def _calculate_ear(
        self,
        landmarks,
        eye_indices: List[int],
        frame_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        When eye is open, EAR is relatively constant
        When eye closes, EAR approaches 0
        """
        h, w = frame_shape[:2]

        # Get eye landmark coordinates
        points = []
        for idx in eye_indices:
            lm = landmarks.landmark[idx]
            points.append((lm.x * w, lm.y * h))

        # Calculate distances
        # Vertical distances
        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

        # Horizontal distance
        h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

        # Calculate EAR
        if h1 == 0:
            return 0.3  # Default open eye value

        ear = (v1 + v2) / (2.0 * h1)
        return ear

    def _analyze_blink(
        self,
        landmarks,
        timestamp: float,
        frame_shape: Tuple[int, int, int]
    ):
        """Detect blinks using Eye Aspect Ratio"""
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES, frame_shape)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES, frame_shape)

        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append((timestamp, ear))

        # Blink detection state machine
        if ear < self.EAR_THRESHOLD:
            self._consecutive_low_ear += 1
        else:
            if self._consecutive_low_ear >= self.EAR_CONSECUTIVE_FRAMES:
                # Blink detected!
                self._blink_counter += 1

                # Calculate blink duration (rough estimate)
                duration = self._consecutive_low_ear * 33  # Assuming ~30fps

                self.blink_events.append(BlinkEvent(
                    timestamp=timestamp,
                    duration_ms=duration,
                    ear_value=ear
                ))
                logger.debug(f"Blink detected at {timestamp:.2f}s (EAR: {ear:.3f})")

            self._consecutive_low_ear = 0

    def _analyze_head_pose(
        self,
        landmarks,
        timestamp: float,
        frame_shape: Tuple[int, int, int]
    ):
        """Estimate head pose from facial landmarks"""
        h, w = frame_shape[:2]

        # Get key landmark points
        nose = landmarks.landmark[self.NOSE_TIP_INDEX]
        chin = landmarks.landmark[self.CHIN_INDEX]
        left_eye = landmarks.landmark[self.LEFT_EYE_OUTER]
        right_eye = landmarks.landmark[self.RIGHT_EYE_OUTER]

        # Convert to image coordinates
        nose_2d = (nose.x * w, nose.y * h)
        chin_2d = (chin.x * w, chin.y * h)

        # Simple head pose estimation
        # Yaw: horizontal difference between eyes
        eye_center_x = (left_eye.x + right_eye.x) / 2
        yaw = (nose.x - eye_center_x) * 90  # Rough degree estimate

        # Pitch: vertical nose-chin relationship
        pitch = (nose.y - 0.5) * 60  # Rough degree estimate

        # Roll: eye line angle
        eye_dx = right_eye.x - left_eye.x
        eye_dy = right_eye.y - left_eye.y
        roll = np.degrees(np.arctan2(eye_dy, eye_dx))

        self.head_poses.append(HeadPose(
            timestamp=timestamp,
            pitch=pitch,
            yaw=yaw,
            roll=roll
        ))

    def _analyze_texture(
        self,
        frame: np.ndarray,
        landmarks
    ) -> float:
        """
        Analyze facial texture for photo detection

        Real faces have natural skin texture
        Photos/screens may show:
        - Moiré patterns
        - Lower texture variance
        - Unnatural frequency spectrum
        """
        h, w = frame.shape[:2]

        # Get face region from landmarks
        face_points = []
        for lm in landmarks.landmark:
            face_points.append((int(lm.x * w), int(lm.y * h)))

        face_points = np.array(face_points)
        x_min, y_min = face_points.min(axis=0)
        x_max, y_max = face_points.max(axis=0)

        # Ensure valid bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # Extract face region
        face_region = frame[y_min:y_max, x_min:x_max]

        if face_region.size == 0:
            return 0.5

        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance (texture measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()

        # Normalize to 0-1 range
        # Higher variance suggests real face, lower suggests photo
        texture_score = min(1.0, texture_variance / 500)

        return texture_score

    def _compile_results(self, texture_scores: List[float]) -> LivenessResult:
        """Compile all analyses into final result"""

        # Blink analysis
        blink_count = len(self.blink_events)
        blink_detected = blink_count >= settings.min_blink_count

        # Determine blink pattern
        if blink_count == 0:
            blink_pattern = "none"
        elif blink_count < 2:
            blink_pattern = "insufficient"
        else:
            # Check for natural blink intervals (typically 2-10 seconds)
            if len(self.blink_events) >= 2:
                intervals = []
                for i in range(1, len(self.blink_events)):
                    interval = self.blink_events[i].timestamp - self.blink_events[i-1].timestamp
                    intervals.append(interval)

                avg_interval = np.mean(intervals)
                if 1.5 < avg_interval < 15:
                    blink_pattern = "natural"
                else:
                    blink_pattern = "suspicious"
            else:
                blink_pattern = "natural"

        # Head movement analysis
        if len(self.head_poses) >= 2:
            # Calculate movement magnitude
            yaw_values = [p.yaw for p in self.head_poses]
            pitch_values = [p.pitch for p in self.head_poses]

            yaw_std = np.std(yaw_values)
            pitch_std = np.std(pitch_values)

            movement_magnitude = (yaw_std + pitch_std) / 2
            head_movement_detected = movement_magnitude > settings.min_head_movement_threshold
        else:
            movement_magnitude = 0.0
            head_movement_detected = False

        # Texture analysis
        texture_score = np.mean(texture_scores) if texture_scores else 0.5

        # Calculate overall confidence
        confidence_factors = []

        # Blink confidence (0-1)
        if blink_detected and blink_pattern == "natural":
            confidence_factors.append(0.9)
        elif blink_detected:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.2)

        # Movement confidence (0-1)
        if head_movement_detected:
            movement_conf = min(1.0, movement_magnitude * 10)
            confidence_factors.append(movement_conf)
        else:
            confidence_factors.append(0.3)

        # Texture confidence (already 0-1)
        confidence_factors.append(texture_score)

        # Weighted average confidence
        weights = [0.4, 0.3, 0.3]  # Blinks most important
        confidence = sum(c * w for c, w in zip(confidence_factors, weights))

        # Final liveness decision
        is_live = (
            confidence >= settings.liveness_confidence_threshold and
            (blink_detected or head_movement_detected)
        )

        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            blink_detected=blink_detected,
            blink_count=blink_count,
            blink_pattern=blink_pattern,
            head_movement_detected=head_movement_detected,
            movement_magnitude=movement_magnitude,
            texture_score=texture_score,
            checks={
                "blink_events": [
                    {"timestamp": b.timestamp, "duration_ms": b.duration_ms}
                    for b in self.blink_events
                ],
                "head_pose_samples": len(self.head_poses),
                "yaw_range": (
                    min(p.yaw for p in self.head_poses) if self.head_poses else 0,
                    max(p.yaw for p in self.head_poses) if self.head_poses else 0
                ),
                "pitch_range": (
                    min(p.pitch for p in self.head_poses) if self.head_poses else 0,
                    max(p.pitch for p in self.head_poses) if self.head_poses else 0
                )
            }
        )

    def close(self):
        """Release resources"""
        self.face_mesh.close()
