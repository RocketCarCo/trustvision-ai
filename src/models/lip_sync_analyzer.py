"""
Lip-Sync Verification Module for TrustVision AI

Verifies that audio matches lip movements in video.
Detects dubbed/manipulated videos where audio doesn't match visuals.

Approach:
1. Extract lip landmarks using MediaPipe Face Mesh
2. Calculate lip movement (mouth opening/closing)
3. Extract audio energy/speech activity
4. Correlate lip movement with audio activity
5. High correlation = genuine, Low correlation = dubbed/fake

This is a video-only capability that CANNOT be done with audio alone.
"""

import numpy as np
import cv2
import librosa
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import mediapipe as mp

from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)


@dataclass
class LipFrame:
    """Lip state for a single frame"""
    timestamp: float
    mouth_aspect_ratio: float  # How open the mouth is
    lip_distance: float        # Vertical lip separation


@dataclass
class AudioFrame:
    """Audio activity for a time window"""
    timestamp: float
    energy: float
    is_speech: bool


@dataclass
class LipSyncResult:
    """Lip-sync analysis result"""
    is_synchronized: bool
    confidence: float
    sync_score: float
    correlation: float
    timing_offset_ms: float
    dubbing_detected: bool
    checks: dict = field(default_factory=dict)


class LipSyncAnalyzer:
    """
    Lip-Sync Verification using Audio-Visual Correlation

    This analyzer compares lip movements (from video) with
    audio activity to detect if they are synchronized.

    A genuine video will show high correlation between:
    - Mouth opening/closing patterns
    - Speech audio energy patterns

    Dubbed/manipulated videos will show:
    - Low correlation
    - Timing offsets
    - Inconsistent patterns
    """

    # MediaPipe Face Mesh lip landmark indices
    # Upper lip
    UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    # Lower lip
    LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    # Inner lip (for mouth opening)
    INNER_UPPER_LIP = [13]
    INNER_LOWER_LIP = [14]

    def __init__(self):
        """Initialize the lip-sync analyzer"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("LipSyncAnalyzer initialized")

    def _calculate_mouth_aspect_ratio(
        self,
        landmarks,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float]:
        """
        Calculate mouth aspect ratio and lip distance

        MAR = vertical_distance / horizontal_distance
        Higher MAR = mouth more open

        Returns:
            (mouth_aspect_ratio, lip_distance)
        """
        h, w = frame_shape[:2]

        # Get inner lip points
        upper_lip = landmarks.landmark[13]  # Upper inner lip
        lower_lip = landmarks.landmark[14]  # Lower inner lip
        left_corner = landmarks.landmark[61]   # Left mouth corner
        right_corner = landmarks.landmark[291]  # Right mouth corner

        # Convert to pixel coordinates
        upper_y = upper_lip.y * h
        lower_y = lower_lip.y * h
        left_x = left_corner.x * w
        right_x = right_corner.x * w

        # Calculate distances
        vertical_distance = abs(lower_y - upper_y)
        horizontal_distance = abs(right_x - left_x)

        # Mouth Aspect Ratio
        if horizontal_distance == 0:
            mar = 0.0
        else:
            mar = vertical_distance / horizontal_distance

        return mar, vertical_distance

    def _extract_lip_movement(
        self,
        frames: List[Tuple[np.ndarray, float]]
    ) -> List[LipFrame]:
        """
        Extract lip movement data from video frames

        Args:
            frames: List of (frame, timestamp) tuples

        Returns:
            List of LipFrame with lip state for each frame
        """
        lip_frames = []

        for frame, timestamp in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                mar, lip_distance = self._calculate_mouth_aspect_ratio(
                    face_landmarks, frame.shape
                )

                lip_frames.append(LipFrame(
                    timestamp=timestamp,
                    mouth_aspect_ratio=mar,
                    lip_distance=lip_distance
                ))
            else:
                # No face detected, interpolate or skip
                if lip_frames:
                    # Use previous value
                    lip_frames.append(LipFrame(
                        timestamp=timestamp,
                        mouth_aspect_ratio=lip_frames[-1].mouth_aspect_ratio,
                        lip_distance=lip_frames[-1].lip_distance
                    ))

        return lip_frames

    def _extract_audio_activity(
        self,
        audio_path: Path,
        frame_timestamps: List[float]
    ) -> List[AudioFrame]:
        """
        Extract audio activity/energy at specified timestamps

        Args:
            audio_path: Path to audio file (WAV)
            frame_timestamps: List of timestamps to analyze

        Returns:
            List of AudioFrame with energy at each timestamp
        """
        # Load audio
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return []

        audio_frames = []

        # Calculate frame duration
        hop_length = 512
        frame_length = 2048

        # Get RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Get timestamps for RMS frames
        rms_times = librosa.frames_to_time(
            range(len(rms)),
            sr=sr,
            hop_length=hop_length
        )

        # Speech activity detection (simple energy threshold)
        energy_threshold = np.percentile(rms, 30)

        for timestamp in frame_timestamps:
            # Find closest RMS frame
            idx = np.argmin(np.abs(rms_times - timestamp))

            if idx < len(rms):
                energy = float(rms[idx])
                is_speech = energy > energy_threshold
            else:
                energy = 0.0
                is_speech = False

            audio_frames.append(AudioFrame(
                timestamp=timestamp,
                energy=energy,
                is_speech=is_speech
            ))

        return audio_frames

    def _calculate_correlation(
        self,
        lip_frames: List[LipFrame],
        audio_frames: List[AudioFrame]
    ) -> Tuple[float, float]:
        """
        Calculate correlation between lip movement and audio

        Returns:
            (correlation_coefficient, optimal_offset_ms)
        """
        if len(lip_frames) < 5 or len(audio_frames) < 5:
            return 0.5, 0.0

        # Normalize signals
        lip_signal = np.array([f.mouth_aspect_ratio for f in lip_frames])
        audio_signal = np.array([f.energy for f in audio_frames])

        # Normalize to 0-1 range
        if lip_signal.max() > lip_signal.min():
            lip_signal = (lip_signal - lip_signal.min()) / (lip_signal.max() - lip_signal.min())
        else:
            lip_signal = np.zeros_like(lip_signal)

        if audio_signal.max() > audio_signal.min():
            audio_signal = (audio_signal - audio_signal.min()) / (audio_signal.max() - audio_signal.min())
        else:
            audio_signal = np.zeros_like(audio_signal)

        # Calculate cross-correlation
        correlation = np.corrcoef(lip_signal, audio_signal)[0, 1]

        if np.isnan(correlation):
            correlation = 0.0

        # Find optimal offset using cross-correlation
        # This detects if audio is shifted relative to video
        max_lag = min(10, len(lip_signal) // 4)  # Max 10 frames lag

        best_corr = correlation
        best_offset = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                shifted_lip = lip_signal[-lag:]
                shifted_audio = audio_signal[:lag]
            elif lag > 0:
                shifted_lip = lip_signal[:-lag]
                shifted_audio = audio_signal[lag:]
            else:
                shifted_lip = lip_signal
                shifted_audio = audio_signal

            if len(shifted_lip) > 2 and len(shifted_audio) > 2:
                min_len = min(len(shifted_lip), len(shifted_audio))
                corr = np.corrcoef(shifted_lip[:min_len], shifted_audio[:min_len])[0, 1]
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_offset = lag

        # Convert lag to milliseconds
        if lip_frames and len(lip_frames) > 1:
            frame_duration = (lip_frames[-1].timestamp - lip_frames[0].timestamp) / len(lip_frames)
            offset_ms = best_offset * frame_duration * 1000
        else:
            offset_ms = 0

        return float(best_corr), float(offset_ms)

    def analyze(
        self,
        frames: List[Tuple[np.ndarray, float]],
        audio_path: Path
    ) -> LipSyncResult:
        """
        Analyze lip-sync between video frames and audio

        Args:
            frames: List of (frame, timestamp) tuples
            audio_path: Path to extracted audio file

        Returns:
            LipSyncResult with comprehensive analysis
        """
        if len(frames) < 5:
            return LipSyncResult(
                is_synchronized=False,
                confidence=0.0,
                sync_score=0.0,
                correlation=0.0,
                timing_offset_ms=0.0,
                dubbing_detected=False,
                checks={"error": "Insufficient frames for analysis"}
            )

        # Extract lip movement
        logger.info("Extracting lip movement from video frames")
        lip_frames = self._extract_lip_movement(frames)

        if len(lip_frames) < 5:
            return LipSyncResult(
                is_synchronized=False,
                confidence=0.0,
                sync_score=0.0,
                correlation=0.0,
                timing_offset_ms=0.0,
                dubbing_detected=False,
                checks={"error": "Could not detect face/lips in video"}
            )

        # Extract audio activity
        logger.info("Extracting audio activity")
        timestamps = [f.timestamp for f in lip_frames]
        audio_frames = self._extract_audio_activity(audio_path, timestamps)

        if len(audio_frames) < 5:
            return LipSyncResult(
                is_synchronized=False,
                confidence=0.0,
                sync_score=0.0,
                correlation=0.0,
                timing_offset_ms=0.0,
                dubbing_detected=False,
                checks={"error": "Could not extract audio activity"}
            )

        # Calculate correlation
        correlation, offset_ms = self._calculate_correlation(lip_frames, audio_frames)

        # Determine sync quality
        # High correlation (>0.5) with low offset (<200ms) = good sync
        sync_score = correlation

        # Penalize for large offset
        offset_penalty = min(1.0, abs(offset_ms) / 500)
        sync_score = sync_score * (1 - offset_penalty * 0.3)

        # Determine if synchronized
        is_synchronized = (
            correlation > settings.lip_sync_threshold and
            abs(offset_ms) < 300  # Max 300ms offset
        )

        # Detect dubbing
        dubbing_detected = (
            correlation < 0.3 or
            abs(offset_ms) > 300
        )

        # Calculate confidence
        confidence = abs(correlation)  # Higher correlation = more confident

        # Check for speech presence
        speech_frames = [f for f in audio_frames if f.is_speech]
        speech_ratio = len(speech_frames) / len(audio_frames) if audio_frames else 0

        # If no speech, can't verify sync
        if speech_ratio < 0.1:
            logger.info("Very little speech detected, sync verification limited")
            is_synchronized = True  # Can't verify, assume OK
            confidence = 0.5
            dubbing_detected = False

        return LipSyncResult(
            is_synchronized=is_synchronized,
            confidence=confidence,
            sync_score=sync_score,
            correlation=correlation,
            timing_offset_ms=offset_ms,
            dubbing_detected=dubbing_detected,
            checks={
                "lip_frames_analyzed": len(lip_frames),
                "audio_frames_analyzed": len(audio_frames),
                "speech_ratio": speech_ratio,
                "correlation": correlation,
                "timing_offset_ms": offset_ms,
                "avg_mouth_opening": np.mean([f.mouth_aspect_ratio for f in lip_frames]),
                "avg_audio_energy": np.mean([f.energy for f in audio_frames])
            }
        )

    def close(self):
        """Release resources"""
        self.face_mesh.close()
