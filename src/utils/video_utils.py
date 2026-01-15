"""
Video Processing Utilities for TrustVision AI

Handles:
- Video file validation
- Frame extraction at configurable intervals
- Audio extraction for transcription
- Video metadata extraction
"""

import cv2
import numpy as np
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoMetadata:
    """Video file metadata"""
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    has_audio: bool


@dataclass
class ExtractedFrame:
    """Extracted frame with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_number: int


class VideoProcessor:
    """
    Video processing utility class

    Handles video file operations including:
    - Metadata extraction
    - Frame sampling at configurable intervals
    - Audio extraction for speech analysis
    """

    def __init__(self, video_path: Path):
        """
        Initialize video processor

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        """Open video file for processing"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        logger.info(f"Opened video: {self.video_path}")

    def close(self) -> None:
        """Release video capture resources"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata (cached)"""
        if self._metadata is None:
            self._metadata = self._extract_metadata()
        return self._metadata

    def _extract_metadata(self) -> VideoMetadata:
        """Extract video file metadata"""
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        # Check for audio stream using ffprobe
        has_audio = self._check_audio_stream()

        return VideoMetadata(
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            has_audio=has_audio
        )

    def _check_audio_stream(self) -> bool:
        """Check if video has audio stream using ffprobe"""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    str(self.video_path)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "audio" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("ffprobe not available, assuming video has audio")
            return True

    def extract_frames(
        self,
        sample_interval: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Generator[ExtractedFrame, None, None]:
        """
        Extract frames from video at specified intervals

        Args:
            sample_interval: Seconds between frame samples (default from settings)
            max_frames: Maximum number of frames to extract

        Yields:
            ExtractedFrame objects with frame data and timestamp
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")

        interval = sample_interval or settings.frame_sample_rate
        meta = self.metadata

        # Calculate frame positions to sample
        frame_interval = int(meta.fps * interval)
        frame_positions = range(0, meta.frame_count, max(1, frame_interval))

        if max_frames:
            frame_positions = list(frame_positions)[:max_frames]

        frames_extracted = 0
        for frame_pos in frame_positions:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = self._cap.read()

            if not ret:
                logger.warning(f"Failed to read frame at position {frame_pos}")
                continue

            timestamp = frame_pos / meta.fps

            yield ExtractedFrame(
                frame=frame,
                timestamp=timestamp,
                frame_number=frame_pos
            )

            frames_extracted += 1

        logger.info(f"Extracted {frames_extracted} frames from video")

    def extract_all_frames(self) -> Generator[ExtractedFrame, None, None]:
        """
        Extract ALL frames from video (for detailed analysis)

        Use with caution - generates many frames!

        Yields:
            ExtractedFrame objects for every frame
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")

        meta = self.metadata
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_number = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            timestamp = frame_number / meta.fps
            yield ExtractedFrame(
                frame=frame,
                timestamp=timestamp,
                frame_number=frame_number
            )
            frame_number += 1

    def extract_audio(self, output_path: Optional[Path] = None) -> Path:
        """
        Extract audio track from video file

        Args:
            output_path: Optional path for audio file (default: temp file)

        Returns:
            Path to extracted audio file (WAV format)
        """
        if output_path is None:
            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir=settings.upload_dir
            )
            output_path = Path(temp_file.name)
            temp_file.close()

        try:
            # Use ffmpeg to extract audio
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(self.video_path),
                    "-vn",  # No video
                    "-acodec", "pcm_s16le",  # PCM format for Whisper
                    "-ar", "16000",  # 16kHz sample rate
                    "-ac", "1",  # Mono
                    str(output_path)
                ],
                capture_output=True,
                check=True,
                timeout=60
            )
            logger.info(f"Extracted audio to: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract audio: {e}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def get_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get a single frame at specific timestamp

        Args:
            timestamp: Time in seconds

        Returns:
            Frame as numpy array or None if failed
        """
        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")

        frame_pos = int(timestamp * self.metadata.fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        ret, frame = self._cap.read()
        return frame if ret else None


def validate_video_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate video file for processing

    Args:
        file_path: Path to video file

    Returns:
        Tuple of (is_valid, message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)"

    # Check file extension
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if file_path.suffix.lower() not in valid_extensions:
        return False, f"Invalid file type: {file_path.suffix}"

    # Try to open video
    try:
        with VideoProcessor(file_path) as vp:
            meta = vp.metadata

            if meta.duration > settings.max_video_duration:
                return False, f"Video too long: {meta.duration:.1f}s (max: {settings.max_video_duration}s)"

            if meta.duration < 1:
                return False, "Video too short (minimum 1 second)"

            if meta.width < 100 or meta.height < 100:
                return False, f"Video resolution too low: {meta.width}x{meta.height}"

    except Exception as e:
        return False, f"Cannot process video: {str(e)}"

    return True, "Video file is valid"
