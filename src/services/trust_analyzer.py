"""
TrustVision AI - Main Analysis Orchestrator

Coordinates all analysis modules:
- Liveness Detection
- Deepfake Detection
- Lip-Sync Verification
- Voice Transcription

Produces a comprehensive trust report for video authenticity.
"""

import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

from src.utils.video_utils import VideoProcessor, validate_video_file
from src.utils.logger import get_logger
from src.models.liveness_detector import LivenessDetector, LivenessResult
from src.models.deepfake_detector import DeepfakeDetector, DeepfakeResult
from src.models.lip_sync_analyzer import LipSyncAnalyzer, LipSyncResult
from src.models.voice_transcriber import VoiceTranscriber, TranscriptionResult
from src.config import settings

logger = get_logger(__name__)


@dataclass
class TrustReport:
    """Complete trust analysis report"""
    job_id: str
    status: str
    processing_time_seconds: float
    trust_score: int  # 0-100
    verdict: str  # AUTHENTIC, SUSPICIOUS, FAKE

    # Individual analysis results
    liveness: Optional[Dict[str, Any]] = None
    deepfake: Optional[Dict[str, Any]] = None
    lip_sync: Optional[Dict[str, Any]] = None
    transcription: Optional[Dict[str, Any]] = None

    # Summary
    flags: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    video_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class TrustAnalyzer:
    """
    Main orchestration service for TrustVision AI

    Coordinates all analysis components and produces
    a comprehensive trust report.

    Analysis Pipeline:
    1. Video Validation
    2. Frame Extraction
    3. Audio Extraction
    4. Parallel Analysis:
       - Liveness Detection (blinks, movement)
       - Deepfake Detection (AI artifacts)
       - Lip-Sync Verification (audio-visual match)
       - Voice Transcription (speech-to-text)
    5. Score Aggregation
    6. Report Generation
    """

    def __init__(self):
        """Initialize all analysis components"""
        logger.info("Initializing TrustAnalyzer...")

        # Initialize detectors (lazy loading for some)
        self.liveness_detector = LivenessDetector()
        self.deepfake_detector = DeepfakeDetector()
        self.lip_sync_analyzer = LipSyncAnalyzer()
        self.voice_transcriber = VoiceTranscriber()

        logger.info("TrustAnalyzer initialized successfully")

    def analyze_video(
        self,
        video_path: Path,
        options: Optional[Dict[str, bool]] = None
    ) -> TrustReport:
        """
        Perform comprehensive trust analysis on video

        Args:
            video_path: Path to video file
            options: Analysis options
                - run_liveness: Run liveness detection (default: True)
                - run_deepfake: Run deepfake detection (default: True)
                - run_lip_sync: Run lip-sync verification (default: True)
                - run_transcription: Run voice transcription (default: True)

        Returns:
            TrustReport with complete analysis results
        """
        job_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(f"Starting analysis job {job_id} for {video_path}")

        # Default options
        options = options or {}
        run_liveness = options.get("run_liveness", True)
        run_deepfake = options.get("run_deepfake", True)
        run_lip_sync = options.get("run_lip_sync", True)
        run_transcription = options.get("run_transcription", True)

        # Validate video
        is_valid, message = validate_video_file(video_path)
        if not is_valid:
            return TrustReport(
                job_id=job_id,
                status="error",
                processing_time_seconds=time.time() - start_time,
                trust_score=0,
                verdict="ERROR",
                flags=[message],
                recommendations=["Please provide a valid video file"]
            )

        # Process video
        try:
            with VideoProcessor(video_path) as vp:
                # Get metadata
                metadata = vp.metadata
                video_metadata = {
                    "duration": metadata.duration,
                    "fps": metadata.fps,
                    "resolution": f"{metadata.width}x{metadata.height}",
                    "frame_count": metadata.frame_count,
                    "has_audio": metadata.has_audio
                }

                # Extract frames
                logger.info("Extracting video frames...")
                frames = list(vp.extract_frames(sample_interval=1.0))  # 1 fps
                frame_data = [(f.frame, f.timestamp) for f in frames]

                logger.info(f"Extracted {len(frames)} frames for analysis")

                # Extract audio if needed
                audio_path = None
                if metadata.has_audio and (run_lip_sync or run_transcription):
                    logger.info("Extracting audio track...")
                    audio_path = vp.extract_audio()

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return TrustReport(
                job_id=job_id,
                status="error",
                processing_time_seconds=time.time() - start_time,
                trust_score=0,
                verdict="ERROR",
                flags=[f"Video processing error: {str(e)}"],
                recommendations=["Check video file format and integrity"]
            )

        # Run analyses
        liveness_result = None
        deepfake_result = None
        lip_sync_result = None
        transcription_result = None

        # Liveness Detection
        if run_liveness:
            logger.info("Running liveness detection...")
            try:
                liveness_result = self.liveness_detector.analyze_video_frames(frame_data)
            except Exception as e:
                logger.error(f"Liveness detection failed: {e}")

        # Deepfake Detection
        if run_deepfake:
            logger.info("Running deepfake detection...")
            try:
                deepfake_result = self.deepfake_detector.analyze_video_frames(frame_data)
            except Exception as e:
                logger.error(f"Deepfake detection failed: {e}")

        # Lip-Sync Verification
        if run_lip_sync and audio_path:
            logger.info("Running lip-sync verification...")
            try:
                lip_sync_result = self.lip_sync_analyzer.analyze(frame_data, audio_path)
            except Exception as e:
                logger.error(f"Lip-sync analysis failed: {e}")

        # Voice Transcription
        if run_transcription and audio_path:
            logger.info("Running voice transcription...")
            try:
                transcription_result = self.voice_transcriber.transcribe(audio_path)
            except Exception as e:
                logger.error(f"Transcription failed: {e}")

        # Cleanup temp audio file
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass

        # Compile report
        report = self._compile_report(
            job_id=job_id,
            start_time=start_time,
            video_metadata=video_metadata,
            liveness=liveness_result,
            deepfake=deepfake_result,
            lip_sync=lip_sync_result,
            transcription=transcription_result
        )

        logger.info(
            f"Analysis complete for job {job_id}: "
            f"Trust Score={report.trust_score}, Verdict={report.verdict}"
        )

        return report

    def _compile_report(
        self,
        job_id: str,
        start_time: float,
        video_metadata: Dict[str, Any],
        liveness: Optional[LivenessResult],
        deepfake: Optional[DeepfakeResult],
        lip_sync: Optional[LipSyncResult],
        transcription: Optional[TranscriptionResult]
    ) -> TrustReport:
        """Compile individual results into comprehensive report"""

        flags = []
        recommendations = []
        score_components = []

        # Process Liveness Result
        liveness_data = None
        if liveness:
            liveness_data = {
                "is_live": liveness.is_live,
                "confidence": round(liveness.confidence, 3),
                "blink_detected": liveness.blink_detected,
                "blink_count": liveness.blink_count,
                "blink_pattern": liveness.blink_pattern,
                "head_movement_detected": liveness.head_movement_detected,
                "movement_magnitude": round(liveness.movement_magnitude, 4),
                "texture_score": round(liveness.texture_score, 3),
                "checks": liveness.checks
            }

            # Score component (0-100)
            liveness_score = int(liveness.confidence * 100) if liveness.is_live else int(liveness.confidence * 50)
            score_components.append(("liveness", liveness_score, 0.3))

            if not liveness.is_live:
                flags.append("Liveness check failed - may be photo or video playback")
                recommendations.append("Ensure person is live in front of camera")
            elif liveness.blink_pattern == "suspicious":
                flags.append("Unusual blink pattern detected")

        # Process Deepfake Result
        deepfake_data = None
        if deepfake:
            deepfake_data = {
                "is_authentic": deepfake.is_authentic,
                "confidence": round(deepfake.confidence, 3),
                "fake_probability": round(deepfake.fake_probability, 3),
                "temporal_consistency": round(deepfake.temporal_consistency, 3),
                "artifact_score": round(deepfake.artifact_score, 3),
                "checks": deepfake.checks
            }

            # Score component
            if deepfake.is_authentic:
                deepfake_score = int(deepfake.confidence * 100)
            else:
                deepfake_score = int((1 - deepfake.fake_probability) * 50)
            score_components.append(("deepfake", deepfake_score, 0.35))

            if not deepfake.is_authentic:
                flags.append("Potential deepfake/manipulation detected")
                recommendations.append("Video may be AI-generated or manipulated")
            elif deepfake.artifact_score > 0.3:
                flags.append("Some visual artifacts detected")

        # Process Lip-Sync Result
        lip_sync_data = None
        if lip_sync:
            lip_sync_data = {
                "is_synchronized": lip_sync.is_synchronized,
                "confidence": round(lip_sync.confidence, 3),
                "sync_score": round(lip_sync.sync_score, 3),
                "correlation": round(lip_sync.correlation, 3),
                "timing_offset_ms": round(lip_sync.timing_offset_ms, 1),
                "dubbing_detected": lip_sync.dubbing_detected,
                "checks": lip_sync.checks
            }

            # Score component
            if lip_sync.is_synchronized:
                lip_sync_score = int(lip_sync.sync_score * 100)
            else:
                lip_sync_score = int(lip_sync.sync_score * 50)
            score_components.append(("lip_sync", lip_sync_score, 0.25))

            if lip_sync.dubbing_detected:
                flags.append("Audio-visual mismatch detected - possible dubbing")
                recommendations.append("Audio may have been replaced or manipulated")
            elif abs(lip_sync.timing_offset_ms) > 200:
                flags.append(f"Audio-video sync offset: {lip_sync.timing_offset_ms:.0f}ms")

        # Process Transcription Result
        transcription_data = None
        if transcription:
            transcription_data = {
                "full_text": transcription.full_text,
                "language": transcription.language,
                "language_confidence": round(transcription.language_confidence, 3),
                "duration": round(transcription.duration, 2),
                "word_count": transcription.word_count,
                "segments": [
                    {
                        "start": round(s.start, 2),
                        "end": round(s.end, 2),
                        "text": s.text
                    }
                    for s in transcription.segments
                ]
            }

            # Transcription doesn't contribute to trust score directly
            # but adds value to the report
            score_components.append(("transcription", 100, 0.1))

        # Calculate overall trust score
        if score_components:
            total_weight = sum(w for _, _, w in score_components)
            trust_score = sum(s * w for _, s, w in score_components) / total_weight
            trust_score = int(trust_score)
        else:
            trust_score = 0

        # Determine verdict
        if trust_score >= 80:
            verdict = "AUTHENTIC"
            if not recommendations:
                recommendations.append("Video appears authentic and unmanipulated")
        elif trust_score >= 50:
            verdict = "SUSPICIOUS"
            if not recommendations:
                recommendations.append("Some concerns detected - manual review recommended")
        else:
            verdict = "FAKE"
            if not recommendations:
                recommendations.append("High likelihood of manipulation - do not trust")

        # Build report
        return TrustReport(
            job_id=job_id,
            status="completed",
            processing_time_seconds=round(time.time() - start_time, 2),
            trust_score=trust_score,
            verdict=verdict,
            liveness=liveness_data,
            deepfake=deepfake_data,
            lip_sync=lip_sync_data,
            transcription=transcription_data,
            flags=flags,
            recommendations=recommendations,
            video_metadata=video_metadata
        )

    def close(self):
        """Release all resources"""
        self.liveness_detector.close()
        self.lip_sync_analyzer.close()
        logger.info("TrustAnalyzer resources released")
