"""
Pydantic Schemas for TrustVision AI API

Defines request/response models for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AnalysisOptions(BaseModel):
    """Options for video analysis"""
    run_liveness: bool = Field(default=True, description="Run liveness detection")
    run_deepfake: bool = Field(default=True, description="Run deepfake detection")
    run_lip_sync: bool = Field(default=True, description="Run lip-sync verification")
    run_transcription: bool = Field(default=True, description="Run voice transcription")


class LivenessAnalysis(BaseModel):
    """Liveness detection results"""
    is_live: bool
    confidence: float
    blink_detected: bool
    blink_count: int
    blink_pattern: str
    head_movement_detected: bool
    movement_magnitude: float
    texture_score: float
    checks: Dict[str, Any] = Field(default_factory=dict)


class DeepfakeAnalysis(BaseModel):
    """Deepfake detection results"""
    is_authentic: bool
    confidence: float
    fake_probability: float
    temporal_consistency: float
    artifact_score: float
    checks: Dict[str, Any] = Field(default_factory=dict)


class LipSyncAnalysis(BaseModel):
    """Lip-sync verification results"""
    is_synchronized: bool
    confidence: float
    sync_score: float
    correlation: float
    timing_offset_ms: float
    dubbing_detected: bool
    checks: Dict[str, Any] = Field(default_factory=dict)


class TranscriptionSegment(BaseModel):
    """A segment of transcribed speech"""
    start: float
    end: float
    text: str


class TranscriptionAnalysis(BaseModel):
    """Voice transcription results"""
    full_text: str
    language: str
    language_confidence: float
    duration: float
    word_count: int
    segments: List[TranscriptionSegment] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    """Video file metadata"""
    duration: float
    fps: float
    resolution: str
    frame_count: int
    has_audio: bool


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status: completed, error, processing")
    processing_time_seconds: float = Field(description="Processing duration")
    trust_score: int = Field(ge=0, le=100, description="Overall trust score 0-100")
    verdict: str = Field(description="AUTHENTIC, SUSPICIOUS, or FAKE")

    liveness: Optional[LivenessAnalysis] = None
    deepfake: Optional[DeepfakeAnalysis] = None
    lip_sync: Optional[LipSyncAnalysis] = None
    transcription: Optional[TranscriptionAnalysis] = None

    flags: List[str] = Field(default_factory=list, description="Warning flags")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    video_metadata: Optional[VideoMetadata] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc12345",
                "status": "completed",
                "processing_time_seconds": 12.5,
                "trust_score": 92,
                "verdict": "AUTHENTIC",
                "liveness": {
                    "is_live": True,
                    "confidence": 0.95,
                    "blink_detected": True,
                    "blink_count": 5,
                    "blink_pattern": "natural",
                    "head_movement_detected": True,
                    "movement_magnitude": 0.08,
                    "texture_score": 0.85,
                    "checks": {}
                },
                "deepfake": {
                    "is_authentic": True,
                    "confidence": 0.91,
                    "fake_probability": 0.09,
                    "temporal_consistency": 0.95,
                    "artifact_score": 0.05,
                    "checks": {}
                },
                "lip_sync": {
                    "is_synchronized": True,
                    "confidence": 0.88,
                    "sync_score": 0.92,
                    "correlation": 0.85,
                    "timing_offset_ms": 45,
                    "dubbing_detected": False,
                    "checks": {}
                },
                "transcription": {
                    "full_text": "Hello, my name is John and I am verifying my identity.",
                    "language": "en",
                    "language_confidence": 0.98,
                    "duration": 5.2,
                    "word_count": 11,
                    "segments": []
                },
                "flags": [],
                "recommendations": ["Video appears authentic and unmanipulated"],
                "video_metadata": {
                    "duration": 10.5,
                    "fps": 30.0,
                    "resolution": "1920x1080",
                    "frame_count": 315,
                    "has_audio": True
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    models_loaded: Dict[str, bool] = Field(description="Model loading status")


class ModelsResponse(BaseModel):
    """Information about loaded models"""
    whisper: Dict[str, str] = Field(description="Whisper model info")
    deepfake_detector: Dict[str, str] = Field(description="Deepfake detector info")
    face_mesh: Dict[str, str] = Field(description="Face mesh info")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error info")
