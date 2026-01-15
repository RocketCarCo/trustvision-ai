"""
ML Models for TrustVision AI

This module contains all ML model implementations:
- LivenessDetector: Detect if person is live (not photo/video playback)
- DeepfakeDetector: Detect AI-generated/manipulated videos
- LipSyncAnalyzer: Verify audio-visual synchronization
- VoiceTranscriber: Speech-to-text using Whisper
"""

from .liveness_detector import LivenessDetector
from .deepfake_detector import DeepfakeDetector
from .lip_sync_analyzer import LipSyncAnalyzer
from .voice_transcriber import VoiceTranscriber

__all__ = [
    "LivenessDetector",
    "DeepfakeDetector",
    "LipSyncAnalyzer",
    "VoiceTranscriber",
]
