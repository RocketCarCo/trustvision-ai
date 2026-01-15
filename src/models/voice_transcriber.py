"""
Voice Transcription Module for TrustVision AI

Uses OpenAI Whisper for speech-to-text transcription.

Whisper is a state-of-the-art automatic speech recognition (ASR) model
trained on 680,000 hours of multilingual and multitask supervised data.

Model Sizes:
- tiny: 39M params, ~32x faster than large
- base: 74M params, ~16x faster than large (DEFAULT)
- small: 244M params, ~6x faster than large
- medium: 769M params, ~2x faster than large
- large: 1550M params, highest accuracy

For demo purposes, we use 'base' which provides good accuracy with
reasonable speed. Can be configured via settings.
"""

import whisper
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech"""
    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Complete transcription result"""
    full_text: str
    language: str
    language_confidence: float
    segments: List[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0
    word_count: int = 0


class VoiceTranscriber:
    """
    Voice Transcription using OpenAI Whisper

    Whisper is an encoder-decoder Transformer model trained for
    speech recognition and translation.

    Key Features:
    - Multilingual (99 languages)
    - Robust to noise, accents, technical language
    - Punctuation and formatting included
    - Timestamp-level alignment

    Model Architecture:
    - Encoder: Audio -> Feature representations
    - Decoder: Features -> Text tokens (autoregressive)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Whisper transcriber

        Args:
            model_name: Whisper model size (tiny/base/small/medium/large)
        """
        self.model_name = model_name or settings.whisper_model
        self.model = None

        logger.info(f"Initializing Whisper model: {self.model_name}")

    def _load_model(self):
        """Lazy load Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper {self.model_name} model...")
            self.model = whisper.load_model(
                self.model_name,
                download_root=str(settings.model_cache_dir)
            )

            # Move to configured device
            device = settings.device
            if device == "cuda":
                self.model = self.model.cuda()
            elif device == "mps":
                # MPS support for Apple Silicon
                import torch
                if torch.backends.mps.is_available():
                    self.model = self.model.to("mps")

            logger.info(f"Whisper model loaded on {device}")

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Optional language hint (e.g., 'en', 'es')

        Returns:
            TranscriptionResult with full transcription and segments
        """
        self._load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio: {audio_path}")

        # Transcription options
        options = {
            "fp16": False,  # Use FP32 for CPU compatibility
            "verbose": False,
        }

        if language:
            options["language"] = language

        # Run transcription
        result = self.model.transcribe(str(audio_path), **options)

        # Parse results
        full_text = result["text"].strip()
        detected_language = result.get("language", "en")

        # Calculate language confidence from log probabilities if available
        language_confidence = 0.9  # Default high confidence

        # Parse segments
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("avg_logprob", 0.0)
            ))

        # Calculate duration
        duration = segments[-1].end if segments else 0.0

        # Word count
        word_count = len(full_text.split())

        logger.info(
            f"Transcription complete: {word_count} words, "
            f"{len(segments)} segments, {duration:.1f}s duration"
        )

        return TranscriptionResult(
            full_text=full_text,
            language=detected_language,
            language_confidence=language_confidence,
            segments=segments,
            duration=duration,
            word_count=word_count
        )

    def transcribe_with_timestamps(
        self,
        audio_path: Path,
        word_level: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe with detailed timestamps

        Args:
            audio_path: Path to audio file
            word_level: If True, attempt word-level timestamps

        Returns:
            TranscriptionResult with detailed segment timestamps
        """
        self._load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing with timestamps: {audio_path}")

        # Use word timestamps if requested
        result = self.model.transcribe(
            str(audio_path),
            fp16=False,
            verbose=False,
            word_timestamps=word_level
        )

        full_text = result["text"].strip()
        detected_language = result.get("language", "en")

        # Parse segments with more detail
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                confidence=np.exp(seg.get("avg_logprob", -1.0))  # Convert log prob
            ))

        duration = segments[-1].end if segments else 0.0
        word_count = len(full_text.split())

        return TranscriptionResult(
            full_text=full_text,
            language=detected_language,
            language_confidence=0.9,
            segments=segments,
            duration=duration,
            word_count=word_count
        )

    def detect_language(self, audio_path: Path) -> tuple:
        """
        Detect language of audio without full transcription

        Args:
            audio_path: Path to audio file

        Returns:
            (language_code, confidence) tuple
        """
        self._load_model()

        # Load audio
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]

        logger.info(f"Detected language: {detected_lang} ({confidence:.2%})")

        return detected_lang, confidence
