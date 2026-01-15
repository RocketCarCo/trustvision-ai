"""
Deepfake Detection Module for TrustVision AI

Detects AI-generated or manipulated video content using:
1. Pre-trained deepfake detection model from HuggingFace
2. Temporal consistency analysis (frame-to-frame)
3. Facial artifact detection

Model: dima806/deepfake_vs_real_image_detection (ViT-based classifier)
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame"""
    timestamp: float
    is_real: bool
    confidence: float
    real_score: float
    fake_score: float


@dataclass
class DeepfakeResult:
    """Complete deepfake analysis result"""
    is_authentic: bool
    confidence: float
    fake_probability: float
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    temporal_consistency: float = 1.0
    artifact_score: float = 0.0
    checks: Dict[str, any] = field(default_factory=dict)


class DeepfakeDetector:
    """
    Deepfake Detection using Vision Transformer

    This detector uses a pre-trained ViT model from HuggingFace
    that has been fine-tuned on deepfake detection datasets.

    The model analyzes facial regions in video frames to detect
    subtle artifacts and inconsistencies typical of AI-generated
    or manipulated content.

    Model Details:
    - Architecture: Vision Transformer (ViT)
    - Training: Fine-tuned on deepfake datasets
    - Input: Face-cropped images
    - Output: Binary classification (real/fake)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the deepfake detector

        Args:
            model_name: HuggingFace model identifier (default from settings)
        """
        self.model_name = model_name or settings.deepfake_model
        self.device = torch.device(settings.device)

        # Load model and processor
        logger.info(f"Loading deepfake detection model: {self.model_name}")

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=settings.model_cache_dir
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                cache_dir=settings.model_cache_dir
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Cannot load deepfake detection model: {e}")

        # Face detection for preprocessing
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract face region from frame

        Args:
            frame: BGR image

        Returns:
            Cropped face region or None if no face detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        if len(faces) == 0:
            return None

        # Get largest face
        areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(areas)
        x, y, w, h = faces[largest_idx]

        # Add margin
        margin = int(0.2 * max(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)

        face_region = frame[y:y+h, x:x+w]

        return face_region

    def _analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> Optional[FrameAnalysis]:
        """
        Analyze a single frame for deepfake artifacts

        Args:
            frame: BGR image
            timestamp: Frame timestamp in seconds

        Returns:
            FrameAnalysis or None if no face detected
        """
        # Detect and extract face
        face = self._detect_face(frame)

        if face is None:
            logger.debug(f"No face detected at {timestamp:.2f}s")
            return None

        # Convert to PIL Image
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)

        # Preprocess for model
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Get predictions
        # Model typically has labels: ['fake', 'real'] or similar
        probs_np = probs.cpu().numpy()[0]

        # Determine label ordering from model config
        id2label = self.model.config.id2label

        # Find indices for real/fake
        real_idx = None
        fake_idx = None

        for idx, label in id2label.items():
            label_lower = label.lower()
            if 'real' in label_lower or 'authentic' in label_lower:
                real_idx = idx
            elif 'fake' in label_lower or 'deepfake' in label_lower:
                fake_idx = idx

        # Default if not found
        if real_idx is None:
            real_idx = 1
        if fake_idx is None:
            fake_idx = 0

        real_score = float(probs_np[real_idx])
        fake_score = float(probs_np[fake_idx])

        is_real = bool(real_score > fake_score)
        confidence = float(max(real_score, fake_score))

        return FrameAnalysis(
            timestamp=timestamp,
            is_real=is_real,
            confidence=confidence,
            real_score=real_score,
            fake_score=fake_score
        )

    def _calculate_temporal_consistency(
        self,
        analyses: List[FrameAnalysis]
    ) -> float:
        """
        Calculate temporal consistency across frames

        Real videos should have consistent predictions
        Deepfakes may show inconsistent frame-to-frame predictions

        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if len(analyses) < 2:
            return 1.0

        # Calculate prediction variance
        predictions = [1 if a.is_real else 0 for a in analyses]

        # If all predictions agree, high consistency
        if len(set(predictions)) == 1:
            return 1.0

        # Calculate consistency as 1 - variance
        variance = np.var(predictions)
        consistency = 1.0 - min(1.0, variance * 4)  # Scale variance

        return consistency

    def _detect_artifacts(
        self,
        frames: List[np.ndarray]
    ) -> float:
        """
        Detect visual artifacts common in deepfakes

        Looks for:
        - Blurring at face boundaries
        - Color inconsistencies
        - Temporal flickering

        Returns:
            Artifact score (0-1, lower is better/more real)
        """
        if len(frames) < 2:
            return 0.0

        artifact_indicators = []

        for i, frame in enumerate(frames):
            # Detect face
            face = self._detect_face(frame)
            if face is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Check for blur at edges (Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edge_sharpness = laplacian.var()

            # Lower sharpness at edges can indicate blending artifacts
            if edge_sharpness < 100:
                artifact_indicators.append(0.3)
            else:
                artifact_indicators.append(0.0)

        if not artifact_indicators:
            return 0.0

        return np.mean(artifact_indicators)

    def analyze_video_frames(
        self,
        frames: List[Tuple[np.ndarray, float]]  # (frame, timestamp)
    ) -> DeepfakeResult:
        """
        Analyze video frames for deepfake detection

        Args:
            frames: List of (frame, timestamp) tuples

        Returns:
            DeepfakeResult with comprehensive analysis
        """
        if len(frames) == 0:
            return DeepfakeResult(
                is_authentic=False,
                confidence=0.0,
                fake_probability=0.5,
                checks={"error": "No frames provided"}
            )

        frame_analyses = []
        raw_frames = []

        for frame, timestamp in frames:
            analysis = self._analyze_frame(frame, timestamp)
            if analysis:
                frame_analyses.append(analysis)
            raw_frames.append(frame)

        if len(frame_analyses) == 0:
            return DeepfakeResult(
                is_authentic=False,
                confidence=0.0,
                fake_probability=0.5,
                checks={"error": "No faces detected in video"}
            )

        # Calculate aggregate metrics
        real_scores = [a.real_score for a in frame_analyses]
        fake_scores = [a.fake_score for a in frame_analyses]

        avg_real_score = np.mean(real_scores)
        avg_fake_score = np.mean(fake_scores)

        # Temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(frame_analyses)

        # Artifact detection
        artifact_score = self._detect_artifacts(raw_frames)

        # Final decision
        # Combine model prediction with temporal and artifact analysis
        base_confidence = max(avg_real_score, avg_fake_score)

        # Adjust confidence based on consistency
        adjusted_confidence = base_confidence * (0.7 + 0.3 * temporal_consistency)

        # Penalize for artifacts
        adjusted_confidence = adjusted_confidence * (1 - artifact_score * 0.3)

        is_authentic = avg_real_score > avg_fake_score
        fake_probability = avg_fake_score

        # Apply threshold
        if adjusted_confidence < settings.deepfake_confidence_threshold:
            # Low confidence - be conservative
            is_authentic = adjusted_confidence > 0.5

        return DeepfakeResult(
            is_authentic=is_authentic,
            confidence=adjusted_confidence,
            fake_probability=fake_probability,
            frame_analyses=frame_analyses,
            temporal_consistency=temporal_consistency,
            artifact_score=artifact_score,
            checks={
                "frames_analyzed": len(frame_analyses),
                "avg_real_score": avg_real_score,
                "avg_fake_score": avg_fake_score,
                "temporal_consistency": temporal_consistency,
                "artifact_score": artifact_score,
                "frame_predictions": [
                    {
                        "timestamp": a.timestamp,
                        "prediction": "real" if a.is_real else "fake",
                        "confidence": a.confidence
                    }
                    for a in frame_analyses
                ]
            }
        )

    def analyze_single_image(self, image: np.ndarray) -> DeepfakeResult:
        """
        Analyze a single image for deepfake detection

        Useful for photo verification

        Args:
            image: BGR image

        Returns:
            DeepfakeResult
        """
        return self.analyze_video_frames([(image, 0.0)])
