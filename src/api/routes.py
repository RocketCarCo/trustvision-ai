"""
API Routes for TrustVision AI

Endpoints:
- GET  /health          - Health check
- POST /analyze/video   - Analyze video file for authenticity
- GET  /models          - List loaded models
"""

import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.config import settings, get_settings, Settings
from src.services.trust_analyzer import TrustAnalyzer
from src.utils.logger import get_logger
from src.api.schemas import (
    AnalysisResponse,
    AnalysisOptions,
    HealthResponse,
    ModelsResponse,
    ErrorResponse
)

logger = get_logger(__name__)

router = APIRouter()

# Global analyzer instance (initialized on startup)
_analyzer: Optional[TrustAnalyzer] = None


def get_analyzer() -> TrustAnalyzer:
    """Get or create the global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        logger.info("Initializing TrustAnalyzer...")
        _analyzer = TrustAnalyzer()
    return _analyzer


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API health and model status"
)
async def health_check(config: Settings = Depends(get_settings)):
    """
    Health check endpoint

    Returns service status and loaded model information.
    """
    return HealthResponse(
        status="healthy",
        version=config.app_version,
        models_loaded={
            "whisper": True,  # Lazy loaded
            "deepfake_detector": True,
            "face_mesh": True
        }
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List Models",
    description="Get information about loaded ML models"
)
async def list_models(config: Settings = Depends(get_settings)):
    """
    List loaded models and their configurations
    """
    return ModelsResponse(
        whisper={
            "name": f"openai/whisper-{config.whisper_model}",
            "source": "OpenAI",
            "purpose": "Speech-to-text transcription",
            "size": config.whisper_model
        },
        deepfake_detector={
            "name": config.deepfake_model,
            "source": "HuggingFace",
            "purpose": "Deepfake/manipulation detection",
            "architecture": "Vision Transformer (ViT)"
        },
        face_mesh={
            "name": "MediaPipe Face Mesh",
            "source": "Google MediaPipe",
            "purpose": "Facial landmark detection for liveness & lip-sync",
            "landmarks": "468 facial landmarks"
        }
    )


@router.post(
    "/analyze/video",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Analyze Video",
    description="Upload a video file for comprehensive trust analysis"
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze (MP4, AVI, MOV, MKV, WebM)"),
    options: Optional[str] = Form(
        default=None,
        description="JSON string with analysis options"
    ),
    analyzer: TrustAnalyzer = Depends(get_analyzer)
):
    """
    Analyze a video file for authenticity and liveness.

    This endpoint performs comprehensive analysis including:
    - **Liveness Detection**: Verifies a real person is present (not photo/video playback)
    - **Deepfake Detection**: Checks for AI-generated or manipulated content
    - **Lip-Sync Verification**: Ensures audio matches lip movements
    - **Voice Transcription**: Converts speech to text using Whisper

    **Supported formats**: MP4, AVI, MOV, MKV, WebM

    **Returns**: A trust report with scores and detailed analysis
    """
    import json

    # Parse options
    analysis_options = {}
    if options:
        try:
            analysis_options = json.loads(options)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid options JSON format"
            )

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Supported: {', '.join(valid_extensions)}"
        )

    # Save uploaded file to temp location
    temp_dir = Path(tempfile.mkdtemp(dir=settings.upload_dir))
    temp_file = temp_dir / f"upload{file_ext}"

    try:
        # Write uploaded file
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing uploaded file: {file.filename} ({temp_file.stat().st_size / 1024:.1f} KB)")

        # Run analysis
        report = analyzer.analyze_video(temp_file, analysis_options)

        # Convert to response model
        return JSONResponse(
            status_code=200,
            content=report.to_dict()
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


@router.post(
    "/analyze/image",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Analyze Image",
    description="Analyze a single image for deepfake detection"
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze (JPG, PNG)"),
    analyzer: TrustAnalyzer = Depends(get_analyzer)
):
    """
    Analyze a single image for deepfake detection.

    This is a simplified endpoint for image-only analysis.
    Only deepfake detection is performed (no liveness or lip-sync).

    **Supported formats**: JPG, JPEG, PNG

    **Returns**: A trust report focused on deepfake detection
    """
    import cv2
    import numpy as np

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Supported: {', '.join(valid_extensions)}"
        )

    try:
        # Read image data
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        logger.info(f"Processing uploaded image: {file.filename}")

        # Run deepfake detection only
        result = analyzer.deepfake_detector.analyze_single_image(image)

        # Build simplified report - convert numpy types to Python types
        is_authentic = bool(result.is_authentic)
        confidence = float(result.confidence)
        fake_prob = float(result.fake_probability)

        trust_score = int(confidence * 100) if is_authentic else int((1 - fake_prob) * 50)

        return JSONResponse(
            status_code=200,
            content={
                "job_id": "img-" + str(hash(file.filename))[:8],
                "status": "completed",
                "processing_time_seconds": 0.0,
                "trust_score": trust_score,
                "verdict": "AUTHENTIC" if is_authentic else "SUSPICIOUS",
                "deepfake": {
                    "is_authentic": is_authentic,
                    "confidence": round(confidence, 3),
                    "fake_probability": round(fake_prob, 3)
                },
                "flags": [] if is_authentic else ["Potential manipulation detected"],
                "recommendations": ["Image appears authentic"] if is_authentic else ["Image may be manipulated"]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
