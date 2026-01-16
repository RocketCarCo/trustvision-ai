"""
API Routes for TrustVision AI

Endpoints:
- GET  /health              - Health check
- POST /analyze/video       - Analyze video file for authenticity
- POST /analyze/authenticity - Alias for video authenticity analysis
- POST /analyze/image       - Analyze image for deepfake detection
- POST /analyze/safety      - Analyze image/video for safety hazards
- GET  /models              - List loaded models
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
    "/analyze/authenticity",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Analyze Authenticity",
    description="Analyze video/image for authenticity and liveness detection"
)
async def analyze_authenticity(
    file: UploadFile = File(..., description="Video or image file to analyze"),
    analyzer: TrustAnalyzer = Depends(get_analyzer)
):
    """
    Analyze a video or image for authenticity and liveness.

    This endpoint performs:
    - **Liveness Detection**: Verifies a real person is present
    - **Deepfake Detection**: Checks for AI-generated content
    - **Lip-Sync Verification**: Ensures audio matches lip movements (video only)

    **Supported formats**: MP4, AVI, MOV, JPG, PNG

    **Returns**: Trust score, verdict, and detailed analysis
    """
    import cv2
    import numpy as np

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    valid_video = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    valid_image = {".jpg", ".jpeg", ".png"}

    # Handle image files
    if file_ext in valid_image:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Could not decode image")

            # Run deepfake detection
            result = analyzer.deepfake_detector.analyze_single_image(image)

            is_authentic = bool(result.is_authentic)
            confidence = float(result.confidence)

            trust_score = int(confidence * 100) if is_authentic else int((1 - result.fake_probability) * 50)

            return JSONResponse(
                status_code=200,
                content={
                    "trust_score": trust_score,
                    "verdict": "AUTHENTIC" if is_authentic else "SUSPICIOUS",
                    "liveness": {
                        "is_live": None,
                        "blinks": None,
                        "note": "Liveness requires video input"
                    },
                    "deepfake": {
                        "is_authentic": is_authentic,
                        "confidence": round(confidence, 2)
                    },
                    "lip_sync": {
                        "match": None,
                        "note": "Lip-sync requires video with audio"
                    }
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authenticity analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Handle video files
    elif file_ext in valid_video:
        temp_dir = Path(tempfile.mkdtemp(dir=settings.upload_dir))
        temp_file = temp_dir / f"upload{file_ext}"

        try:
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            report = analyzer.analyze_video(temp_file, {})

            # Format response to match documentation
            return JSONResponse(
                status_code=200,
                content={
                    "trust_score": report.trust_score,
                    "verdict": report.verdict,
                    "liveness": {
                        "is_live": report.liveness.is_live if report.liveness else None,
                        "blinks": report.liveness.blink_count if report.liveness else 0
                    },
                    "deepfake": {
                        "is_authentic": report.deepfake.is_authentic if report.deepfake else None
                    },
                    "lip_sync": {
                        "match": round(report.lip_sync.sync_score, 2) if report.lip_sync else None
                    }
                }
            )
        except Exception as e:
            logger.error(f"Authenticity analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Supported: video ({', '.join(valid_video)}) or image ({', '.join(valid_image)})"
        )


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


# Safety hazard categories mapped from YOLO detections
HAZARD_MAPPINGS = {
    # Fall risks
    "backpack": {"type": "FALL_RISK", "severity": "low", "recommendation": "Move backpack from walkway"},
    "handbag": {"type": "FALL_RISK", "severity": "low", "recommendation": "Store handbag in designated area"},
    "suitcase": {"type": "FALL_RISK", "severity": "medium", "recommendation": "Move suitcase from pathway"},
    "sports ball": {"type": "FALL_RISK", "severity": "medium", "recommendation": "Store sports equipment safely"},
    "skateboard": {"type": "FALL_RISK", "severity": "high", "recommendation": "Remove skateboard from floor"},
    "skis": {"type": "FALL_RISK", "severity": "high", "recommendation": "Store skis properly"},
    "bottle": {"type": "FALL_RISK", "severity": "low", "recommendation": "Place bottles on stable surface"},
    "cup": {"type": "FALL_RISK", "severity": "low", "recommendation": "Secure cups to prevent spills"},
    "book": {"type": "CLUTTER", "severity": "low", "recommendation": "Organize books on shelves"},
    "vase": {"type": "FALL_RISK", "severity": "medium", "recommendation": "Secure vase to prevent tipping"},
    "scissors": {"type": "SHARP_OBJECT", "severity": "high", "recommendation": "Store scissors in drawer"},
    "knife": {"type": "SHARP_OBJECT", "severity": "high", "recommendation": "Store knives safely in block or drawer"},
    "fork": {"type": "SHARP_OBJECT", "severity": "low", "recommendation": "Return utensils to drawer"},
    # Fire hazards
    "oven": {"type": "FIRE_HAZARD", "severity": "medium", "recommendation": "Ensure oven is turned off when not in use"},
    "toaster": {"type": "FIRE_HAZARD", "severity": "low", "recommendation": "Unplug toaster when not in use"},
    "microwave": {"type": "FIRE_HAZARD", "severity": "low", "recommendation": "Keep microwave area clear"},
    # Electrical
    "hair drier": {"type": "ELECTRICAL", "severity": "medium", "recommendation": "Unplug hair dryer after use"},
    "cell phone": {"type": "ELECTRICAL", "severity": "low", "recommendation": "Don't leave phone charging unattended overnight"},
    "laptop": {"type": "ELECTRICAL", "severity": "low", "recommendation": "Ensure proper ventilation for laptop"},
    "tv": {"type": "ELECTRICAL", "severity": "low", "recommendation": "Secure TV to prevent tipping"},
    # Mobility obstacles
    "chair": {"type": "MOBILITY_OBSTACLE", "severity": "medium", "recommendation": "Push chair under table when not in use"},
    "couch": {"type": "MOBILITY_OBSTACLE", "severity": "low", "recommendation": "Ensure clear pathway around furniture"},
    "bed": {"type": "MOBILITY_OBSTACLE", "severity": "low", "recommendation": "Keep area around bed clear"},
    "dining table": {"type": "MOBILITY_OBSTACLE", "severity": "low", "recommendation": "Ensure adequate space around table"},
    "potted plant": {"type": "FALL_RISK", "severity": "medium", "recommendation": "Move plant from walkway"},
    # Pets (trip hazard)
    "dog": {"type": "TRIP_HAZARD", "severity": "medium", "recommendation": "Be aware of pet location when walking"},
    "cat": {"type": "TRIP_HAZARD", "severity": "medium", "recommendation": "Be aware of pet location when walking"},
}


@router.post(
    "/analyze/safety",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    summary="Analyze Safety Hazards",
    description="Analyze an image for safety hazards in senior living environments"
)
async def analyze_safety(
    file: UploadFile = File(..., description="Image file to analyze (JPG, PNG)"),
    location: Optional[str] = Form(default="room", description="Location name (e.g., kitchen, hallway)")
):
    """
    Analyze an image for safety hazards using YOLO object detection.

    This endpoint identifies potential hazards for senior living including:
    - **Fall Risks**: Objects on floor, clutter, trip hazards
    - **Fire Hazards**: Appliances, heat sources
    - **Sharp Objects**: Knives, scissors
    - **Mobility Obstacles**: Furniture blocking pathways

    **Supported formats**: JPG, JPEG, PNG

    **Returns**: Safety score, risk level, detected hazards, and recommendations
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

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

        logger.info(f"Processing safety analysis: {file.filename}")

        # Load YOLO model
        model_path = Path("models/yolov8n.pt")
        if not model_path.exists():
            model_path = Path("/app/models/yolov8n.pt")

        model = YOLO(str(model_path))

        # Run detection
        results = model(image, verbose=False)

        hazards = []
        recommendations = []
        severity_scores = {"low": 10, "medium": 25, "high": 40}
        total_penalty = 0

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    # Check if this object is a potential hazard
                    if class_name in HAZARD_MAPPINGS and confidence > 0.4:
                        hazard_info = HAZARD_MAPPINGS[class_name]

                        hazard = {
                            "type": hazard_info["type"],
                            "location": location,
                            "item": class_name,
                            "confidence": round(confidence, 2),
                            "severity": hazard_info["severity"]
                        }
                        hazards.append(hazard)

                        if hazard_info["recommendation"] not in recommendations:
                            recommendations.append(hazard_info["recommendation"])

                        total_penalty += severity_scores.get(hazard_info["severity"], 10)

        # Calculate safety score (100 - penalties, minimum 0)
        safety_score = max(0, 100 - total_penalty)

        # Determine risk level
        if safety_score >= 80:
            risk_level = "LOW"
        elif safety_score >= 60:
            risk_level = "MODERATE"
        elif safety_score >= 40:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # Add general recommendation if no hazards found
        if not hazards:
            recommendations = ["No significant hazards detected. Room appears safe."]

        return JSONResponse(
            status_code=200,
            content={
                "safety_score": safety_score,
                "risk_level": risk_level,
                "hazards_detected": len(hazards),
                "hazards": hazards,
                "recommendations": recommendations,
                "location": location,
                "analysis": {
                    "objects_scanned": len(results[0].boxes) if results and results[0].boxes is not None else 0,
                    "hazard_categories": list(set(h["type"] for h in hazards))
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Safety analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
