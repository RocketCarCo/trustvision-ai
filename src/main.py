"""
TrustVision AI - Main FastAPI Application

Video Authenticity & Liveness Detection API

This API provides comprehensive video analysis for:
- Liveness Detection (is a real person present?)
- Deepfake Detection (is the video AI-generated/manipulated?)
- Lip-Sync Verification (does audio match lip movements?)
- Voice Transcription (what is being said?)

Run with: uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.config import settings
from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler

    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Whisper model: {settings.whisper_model}")
    logger.info(f"Deepfake model: {settings.deepfake_model}")

    # Ensure directories exist
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.samples_dir.mkdir(parents=True, exist_ok=True)

    logger.info("TrustVision AI started successfully")

    yield

    # Shutdown
    logger.info("Shutting down TrustVision AI...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## TrustVision AI - Video Authenticity & Liveness Detection

A comprehensive API for analyzing video authenticity using multiple ML models.

### Features

- **Liveness Detection**: Verify a real person is present using blink detection,
  head movement analysis, and facial texture analysis.

- **Deepfake Detection**: Detect AI-generated or manipulated videos using a
  Vision Transformer model trained on deepfake datasets.

- **Lip-Sync Verification**: Ensure audio matches lip movements to detect
  dubbed or manipulated audio tracks.

- **Voice Transcription**: Convert speech to text using OpenAI Whisper for
  content analysis.

### Models Used

| Component | Model | Source |
|-----------|-------|--------|
| Speech-to-Text | Whisper Base | OpenAI |
| Deepfake Detection | dima806/deepfake_vs_real_image_detection | HuggingFace |
| Face Analysis | MediaPipe Face Mesh | Google |

### Use Cases

- **KYC/Identity Verification**: Verify person is live during identity checks
- **Content Moderation**: Detect manipulated media content
- **Media Authentication**: Verify authenticity of video evidence
- **Interview Verification**: Ensure candidates are who they claim to be

""",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Analysis",
            "description": "Video and image analysis endpoints"
        },
        {
            "name": "System",
            "description": "Health checks and system information"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, tags=["Analysis"])


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")


# Health check at root level too
@app.get("/ping", tags=["System"])
async def ping():
    """Simple ping endpoint for load balancers"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
