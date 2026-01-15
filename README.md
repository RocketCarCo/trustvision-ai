# TrustVision AI

> **Video Authenticity & Liveness Detection Platform**

A comprehensive ML-powered platform for detecting deepfakes, verifying liveness, and analyzing video authenticity using state-of-the-art models from HuggingFace and OpenAI.

**Two Deployment Options:**
- **Real-time Web App** (Port 5001) - Live webcam analysis with interactive UI
- **REST API** (Port 8000) - File upload-based analysis for integration

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Real-time Web Application](#real-time-web-application)
- [Architecture](#architecture)
- [Models Used](#models-used)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)

---

## Overview

TrustVision AI provides comprehensive video analysis for authenticity verification. It combines multiple ML techniques to detect:

- **Fake/AI-generated videos** (deepfakes)
- **Photo/video replay attacks** (liveness)
- **Audio-video manipulation** (lip-sync)

This solution demonstrates the deployment of multiple HuggingFace models as a production-ready API endpoint using Docker.

### Use Cases

| Industry | Application |
|----------|-------------|
| **FinTech** | KYC/Identity verification |
| **HR Tech** | Remote interview verification |
| **Media** | Content authenticity validation |
| **Legal** | Evidence verification |
| **Social** | Deepfake detection |

---

## Features

### 1. Liveness Detection
Verifies a real person is present (not a photo or video playback):
- **Blink Detection**: Analyzes eye aspect ratio (EAR) for natural blink patterns
- **Head Movement**: Tracks micro-movements indicating live presence
- **Texture Analysis**: Differentiates real skin from printed/screen display

### 2. Deepfake Detection
Detects AI-generated or manipulated videos using:
- **Vision Transformer (ViT)**: Pre-trained model from HuggingFace
- **Temporal Consistency**: Frame-to-frame coherence analysis
- **Artifact Detection**: Identifies GAN fingerprints and blending artifacts

### 3. Lip-Sync Verification
Ensures audio matches lip movements:
- **Lip Landmark Tracking**: MediaPipe Face Mesh with 468 landmarks
- **Audio Energy Correlation**: Compares speech patterns with lip movement
- **Timing Analysis**: Detects audio-video desync indicative of dubbing

### 4. Voice Transcription
Converts speech to text using OpenAI Whisper:
- **Multilingual Support**: 99 languages
- **Timestamp Alignment**: Word/segment level timestamps
- **Noise Robust**: Handles various audio qualities

### 5. Safety Hazard Detection (NEW)
YOLO-based object detection for senior living environments:
- **Real-time Detection**: Identifies potential hazards in living spaces
- **Mobile Friendly**: Works on phones with front/back camera switching
- **Alert System**: Visual alerts for detected hazards

---

## Real-time Web Application

The platform includes a **real-time web application** with live webcam support, designed for interactive demonstrations and mobile use.

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Web App** | http://localhost:5001 | Real-time webcam analysis |
| **REST API** | http://localhost:8000/docs | File upload API with Swagger UI |

### Two Analysis Modes

#### Mode 1: Live Interview Analysis
Perfect for remote interview verification and KYC processes.

**Features:**
- Real-time liveness detection with blink counting
- Eye Aspect Ratio (EAR) monitoring
- Deepfake detection on live video feed
- Head rotation tracking
- Visual status badges (LIVE / CHECKING / FAKE)

**Use Cases:**
- Remote job interviews
- Video KYC for banking
- Identity verification

#### Mode 2: Safety Hazard Scanner
YOLO-powered object detection for senior living safety assessments.

**Features:**
- Real-time hazard detection using YOLOv8
- Mobile-friendly with camera flip (front/back)
- Detection box overlays on video
- Room safety status indicator
- Alert panel with detected objects

**Detectable Objects:**
- Electrical hazards (exposed wires, overloaded outlets)
- Trip hazards (loose rugs, clutter)
- Fire hazards (candles, space heaters)
- General safety concerns

### Screenshots

```
┌─────────────────────────────────────────────┐
│         TrustVision AI                      │
│                                             │
│  ┌─────────────┐    ┌─────────────┐        │
│  │  INTERVIEW  │    │   SAFETY    │        │
│  │  ANALYSIS   │    │   SCANNER   │        │
│  │             │    │             │        │
│  │  Liveness + │    │  YOLO-based │        │
│  │  Deepfake   │    │  Hazard Det │        │
│  └─────────────┘    └─────────────┘        │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRUSTVISION AI - ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   VIDEO INPUT                                                               │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     PROCESSING PIPELINE                              │   │
│   │                                                                      │   │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│   │   │   LIVENESS   │    │   DEEPFAKE   │    │   LIP-SYNC   │         │   │
│   │   │  DETECTION   │    │  DETECTION   │    │ VERIFICATION │         │   │
│   │   ├──────────────┤    ├──────────────┤    ├──────────────┤         │   │
│   │   │• Blink Det.  │    │• ViT Model   │    │• Face Mesh   │         │   │
│   │   │• Head Move   │    │• Artifacts   │    │• Audio Corr  │         │   │
│   │   │• Texture     │    │• Temporal    │    │• Sync Score  │         │   │
│   │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
│   │          │                   │                   │                  │   │
│   │          └───────────────────┼───────────────────┘                  │   │
│   │                              ▼                                      │   │
│   │                    ┌──────────────────┐                             │   │
│   │                    │  TRUST ANALYZER  │                             │   │
│   │                    │  Score: 0-100    │                             │   │
│   │                    └──────────────────┘                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         TRUST REPORT                                 │   │
│   │                                                                      │   │
│   │  Trust Score: 92/100  |  Verdict: AUTHENTIC                         │   │
│   │  ✓ Liveness: PASSED   |  ✓ Deepfake: PASSED  |  ✓ Lip-Sync: PASSED │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Models Used

| Component | Model | Source | Purpose |
|-----------|-------|--------|---------|
| **Speech-to-Text** | `openai/whisper-base` | OpenAI | Audio transcription |
| **Deepfake Detection** | `dima806/deepfake_vs_real_image_detection` | HuggingFace | Detect AI-generated faces |
| **Face Analysis** | MediaPipe Face Mesh | Google | 468 facial landmarks for liveness & lip-sync |
| **Object Detection** | YOLOv8n | Ultralytics | Real-time hazard detection |

### Why These Models?

1. **Whisper** - State-of-the-art ASR with excellent noise robustness and multilingual support
2. **ViT Deepfake Detector** - Vision Transformer architecture achieves high accuracy on deepfake detection
3. **MediaPipe** - Fast, accurate facial landmark detection optimized for real-time applications
4. **YOLOv8** - Best-in-class object detection with real-time performance on CPU

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB RAM (minimum)
- 10GB disk space (for models)

### 1. Clone & Setup

```bash
# Clone the repository
git clone <repository-url>
cd deloitte-MLModel

# Copy environment file
cp .env.example .env
```

### 2. Build & Run with Docker

```bash
# Build and start all services
docker-compose up --build

# Or run in background (detached mode)
docker-compose up -d --build

# Check status
docker-compose ps
```

### 3. Access the Platform

| Service | URL | Description |
|---------|-----|-------------|
| **Web Application** | http://localhost:5001 | Real-time webcam analysis |
| **API Documentation** | http://localhost:8000/docs | Swagger UI for REST API |
| **API Health Check** | http://localhost:8000/health | API status |
| **Web App Health** | http://localhost:5001/health | Web app status |

### 4. Using the Web App

1. Open http://localhost:5001 in your browser
2. Choose a mode:
   - **Live Interview** - For liveness + deepfake detection
   - **Safety Scanner** - For hazard detection
3. Click **Start Analysis** and allow camera access
4. View real-time results on screen

---

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check & model status |
| `/models` | GET | List loaded models |
| `/analyze/video` | POST | Analyze video for authenticity |
| `/analyze/image` | POST | Analyze image for deepfake |

### Response Format

```json
{
  "job_id": "abc12345",
  "status": "completed",
  "processing_time_seconds": 12.5,
  "trust_score": 92,
  "verdict": "AUTHENTIC",

  "liveness": {
    "is_live": true,
    "confidence": 0.95,
    "blink_detected": true,
    "blink_count": 5,
    "blink_pattern": "natural",
    "head_movement_detected": true
  },

  "deepfake": {
    "is_authentic": true,
    "confidence": 0.91,
    "fake_probability": 0.09,
    "temporal_consistency": 0.95
  },

  "lip_sync": {
    "is_synchronized": true,
    "sync_score": 0.92,
    "correlation": 0.85,
    "dubbing_detected": false
  },

  "transcription": {
    "full_text": "Hello, my name is John...",
    "language": "en",
    "duration": 5.2,
    "word_count": 11
  },

  "flags": [],
  "recommendations": ["Video appears authentic"]
}
```

---

## Usage Examples

### Using cURL

```bash
# Analyze a video file
curl -X POST "http://localhost:8000/analyze/video" \
  -F "file=@sample_video.mp4" \
  -F 'options={"run_liveness": true, "run_deepfake": true}'

# Analyze an image
curl -X POST "http://localhost:8000/analyze/image" \
  -F "file=@face_photo.jpg"

# Health check
curl http://localhost:8000/health
```

### Using Python

```python
import httpx

# Analyze video
with open("sample_video.mp4", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/analyze/video",
        files={"file": f},
        data={"options": '{"run_liveness": true}'}
    )

result = response.json()
print(f"Trust Score: {result['trust_score']}")
print(f"Verdict: {result['verdict']}")
```

### Using JavaScript

```javascript
const formData = new FormData();
formData.append('file', videoFile);
formData.append('options', JSON.stringify({
  run_liveness: true,
  run_deepfake: true,
  run_lip_sync: true,
  run_transcription: true
}));

const response = await fetch('http://localhost:8000/analyze/video', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Trust Score: ${result.trust_score}`);
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cpu` | Processing device (cpu, cuda, mps) |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_VIDEO_DURATION` | `300` | Maximum video length (seconds) |
| `MAX_FILE_SIZE_MB` | `100` | Maximum upload size |

### Model Configuration

```bash
# For faster processing (less accurate)
WHISPER_MODEL=tiny

# For better accuracy (slower)
WHISPER_MODEL=small

# For GPU acceleration (requires CUDA)
DEVICE=cuda
```

---

## Project Structure

```
trustvision-ai/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Pydantic models
│   ├── models/
│   │   ├── liveness_detector.py    # Blink & movement detection
│   │   ├── deepfake_detector.py    # ViT-based deepfake detection
│   │   ├── lip_sync_analyzer.py    # Audio-visual correlation
│   │   └── voice_transcriber.py    # Whisper transcription
│   ├── services/
│   │   └── trust_analyzer.py   # Main orchestration
│   ├── utils/
│   │   ├── video_utils.py      # Video processing
│   │   └── logger.py           # Structured logging
│   └── webapp/                 # Real-time Web Application
│       ├── app.py              # Flask-SocketIO server
│       └── templates/
│           ├── index.html      # Landing page
│           ├── interview.html  # Interview analysis mode
│           └── safety.html     # Safety scanner mode
├── models/
│   └── yolov8n.pt              # YOLO model for hazard detection
├── Dockerfile                  # API container
├── Dockerfile.webapp           # Web app container
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## How It Works

### Liveness Detection Algorithm

```
1. Extract frames at 1 FPS
2. For each frame:
   a. Detect face using MediaPipe
   b. Calculate Eye Aspect Ratio (EAR)
   c. Track EAR over time for blink detection
   d. Estimate head pose (pitch, yaw, roll)
   e. Analyze facial texture variance
3. Aggregate results:
   - Blink count and pattern
   - Head movement magnitude
   - Texture score
4. Calculate confidence score
```

### Deepfake Detection Algorithm

```
1. Extract frames at 1 FPS
2. For each frame:
   a. Detect and crop face region
   b. Preprocess for ViT model
   c. Run inference (real vs fake classification)
   d. Store prediction and confidence
3. Post-processing:
   a. Calculate temporal consistency
   b. Detect visual artifacts
   c. Aggregate frame predictions
4. Final verdict with confidence
```

### Lip-Sync Verification Algorithm

```
1. Extract video frames and audio track
2. Video analysis:
   a. Track lip landmarks (MediaPipe)
   b. Calculate mouth aspect ratio over time
3. Audio analysis:
   a. Extract RMS energy over time
   b. Detect speech activity
4. Correlation:
   a. Align lip movement with audio energy
   b. Calculate cross-correlation
   c. Find optimal timing offset
5. Determine sync quality
```

---

## Performance Considerations

| Component | Processing Time | Memory |
|-----------|----------------|--------|
| Liveness Detection | ~2s per 10s video | ~500MB |
| Deepfake Detection | ~3s per 10s video | ~2GB |
| Lip-Sync Analysis | ~1s per 10s video | ~500MB |
| Transcription | ~5s per 10s audio | ~1GB |
| **Total** | ~12s per 10s video | ~4GB |

### Optimization Tips

1. Use `WHISPER_MODEL=tiny` for faster transcription
2. Disable unused features via options
3. Use GPU (`DEVICE=cuda`) for 5-10x speedup
4. Reduce `FRAME_SAMPLE_RATE` for faster processing

---

## Limitations

- Maximum video duration: 5 minutes
- Requires face to be visible in video
- Audio must be present for lip-sync/transcription
- CPU processing can be slow (GPU recommended for production)

---

## License

This project is for demonstration purposes.

---

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [HuggingFace Transformers](https://huggingface.co/) - Model hosting
- [MediaPipe](https://mediapipe.dev/) - Face detection
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
