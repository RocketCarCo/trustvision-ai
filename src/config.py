"""
TrustVision AI - Configuration Module
Centralized configuration management using pydantic-settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "TrustVision AI"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    model_cache_dir: Path = Field(default=Path("./models"))
    upload_dir: Path = Field(default=Path("./uploads"))
    samples_dir: Path = Field(default=Path("./samples"))

    # Device Configuration
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # Model Settings
    whisper_model: str = "base"  # tiny, base, small, medium, large
    deepfake_model: str = "dima806/deepfake_vs_real_image_detection"

    # Processing Settings
    frame_sample_rate: int = 2  # Sample every N seconds
    max_video_duration: int = 300  # Max 5 minutes
    max_file_size_mb: int = 100

    # Liveness Detection Thresholds
    min_blink_count: int = 2
    min_head_movement_threshold: float = 0.02

    # Confidence Thresholds
    liveness_confidence_threshold: float = 0.7
    deepfake_confidence_threshold: float = 0.7
    lip_sync_threshold: float = 0.6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection helper for FastAPI"""
    return settings
