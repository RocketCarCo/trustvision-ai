"""
API Tests for TrustVision AI

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self, client):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data

    def test_ping(self, client):
        """Test ping endpoint"""
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModelsEndpoint:
    """Tests for models endpoint"""

    def test_list_models(self, client):
        """Test models endpoint returns model info"""
        response = client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert "whisper" in data
        assert "deepfake_detector" in data
        assert "face_mesh" in data


class TestVideoAnalysis:
    """Tests for video analysis endpoint"""

    def test_invalid_file_type(self, client):
        """Test rejection of invalid file types"""
        # Create a fake text file
        response = client.post(
            "/analyze/video",
            files={"file": ("test.txt", b"not a video", "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_no_file(self, client):
        """Test error when no file provided"""
        response = client.post("/analyze/video")
        assert response.status_code == 422  # Validation error


class TestImageAnalysis:
    """Tests for image analysis endpoint"""

    def test_invalid_image_type(self, client):
        """Test rejection of invalid image types"""
        response = client.post(
            "/analyze/image",
            files={"file": ("test.gif", b"not valid", "image/gif")}
        )
        assert response.status_code == 400


class TestDocs:
    """Test documentation endpoints"""

    def test_docs_available(self, client):
        """Test OpenAPI docs are available"""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """Test ReDoc is available"""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()
