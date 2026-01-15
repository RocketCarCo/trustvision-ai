# TrustVision AI - Makefile
# Common commands for development and deployment

.PHONY: help build run stop logs test clean

# Default target
help:
	@echo "TrustVision AI - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "  make build    - Build Docker image"
	@echo "  make run      - Start the API server"
	@echo "  make stop     - Stop the API server"
	@echo "  make logs     - View container logs"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Remove containers and volumes"
	@echo "  make dev      - Run locally for development"
	@echo ""

# Build Docker image
build:
	docker-compose build

# Run the API
run:
	docker-compose up -d
	@echo ""
	@echo "TrustVision AI is starting..."
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Health:   http://localhost:8000/health"
	@echo ""

# Stop the API
stop:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Run tests
test:
	pytest tests/ -v

# Clean up
clean:
	docker-compose down -v --remove-orphans
	docker volume rm trustvision_models trustvision_uploads 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Development mode (local, not Docker)
dev:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Starting development server..."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Quick test with curl
test-health:
	curl -s http://localhost:8000/health | python -m json.tool

# List models
test-models:
	curl -s http://localhost:8000/models | python -m json.tool
