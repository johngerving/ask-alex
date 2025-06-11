#!/bin/bash

# Exit on error
set -e

# Configuration
REGISTRY="gitlab-registry.nrp-nautilus.io/humboldt/ask-alex" 
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BACKEND_IMAGE="${REGISTRY}/backend"
FRONTEND_IMAGE="${REGISTRY}/frontend"
PARSE_IMAGE="${REGISTRY}/parse"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with timestamp
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build backend
log "Building backend image..."
if ! docker build -t "${BACKEND_IMAGE}:${VERSION}" -t "${BACKEND_IMAGE}:latest" ./backend/generate; then
    error "Failed to build backend image"
    exit 1
fi
log "Backend image built successfully"

# Build parse pipeline
log "Building document processing pipeline image..."
if ! docker build -t "${PARSE_IMAGE}:${VERSION}" -t "${PARSE_IMAGE}:latest" ./backend/parse; then
    error "Failed to build document processing pipeline image"
    exit 1
fi
log "Document processing pipeline image built successfully"

# Build frontend
log "Building frontend image..."
if ! docker build -t "${FRONTEND_IMAGE}:${VERSION}" -t "${FRONTEND_IMAGE}:latest" ./frontend; then
    error "Failed to build frontend image"
    exit 1
fi
log "Frontend image built successfully"

# Print build summary
log "Build completed successfully!"
log "Images created:"
log "  Backend: ${BACKEND_IMAGE}:${VERSION}"
log "  Document Processing Pipeline: ${PARSE_IMAGE}:${VERSION}"
log "  Frontend: ${FRONTEND_IMAGE}:${VERSION}"

# Optional: Push images to registry
if [ "$1" == "--push" ]; then
    log "Pushing images to registry..."
    
    # Push backend
    if ! docker push "${BACKEND_IMAGE}:${VERSION}"; then
        error "Failed to push backend image"
        exit 1
    fi
    if ! docker push "${BACKEND_IMAGE}:latest"; then
        error "Failed to push backend latest tag"
        exit 1
    fi
    
    # Push parse pipeline
    if ! docker push "${PARSE_IMAGE}:${VERSION}"; then
        error "Failed to push document processing pipeline image"
        exit 1
    fi
    if ! docker push "${PARSE_IMAGE}:latest"; then
        error "Failed to push document processing pipeline latest tag"
        exit 1
    fi
    
    # Push frontend
    if ! docker push "${FRONTEND_IMAGE}:${VERSION}"; then
        error "Failed to push frontend image"
        exit 1
    fi
    if ! docker push "${FRONTEND_IMAGE}:latest"; then
        error "Failed to push frontend latest tag"
        exit 1
    fi
    
    log "Images pushed successfully!"
fi

# Print usage instructions
echo
log "To use these images with the Helm chart, update the values.yaml with:"
echo "  backend:"
echo "    image:"
echo "      repository: ${BACKEND_IMAGE}"
echo "      tag: ${VERSION}"
echo "  parse:"
echo "    image:"
echo "      repository: ${PARSE_IMAGE}"
echo "      tag: ${VERSION}"
echo "  frontend:"
echo "    image:"
echo "      repository: ${FRONTEND_IMAGE}"
echo "      tag: ${VERSION}" 