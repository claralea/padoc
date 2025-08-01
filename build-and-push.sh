#!/bin/bash
set -e

# ---- SET THESE VARIABLES ----
REGION="us-central1"
PROJECT_ID="rag-test-467013"
REPO="llm-rag-repo"
IMAGE_NAME="llm-rag"
TAG="v2"  # increment when needed
# -----------------------------

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image for Vertex AI Agent..."
docker buildx build \
  --platform linux/amd64 \
  -t ${IMAGE_URI} \
  -f Dockerfile \
  .

echo "Authenticating Docker with Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

echo "Pushing image to Artifact Registry..."
docker push ${IMAGE_URI}

echo "Image pushed successfully: ${IMAGE_URI}"
