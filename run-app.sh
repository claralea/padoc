#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set variables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/secrets/
export GCP_PROJECT="rag-test-467013" # CHANGE TO YOUR PROJECT ID
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/cl-rag-docs.json"
export OPENAI_API_KEY=$(cat ./secrets/openai-key.txt)
export IMAGE_NAME="llm-rag-cli"

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Check for command line argument
MODE=${1:-streamlit}

case $MODE in
    "streamlit")
        echo "🚀 Starting Streamlit web app..."
        echo "📱 Access your app at: http://localhost:8501"
        docker-compose run --rm --service-ports $IMAGE_NAME streamlit
        ;;
    "cli")
        echo "💻 Starting CLI version..."
        docker-compose run --rm --service-ports $IMAGE_NAME cli
        ;;
    "shell")
        echo "🐚 Starting interactive shell..."
        docker-compose run --rm --service-ports $IMAGE_NAME shell
        ;;
    *)
        echo "Usage: $0 [streamlit|cli|shell]"
        echo ""
        echo "Options:"
        echo "  streamlit  - Start Streamlit web interface (default)"
        echo "  cli        - Start command-line interface"
        echo "  shell      - Start interactive shell"
        exit 1
        ;;
esac