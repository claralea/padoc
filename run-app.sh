#!/bin/bash
set -e

# Set variables (NO SECRETS HERE)
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/secrets/
export GCP_PROJECT="${GCP_PROJECT:-rag-test-467013}"
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/cl-rag-docs.json"
export IMAGE_NAME="llm-rag-cli"

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

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