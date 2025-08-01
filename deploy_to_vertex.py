# deploy_to_vertex.py - Deployment configuration for Vertex AI

import os
from google.cloud import aiplatform
from datetime import datetime

# Deployment configuration
DEPLOYMENT_CONFIG = {
    "project_id": os.environ.get("GCP_PROJECT"),
    "location": "us-central1",
    "model_display_name": "pharma-deviation-agent",
    "endpoint_display_name": "pharma-deviation-endpoint",
    "machine_type": "n1-standard-4",
    "min_replica_count": 1,
    "max_replica_count": 3,
    "accelerator_type": None,  # Add if needed
    "accelerator_count": 0
}

# Create prediction service
class DeviationPredictionService:
    """Service class for Vertex AI predictions"""
    
    def __init__(self):
        self.collection = None
        self.agent = None
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize ChromaDB and agent on startup"""
        # This runs when the container starts
        from cli import PersistentClient, generate_query_embedding, generative_model
        from run_integrated_agent import IntegratedDeviationInterviewAgent
        
        # Connect to ChromaDB
        self.collection = PersistentClient(path="./chroma").get_collection(
            name="char-split-collection"
        )
        
        # Initialize agent
        openai_key = os.environ.get("OPENAI_API_KEY")
        self.agent = IntegratedDeviationInterviewAgent(
            openai_api_key=openai_key,
            collection=self.collection,
            embed_func=generate_query_embedding,
            vertex_model=generative_model
        )
    
    def predict(self, instances):
        """
        Vertex AI prediction endpoint
        
        Args:
            instances: List of prediction requests
            
        Returns:
            List of prediction responses
        """
        predictions = []
        
        for instance in instances:
            action = instance.get("action", "process")
            
            if action == "start_session":
                # Start new session
                session_id, greeting = self.agent.start_session()
                predictions.append({
                    "session_id": session_id,
                    "response": greeting,
                    "status": "active"
                })
                
            elif action == "process_input":
                # Process user input
                session_id = instance.get("session_id")
                user_input = instance.get("input", "")
                
                if not session_id:
                    predictions.append({
                        "error": "session_id required",
                        "status": "error"
                    })
                else:
                    response = self.agent.process_input(session_id, user_input)
                    predictions.append({
                        "session_id": session_id,
                        "response": response,
                        "status": "active"
                    })
                    
            elif action == "search_regulations":
                # Direct RAG search
                query = instance.get("query", "")
                results = self.agent.rag_system.search_with_context(query)
                predictions.append({
                    "query": query,
                    "results": results["formatted"],
                    "status": "success"
                })
                
            else:
                predictions.append({
                    "error": f"Unknown action: {action}",
                    "status": "error"
                })
        
        return predictions

# Model serving configuration
def create_model_config():
    """Create model configuration for Vertex AI"""
    return {
        "display_name": DEPLOYMENT_CONFIG["model_display_name"],
        "description": "Pharmaceutical Deviation Reporting Agent with RAG",
        "version_aliases": ["production", f"v_{datetime.now().strftime('%Y%m%d')}"],
        "labels": {
            "application": "pharma-deviation",
            "type": "rag-agent",
            "framework": "vertex-ai"
        }
    }

# Endpoint configuration
def create_endpoint_config():
    """Create endpoint configuration"""
    return {
        "display_name": DEPLOYMENT_CONFIG["endpoint_display_name"],
        "description": "API endpoint for pharmaceutical deviation reporting",
        "labels": {
            "environment": "production",
            "application": "pharma-deviation"
        }
    }

# Docker configuration for custom container
DOCKERFILE_CONTENT = """
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Copy pre-built ChromaDB data
COPY ./chroma /app/chroma

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Health check endpoint
HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1

# Run the service
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
"""

# Requirements file
REQUIREMENTS_CONTENT = """
# Core dependencies
google-cloud-aiplatform>=1.38.0
vertexai>=1.38.0
chromadb>=0.4.0
openai>=1.0.0
smolagents>=0.1.0
pandas>=2.0.0
numpy>=1.24.0

# Web framework for API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Document generation
weasyprint>=60.0
python-docx>=1.0.0

# Utilities
python-multipart>=0.0.6
aiofiles>=23.0.0
"""

# FastAPI application wrapper
MAIN_APP_CONTENT = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from deploy_to_vertex import DeviationPredictionService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pharmaceutical Deviation Agent API",
    description="AI-powered deviation reporting with regulatory compliance",
    version="1.0.0"
)

# Initialize prediction service
try:
    prediction_service = DeviationPredictionService()
    logger.info("Prediction service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize prediction service: {e}")
    prediction_service = None

# Request/Response models
class PredictionRequest(BaseModel):
    instances: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = {}

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Vertex AI"""
    if prediction_service and prediction_service.collection:
        return {
            "status": "healthy",
            "service": "pharma-deviation-agent",
            "collection_docs": prediction_service.collection.count()
        }
    else:
        raise HTTPException(status_code=503, detail="Service not ready")

# Prediction endpoint (Vertex AI compatible)
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint for Vertex AI"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        predictions = prediction_service.predict(request.instances)
        return PredictionResponse(
            predictions=predictions,
            metadata={"model_version": "1.0.0"}
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional convenience endpoints
@app.post("/start_session")
async def start_session():
    """Start a new deviation reporting session"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        session_id, greeting = prediction_service.agent.start_session()
        return {
            "session_id": session_id,
            "message": greeting,
            "status": "session_started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_input/{session_id}")
async def process_input(session_id: str, user_input: str):
    """Process user input for a session"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        response = prediction_service.agent.process_input(session_id, user_input)
        return {
            "session_id": session_id,
            "response": response,
            "status": "processed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_regulations")
async def search_regulations(query: str, limit: int = 5):
    """Search regulatory database"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = prediction_service.agent.rag_system.search_with_context(query)
        return {
            "query": query,
            "results": results["formatted"][:limit],
            "total_found": len(results["documents"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Pharmaceutical Deviation Agent",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "start_session": "/start_session (POST)",
            "process_input": "/process_input/{session_id} (POST)",
            "search_regulations": "/search_regulations (POST)"
        }
    }
'''

# Deployment script
def deploy_to_vertex():
    """Deploy the model to Vertex AI"""
    
    print("🚀 Deploying to Vertex AI...")
    
    # Initialize Vertex AI
    aiplatform.init(
        project=DEPLOYMENT_CONFIG["project_id"],
        location=DEPLOYMENT_CONFIG["location"]
    )
    
    # Steps for deployment:
    print("""
    Deployment Steps:
    
    1. Build Docker container:
       docker build -t gcr.io/{project_id}/pharma-deviation-agent:latest .
    
    2. Push to Container Registry:
       docker push gcr.io/{project_id}/pharma-deviation-agent:latest
    
    3. Create Model in Vertex AI:
       - Use custom container
       - Set prediction route to /predict
       - Set health route to /health
    
    4. Deploy to Endpoint:
       - Machine type: {machine_type}
       - Min replicas: {min_replica_count}
       - Max replicas: {max_replica_count}
    
    5. Test the endpoint:
       curl -X POST https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/ENDPOINT_ID:predict
    """.format(**DEPLOYMENT_CONFIG))
    
    # Save configuration files
    with open("Dockerfile", "w") as f:
        f.write(DOCKERFILE_CONTENT)
    
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS_CONTENT)
    
    with open("main.py", "w") as f:
        f.write(MAIN_APP_CONTENT)
    
    print("\n✅ Deployment files created successfully!")
    print("📁 Files created: Dockerfile, requirements.txt, main.py")
    print("\n⚠️  Remember to:")
    print("1. Set OPENAI_API_KEY in your container environment")
    print("2. Copy your ChromaDB data to ./chroma directory")
    print("3. Test locally before deploying to Vertex AI")

if __name__ == "__main__":
    deploy_to_vertex()