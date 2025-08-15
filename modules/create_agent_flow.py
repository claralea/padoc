from google.cloud.aiplatform.preview.agent import AgentServiceClient

PROJECT_ID = "rag-test-467013"
LOCATION = "us-central1"

client = AgentServiceClient()
parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"

agent_config = {
    "display_name": "batch-documentation-agent",
    "description": "Agent for generating pharma BPRs using local RAG endpoint.",
    "default_language_code": "en",
    "enable_tool_use": True,
}

response = client.create_agent(parent=parent, agent=agent_config)
print(f"✅ Created agent: {response.name}")


