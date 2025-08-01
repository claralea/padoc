from google.cloud import aiplatform_v1beta1 as aiplatform
from google.protobuf import struct_pb2

project = "rag-test-467013"
location = "us-central1"
agent_id = "YOUR_AGENT_ID"  # Replace with the one printed earlier

client = aiplatform.AgentServiceClient()
parent = f"projects/{project}/locations/{location}/agents/{agent_id}"

flow = {
    "display_name": "bpr-generation-flow",
    "start_node": {
        "next_steps": [
            {
                "call_rest_endpoint": {
                    "display_name": "call-rag-chat-endpoint",
                    "endpoint_config": {
                        "uri": "https://rag-agent-service-594158551617.us-central1.run.app/chat",
                        "method": "POST",
                        "request_body": struct_pb2.Struct(
                            fields={
                                "query": struct_pb2.Value(string_value="{{inputs.user_input}}"),
                                "chunk_type": struct_pb2.Value(string_value="recursive-split"),
                            }
                        ),
                        "response_mappings": {
                            "output": "rag_context"
                        }
                    },
                    "next_steps": [
                        {
                            "llm": {
                                "display_name": "generate-bpr",
                                "prompt": """
Using the following context from regulatory documents, generate a Batch Production Record (BPR) including:

- Materials used
- Process steps
- In-process quality checks
- Operator sign-off fields

Context:
{{steps.call-rag-chat-endpoint.rag_context}}
""",
                                "model": "gemini-1.5-flash",
                                "temperature": 0.2
                            },
                            "next_steps": [
                                {
                                    "return_response": {
                                        "text_response": "{{steps.generate-bpr.output}}"
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        ]
    }
}

created_flow = client.create_flow(parent=parent, flow=flow)
print(f"✅ Flow created: {created_flow.name}")
