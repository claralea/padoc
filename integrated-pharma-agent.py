import os
import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb
from chromadb import PersistentClient
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Vertex AI imports
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig
from google.api_core.exceptions import InternalServerError, ServiceUnavailable, ResourceExhausted

# Import the deviation data structures and tools from your smolagent notebook
from smolagents import tool, ToolCallingAgent, LiteLLMModel

# ============================================
# CONFIGURATION
# ============================================
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-2.5-flash"
INPUT_FOLDER = "input-datasets"
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000

# ============================================
# ENHANCED AGENT TOOLS WITH RAG INTEGRATION
# ============================================

class DeviationRAGTools:
    """Tools that integrate RAG capabilities with deviation analysis"""
    
    def __init__(self, collection, embed_func):
        self.collection = collection
        self.embed_func = embed_func
    
    def search_regulatory_guidance(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search regulatory documents for relevant guidance"""
        query_embedding = self.embed_func(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def find_similar_deviations(self, deviation_description: str) -> List[str]:
        """Find similar historical deviations from the knowledge base"""
        # Search for similar deviation patterns
        results = self.search_regulatory_guidance(
            f"deviation {deviation_description}", 
            n_results=3
        )
        
        similar_cases = []
        for doc, metadata in zip(results["documents"], results["metadatas"]):
            if "deviation" in doc.lower() or "nonconformance" in doc.lower():
                similar_cases.append({
                    "source": metadata.get("book", "Unknown"),
                    "content": doc[:200] + "..."  # Truncate for readability
                })
        
        return similar_cases

# Create enhanced tools that use both RAG and LLM
@tool
def analyze_deviation_with_regulations(description: str, rag_tools: DeviationRAGTools) -> str:
    """
    Analyze deviation using both AI and regulatory knowledge base.
    
    Args:
        description: The deviation description
        rag_tools: RAG tools instance
    
    Returns:
        Enhanced analysis with regulatory context
    """
    # First, search for relevant regulatory guidance
    regulatory_results = rag_tools.search_regulatory_guidance(
        f"pharmaceutical deviation {description} investigation requirements",
        n_results=5
    )
    
    # Build context from regulatory documents
    regulatory_context = "\n\n".join([
        f"From {meta.get('book', 'Regulation')}: {doc}"
        for doc, meta in zip(regulatory_results["documents"], regulatory_results["metadatas"])
    ])
    
    # Now use AI with regulatory context
    prompt = f"""
    You are a pharmaceutical quality expert analyzing a deviation report.
    
    Deviation description: "{description}"
    
    Relevant regulatory guidance:
    {regulatory_context}
    
    Based on the regulatory requirements above, provide:
    1. Enhanced technical description with regulatory context
    2. Specific regulatory requirements that apply
    3. Critical quality attributes that may be affected
    4. Required investigation elements per regulations
    
    Be specific and cite the regulatory sources when applicable.
    """
    
    # Use the existing OpenAI client or Vertex AI model
    response = generate_ai_response(prompt)
    return response

@tool
def suggest_corrective_actions_with_guidance(description: str, deviation_type: str, rag_tools: DeviationRAGTools) -> str:
    """
    Suggest corrective actions based on regulations and best practices.
    
    Args:
        description: The deviation description
        deviation_type: Type of deviation
        rag_tools: RAG tools instance
    
    Returns:
        Corrective actions with regulatory basis
    """
    # Search for corrective action requirements
    ca_results = rag_tools.search_regulatory_guidance(
        f"corrective action preventive action CAPA {deviation_type} pharmaceutical",
        n_results=5
    )
    
    # Find similar historical deviations
    similar_cases = rag_tools.find_similar_deviations(description)
    
    regulatory_context = "\n\n".join([
        f"From {meta.get('book', 'Regulation')}: {doc}"
        for doc, meta in zip(ca_results["documents"], ca_results["metadatas"])
    ])
    
    similar_context = "\n\n".join([
        f"Similar case from {case['source']}: {case['content']}"
        for case in similar_cases
    ])
    
    prompt = f"""
    You are a pharmaceutical quality expert suggesting corrective actions.
    
    Deviation: "{description}"
    Type: {deviation_type}
    
    Regulatory requirements for corrective actions:
    {regulatory_context}
    
    Similar historical cases:
    {similar_context}
    
    Based on regulations and similar cases, suggest:
    1. Immediate corrective actions with regulatory basis
    2. Long-term preventive actions per CAPA requirements
    3. Effectiveness checks as required by regulations
    4. Documentation requirements
    
    Cite specific regulatory requirements where applicable.
    """
    
    response = generate_ai_response(prompt)
    return response

@tool
def perform_risk_assessment_with_guidance(description: str, batch_info: str, rag_tools: DeviationRAGTools) -> str:
    """
    Perform risk assessment using regulatory framework.
    
    Args:
        description: The deviation description
        batch_info: Batch/lot information
        rag_tools: RAG tools instance
    
    Returns:
        Risk assessment with regulatory framework
    """
    # Search for risk assessment requirements
    risk_results = rag_tools.search_regulatory_guidance(
        "risk assessment quality risk management pharmaceutical deviation",
        n_results=5
    )
    
    regulatory_context = "\n\n".join([
        f"From {meta.get('book', 'Regulation')}: {doc}"
        for doc, meta in zip(risk_results["documents"], risk_results["metadatas"])
    ])
    
    prompt = f"""
    You are performing a quality risk assessment for a pharmaceutical deviation.
    
    Deviation: "{description}"
    Batch Information: {batch_info}
    
    Regulatory requirements for risk assessment:
    {regulatory_context}
    
    Provide a risk assessment including:
    1. Patient safety impact assessment
    2. Product quality impact assessment
    3. Regulatory compliance impact
    4. Risk level determination (High/Medium/Low) with justification
    5. Required risk mitigation measures
    
    Use ICH Q9 principles and cite regulatory requirements.
    """
    
    response = generate_ai_response(prompt)
    return response

# ============================================
# INTEGRATED DEVIATION INTERVIEW AGENT
# ============================================

class IntegratedDeviationAgent:
    """Enhanced agent that combines interview flow with RAG capabilities"""
    
    def __init__(self, openai_api_key: str, rag_collection, embed_func, generative_model):
        self.openai_api_key = openai_api_key
        self.rag_collection = rag_collection
        self.embed_func = embed_func
        self.generative_model = generative_model
        
        # Initialize RAG tools
        self.rag_tools = DeviationRAGTools(rag_collection, embed_func)
        
        # Initialize the agent with enhanced tools
        self.agent = ToolCallingAgent(
            tools=[
                lambda desc: analyze_deviation_with_regulations(desc, self.rag_tools),
                lambda desc, dtype: suggest_corrective_actions_with_guidance(desc, dtype, self.rag_tools),
                lambda desc, batch: perform_risk_assessment_with_guidance(desc, batch, self.rag_tools)
            ],
            model=LiteLLMModel("gpt-4o-mini")
        )
        
        # Session storage
        self.current_session = {}
        self.interview_sessions = {}
        
        # Keep all the interview flow logic from your original agent
        self.data_questions = {
            "initiator": "What is your name (the person initiating this deviation report)?",
            "title": "What is the title of this deviation?",
            "type": "What type of deviation is this?\nOptions: Equipment, Process, Procedure, Documentation, Personnel, Material, Facility, Formulation",
            "date": "When did this deviation occur? (Please provide date in MM/DD/YYYY format)",
            "batch": "What is the batch/lot number affected?",
            "quantity": "What quantity was impacted?",
            "planned": "Is this a planned deviation? (yes/no)",
            "department": "Which department(s) are involved?"
        }
    
    def search_regulatory_context(self, query: str) -> str:
        """Search and format regulatory context for a query"""
        results = self.rag_tools.search_regulatory_guidance(query, n_results=3)
        
        if not results["documents"]:
            return "No specific regulatory guidance found."
        
        context = "📚 **Relevant Regulatory Guidance:**\n\n"
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
            source = meta.get("book", "Unknown source")
            author = meta.get("author", "")
            year = meta.get("year", "")
            
            context += f"**{i}. From {source}**"
            if author:
                context += f" ({author}"
                if year:
                    context += f", {year}"
                context += ")"
            context += f"\n{doc[:300]}...\n\n"
        
        return context
    
    def process_description_with_rag(self, session: Dict, description: str) -> str:
        """Enhanced description processing with RAG context"""
        
        # Store the original description
        session["collected_info"]["original_description"] = description
        
        # Get regulatory context
        regulatory_context = self.search_regulatory_context(
            f"deviation investigation requirements {description}"
        )
        
        # Find similar cases
        similar_cases = self.rag_tools.find_similar_deviations(description)
        
        # Use enhanced analysis tool
        try:
            enhanced_analysis = analyze_deviation_with_regulations(description, self.rag_tools)
            session["collected_info"]["enhanced_description"] = enhanced_analysis
            
            # Get risk assessment
            batch_info = session.get("structured_data", {}).get("batch", "Not specified")
            risk_assessment = perform_risk_assessment_with_guidance(description, batch_info, self.rag_tools)
            session["collected_info"]["risk_assessment"] = risk_assessment
            
            session["interview_stage"] = "enhancement"
            
            response = f"""
**Thank you for the detailed description!**

**Original Description:**
{description}

{regulatory_context}

**🤖 AI-Enhanced Analysis with Regulatory Context:**
{enhanced_analysis}

**⚠️ Risk Assessment:**
{risk_assessment}
"""
            
            if similar_cases:
                response += "\n\n**📋 Similar Historical Cases:**\n"
                for case in similar_cases[:2]:
                    response += f"- From {case['source']}: {case['content']}\n"
            
            response += """

---

**Does this enhanced analysis capture the situation accurately?**
Please let me know if you'd like to modify anything or add additional details.

(Type 'yes' to continue or provide additional information)
"""
            
        except Exception as e:
            response = f"""
I've recorded your description but encountered an issue with the analysis: {str(e)}

Would you like to continue with corrective action suggestions? (Type 'yes' to continue)
"""
            session["interview_stage"] = "enhancement"
        
        return response
    
    def generate_corrective_actions_with_rag(self, session: Dict) -> str:
        """Generate corrective actions with regulatory guidance"""
        
        description = session["collected_info"]["original_description"]
        deviation_type = session["structured_data"].get("type", "")
        
        try:
            # Get regulatory-based corrective actions
            corrective_actions = suggest_corrective_actions_with_guidance(
                description, deviation_type, self.rag_tools
            )
            session["collected_info"]["suggested_actions"] = corrective_actions
            
            # Search for specific CAPA requirements
            capa_context = self.search_regulatory_context(
                "corrective action preventive action effectiveness verification"
            )
            
            session["interview_stage"] = "corrective_actions"
            
            response = f"""
**🔧 Suggested Corrective Actions Based on Regulatory Requirements:**

{corrective_actions}

{capa_context}

---

**Do these corrective actions look appropriate?**
- Type 'yes' to accept these suggestions
- Provide additional actions you'd like to include
- Type 'modify' if you want to change any of these actions
"""
            
        except Exception as e:
            response = self._generate_fallback_corrective_actions(session)
        
        return response
    
    # Include all other methods from your original agent (start_interview, process_user_input, etc.)
    # but enhance them to use RAG where appropriate

# ============================================
# MAIN INTEGRATION FUNCTION
# ============================================

def create_integrated_deviation_system(openai_api_key: str, chunk_type: str = "char-split"):
    """
    Create and initialize the integrated deviation system with RAG and interview capabilities
    """
    # Initialize Vertex AI
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    
    # Initialize embedding model
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    
    # Initialize generative model
    generative_model = GenerativeModel(
        GENERATIVE_MODEL,
        system_instruction=["""
        You are an AI assistant specialized in pharmaceutical manufacturing compliance and quality.
        You help with deviation reports, investigations, and corrective actions based on regulatory requirements.
        Always cite relevant regulations when providing guidance.
        """]
    )
    
    # Connect to ChromaDB
    client = PersistentClient(path="./chroma")
    collection_name = f"{chunk_type}-collection"
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ Connected to collection: {collection_name}")
        print(f"📚 Documents in collection: {collection.count()}")
    except Exception as e:
        print(f"❌ Error: Collection '{collection_name}' not found. Please run the chunk, embed, and load steps first.")
        return None
    
    # Create embedding function
    def generate_query_embedding(query):
        query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
        kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
        embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
        return embeddings[0].values
    
    # Initialize the integrated agent
    agent = IntegratedDeviationAgent(
        openai_api_key=openai_api_key,
        rag_collection=collection,
        embed_func=generate_query_embedding,
        generative_model=generative_model
    )
    
    return agent

# ============================================
# HELPER FUNCTION FOR AI RESPONSES
# ============================================

def generate_ai_response(prompt: str) -> str:
    """Generate AI response using available model (OpenAI or Vertex AI)"""
    # This should use your configured AI model
    # For now, returning a placeholder
    return "AI response would be generated here based on the prompt and context"

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main function to run the integrated deviation agent"""
    
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set your OpenAI API key: os.environ['OPENAI_API_KEY'] = 'your-key'")
        return
    
    # Create the integrated system
    print("🚀 Initializing Integrated Pharmaceutical Deviation System...")
    agent = create_integrated_deviation_system(openai_api_key)
    
    if not agent:
        return
    
    print("✅ System initialized successfully!")
    print("\n" + "="*60)
    print("PHARMACEUTICAL DEVIATION REPORTING SYSTEM")
    print("Powered by RAG + AI Agent Technology")
    print("="*60 + "\n")
    
    # Start interview session
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # You can now use the agent for interviews with full RAG support
    # The agent will automatically search regulatory documents and provide
    # context-aware responses based on your loaded regulations
    
    print("Ready to start deviation interviews with regulatory guidance!")
    print("The system will automatically search and cite relevant regulations.")
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Pharmaceutical Deviation Agent")
    parser.add_argument("--chunk_type", default="char-split", help="Type of chunking to use")
    parser.add_argument("--mode", default="interview", choices=["interview", "search", "report"],
                       help="Mode of operation")
    args = parser.parse_args()
    
    main()