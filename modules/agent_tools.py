# agent_tools.py - Enhanced version with full RAG integration

from vertexai.generative_models import Content, Part, FunctionDeclaration, Tool, GenerationResponse
from typing import List, Dict, Any, Optional
import json
import vertexai
from vertexai.generative_models import FunctionDeclaration, Tool, Part

# Specify a function declaration and parameters for an API request
get_book_by_author_func = FunctionDeclaration(
    name="get_book_by_author",
    description="Get the book chunks filtered by author name",
    # Function parameters are specified in OpenAPI JSON schema format
    parameters={
        "type": "object",
        "properties": {
            "author": {"type": "string", "description": "The author name","enum":["C. F. Langworthy and Caroline Louisa Hunt", "J. Twamley", "George E. Newell", "T. D. Curtis", "Charles Thom and W. W. Fisk", "Thomas Wilson Reid","Bob Brown", "Charles S. Brooks", "Pavlos Protopapas"]},
            "search_content": {"type": "string", "description": "The search text to filter content from books. The search term is compared against the book text based on cosine similarity. Expand the search term to a a sentence or two to get better matches"},
        },
        "required": ["author","search_content"],
    },
)
def get_book_by_author(author, search_content, collection, embed_func):

    query_embedding = embed_func(search_content)

    # Query based on embedding value 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"author":author}
    )
    return "\n".join(results["documents"][0])


get_book_by_search_content_func = FunctionDeclaration(
    name="get_book_by_search_content",
    description="Get the book chunks filtered by search terms",
    # Function parameters are specified in OpenAPI JSON schema format
    parameters={
        "type": "object",
        "properties": {
            "search_content": {"type": "string", "description": "The search text to filter content from books. The search term is compared against the book text based on cosine similarity. Expand the search term to a a sentence or two to get better matches"},
        },
        "required": ["search_content"],
    },
)
def get_book_by_search_content(search_content, collection, embed_func):

    query_embedding = embed_func(search_content)

    # Query based on embedding value 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    return "\n".join(results["documents"][0])

# Define all functions available to the cheese expert
cheese_expert_tool = Tool(function_declarations=[get_book_by_author_func,get_book_by_search_content_func])


def execute_function_calls(function_calls,collection, embed_func):
    parts = []
    for function_call in function_calls:
        print("Function:",function_call.name)
        if function_call.name == "get_book_by_author":
            print("Calling function with args:", function_call.args["author"], function_call.args["search_content"])
            response = get_book_by_author(function_call.args["author"], function_call.args["search_content"],collection, embed_func)
            print("Response:", response)
            #function_responses.append({"function_name":function_call.name, "response": response})
            parts.append(
					Part.from_function_response(
						name=function_call.name,
						response={
							"content": response,
						},
					),
			)
        if function_call.name == "get_book_by_search_content":
            print("Calling function with args:", function_call.args["search_content"])
            response = get_book_by_search_content(function_call.args["search_content"],collection, embed_func)
            print("Response:", response)
            #function_responses.append({"function_name":function_call.name, "response": response})
            parts.append(
					Part.from_function_response(
						name=function_call.name,
						response={
							"content": response,
						},
					),
			)

    
    return parts


# Enhanced tools for pharmaceutical deviation analysis
def get_deviation_analysis_tool():
    """Define the deviation analysis tool for Vertex AI"""
    
    get_relevant_regulations = FunctionDeclaration(
        name="get_relevant_regulations",
        description="Search for relevant pharmaceutical regulations and guidance",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for regulations"
                },
                "deviation_type": {
                    "type": "string",
                    "description": "Type of deviation (Equipment, Process, Procedure, etc.)",
                    "enum": ["Equipment", "Process", "Procedure", "Documentation", "Personnel", "Material", "Facility", "Formulation"]
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about the deviation"
                }
            },
            "required": ["query"]
        }
    )
    
    analyze_deviation_impact = FunctionDeclaration(
        name="analyze_deviation_impact",
        description="Analyze the impact of a deviation on product quality and patient safety",
        parameters={
            "type": "object",
            "properties": {
                "deviation_description": {
                    "type": "string",
                    "description": "Detailed description of the deviation"
                },
                "product_info": {
                    "type": "string",
                    "description": "Product name and batch information"
                },
                "regulatory_context": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant regulatory requirements"
                }
            },
            "required": ["deviation_description", "product_info"]
        }
    )
    
    suggest_investigation_approach = FunctionDeclaration(
        name="suggest_investigation_approach",
        description="Suggest investigation approach based on deviation type and regulations",
        parameters={
            "type": "object",
            "properties": {
                "deviation_type": {
                    "type": "string",
                    "description": "Type of deviation"
                },
                "description": {
                    "type": "string",
                    "description": "Deviation description"
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level",
                    "enum": ["Critical", "Major", "Minor"]
                }
            },
            "required": ["deviation_type", "description"]
        }
    )
    
    return Tool(
        function_declarations=[
            get_relevant_regulations,
            analyze_deviation_impact,
            suggest_investigation_approach
        ]
    )

# Enhanced cheese_expert_tool for backward compatibility
cheese_expert_tool = get_deviation_analysis_tool()

def execute_function_calls(function_calls, collection, embed_func=None, generative_model=None):
    """
    Execute function calls with RAG integration
    
    Args:
        function_calls: List of function calls from the model
        collection: ChromaDB collection for vector search
        embed_func: Function to generate embeddings
        generative_model: Vertex AI generative model for enhanced responses
    """
    function_responses = []
    
    for function_call in function_calls:
        function_name = function_call.name
        args = function_call.args
        
        if function_name == "get_relevant_regulations":
            # Search the RAG database for relevant regulations
            query = args.get("query", "")
            deviation_type = args.get("deviation_type", "")
            context = args.get("context", "")
            
            # Enhance query with deviation type and context
            enhanced_query = f"{query} {deviation_type} {context}".strip()
            
            if embed_func:
                query_embedding = embed_func(enhanced_query)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    where={"$or": [
                        {"book": {"$contains": "CFR"}},
                        {"book": {"$contains": "Q7"}},
                        {"book": {"$contains": "Manufacturing"}}
                    ]} if deviation_type else None
                )
                
                # Format results
                regulations = []
                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0] if results["documents"] else [],
                    results["metadatas"][0] if results["metadatas"] else [],
                    results["distances"][0] if results["distances"] else []
                )):
                    regulations.append({
                        "source": meta.get("book", "Unknown"),
                        "author": meta.get("author", ""),
                        "year": meta.get("year", ""),
                        "content": doc,
                        "relevance_score": 1 - dist  # Convert distance to relevance
                    })
                
                response_content = {
                    "regulations": regulations,
                    "total_found": len(regulations),
                    "query_used": enhanced_query
                }
            else:
                response_content = {
                    "error": "Embedding function not provided",
                    "regulations": []
                }
            
            function_responses.append(
                Part.from_function_response(
                    name=function_name,
                    response=response_content
                )
            )
        
        elif function_name == "analyze_deviation_impact":
            # Analyze deviation impact using both RAG and AI
            deviation_desc = args.get("deviation_description", "")
            product_info = args.get("product_info", "")
            regulatory_context = args.get("regulatory_context", [])
            
            impact_analysis = {
                "patient_safety_impact": "To be determined based on investigation",
                "product_quality_impact": "Assessment required",
                "regulatory_compliance_impact": "Review required against applicable regulations",
                "recommended_actions": [
                    "Immediate containment of affected batch",
                    "Risk assessment per ICH Q9",
                    "Investigation per site procedures"
                ]
            }
            
            # Search for similar deviations if collection available
            if collection and embed_func:
                similar_search = f"deviation impact {deviation_desc}"
                query_embedding = embed_func(similar_search)
                similar_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    where_document={"$contains": "impact"}
                )
                
                if similar_results["documents"]:
                    impact_analysis["similar_cases"] = [
                        {"content": doc[:200] + "...", "source": meta.get("book", "")}
                        for doc, meta in zip(
                            similar_results["documents"][0],
                            similar_results["metadatas"][0]
                        )
                    ]
            
            function_responses.append(
                Part.from_function_response(
                    name=function_name,
                    response=impact_analysis
                )
            )
        
        elif function_name == "suggest_investigation_approach":
            # Suggest investigation approach based on deviation type
            deviation_type = args.get("deviation_type", "")
            description = args.get("description", "")
            priority = args.get("priority", "Major")
            
            # Base investigation approach
            investigation_approach = {
                "immediate_actions": [
                    "Secure and quarantine affected materials/product",
                    "Document current state with photos if applicable",
                    "Notify QA management",
                    "Initiate deviation record"
                ],
                "investigation_steps": [
                    "Review batch records and associated documentation",
                    "Interview personnel involved",
                    "Examine equipment logs and calibration records",
                    "Perform root cause analysis (5 Whys, Fishbone)",
                    "Assess product impact"
                ],
                "timeline": "Complete within 30 days per SOP",
                "team_members": ["QA Lead", "Production Supervisor", "Technical Expert"]
            }
            
            # Customize based on deviation type
            if deviation_type == "Equipment":
                investigation_approach["investigation_steps"].insert(
                    2, "Review equipment maintenance and calibration history"
                )
                investigation_approach["team_members"].append("Engineering")
            elif deviation_type == "Documentation":
                investigation_approach["investigation_steps"].insert(
                    1, "Perform detailed documentation review and gap analysis"
                )
            elif deviation_type == "Process":
                investigation_approach["investigation_steps"].insert(
                    2, "Review process parameters and trend data"
                )
            
            # Search for specific investigation requirements
            if collection and embed_func:
                investigation_query = f"investigation requirements {deviation_type} pharmaceutical"
                query_embedding = embed_func(investigation_query)
                invest_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3
                )
                
                if invest_results["documents"]:
                    investigation_approach["regulatory_requirements"] = [
                        doc[:300] + "..." for doc in invest_results["documents"][0]
                    ]
            
            function_responses.append(
                Part.from_function_response(
                    name=function_name,
                    response=investigation_approach
                )
            )
    
    return function_responses

# Additional helper functions for the integrated system

def format_rag_results(results: Dict[str, Any], max_length: int = 500) -> str:
    """Format RAG search results for display"""
    if not results.get("documents"):
        return "No relevant documents found."
    
    formatted = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        source = meta.get("book", "Unknown")
        author = meta.get("author", "")
        year = meta.get("year", "")
        
        header = f"{i}. {source}"
        if author:
            header += f" ({author}"
            if year:
                header += f", {year}"
            header += ")"
        
        # Truncate document if too long
        content = doc if len(doc) <= max_length else doc[:max_length] + "..."
        
        formatted.append(f"{header}\n{content}\n")
    
    return "\n".join(formatted)

def extract_regulatory_requirements(documents: List[str], metadatas: List[Dict]) -> Dict[str, List[str]]:
    """Extract and categorize regulatory requirements from search results"""
    requirements = {
        "investigation": [],
        "documentation": [],
        "corrective_action": [],
        "risk_assessment": [],
        "reporting": []
    }
    
    keywords = {
        "investigation": ["investigate", "investigation", "root cause", "determine"],
        "documentation": ["document", "record", "maintain", "retain"],
        "corrective_action": ["corrective", "preventive", "CAPA", "action"],
        "risk_assessment": ["risk", "assessment", "impact", "evaluate"],
        "reporting": ["report", "notify", "communicate", "inform"]
    }
    
    for doc, meta in zip(documents, metadatas):
        doc_lower = doc.lower()
        source = meta.get("book", "Unknown")
        
        for category, terms in keywords.items():
            if any(term in doc_lower for term in terms):
                # Extract sentences containing keywords
                sentences = doc.split(".")
                for sentence in sentences:
                    if any(term in sentence.lower() for term in terms):
                        requirement = f"{sentence.strip()}. (Source: {source})"
                        if requirement not in requirements[category]:
                            requirements[category].append(requirement)
    
    return requirements

def generate_deviation_summary(session_data: Dict[str, Any]) -> str:
    """Generate a comprehensive summary of the deviation for reporting"""
    structured = session_data.get("structured_data", {})
    collected = session_data.get("collected_info", {})
    
    summary = f"""
DEVIATION SUMMARY
================

Basic Information:
- Title: {structured.get('title', 'Not specified')}
- Type: {structured.get('type', 'Not specified')}
- Date: {structured.get('date', 'Not specified')}
- Batch/Lot: {structured.get('batch', 'Not specified')}
- Quantity: {structured.get('quantity', 'Not specified')}
- Department(s): {structured.get('department', 'Not specified')}
- Planned: {structured.get('planned', 'No')}

Description:
{collected.get('original_description', 'No description provided')}

AI Analysis:
{collected.get('enhanced_description', 'Analysis pending')}

Risk Assessment:
{collected.get('risk_assessment', 'Assessment pending')}

Suggested Actions:
{collected.get('suggested_actions', 'Actions pending')}
"""
    
    return summary

# Integration functions for the complete system

class IntegratedRAGSystem:
    """Class to manage the integrated RAG and agent system"""
    
    def __init__(self, collection, embed_func, generative_model):
        self.collection = collection
        self.embed_func = embed_func
        self.generative_model = generative_model
        self.deviation_tool = get_deviation_analysis_tool()
    
    def search_with_context(self, query: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced search with contextual filtering"""
        query_embedding = self.embed_func(query)
        
        search_params = {
            "query_embeddings": [query_embedding],
            "n_results": 10
        }
        
        if filters:
            search_params["where"] = filters
        
        results = self.collection.query(**search_params)
        
        # Post-process results to add context
        enhanced_results = {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
            "formatted": format_rag_results(results),
            "requirements": extract_regulatory_requirements(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0]
            )
        }
        
        return enhanced_results
    
    def generate_regulatory_report(self, deviation_data: Dict) -> str:
        """Generate a regulatory-compliant deviation report"""
        
        # Search for report requirements
        report_reqs = self.search_with_context(
            "deviation report requirements format content pharmaceutical"
        )
        
        # Build the report template based on regulations
        report = f"""
PHARMACEUTICAL DEVIATION REPORT
==============================

Report Number: {deviation_data.get('record_id', 'TBD')}
Date: {datetime.now().strftime('%Y-%m-%d')}

1. DEVIATION IDENTIFICATION
---------------------------
{self._format_section(deviation_data, 'identification')}

2. REGULATORY CONTEXT
--------------------
Based on applicable regulations:
{report_reqs['formatted'][:1000]}

3. IMPACT ASSESSMENT
-------------------
{self._format_section(deviation_data, 'impact')}

4. ROOT CAUSE ANALYSIS
---------------------
{self._format_section(deviation_data, 'root_cause')}

5. CORRECTIVE AND PREVENTIVE ACTIONS
-----------------------------------
{self._format_section(deviation_data, 'capa')}

6. REGULATORY COMPLIANCE
-----------------------
This report has been prepared in accordance with:
- 21 CFR Part 211 (where applicable)
- ICH Q7 Good Manufacturing Practice
- Site Quality Management System requirements

7. APPROVAL
-----------
Quality Assurance Review: _________________ Date: _________
Management Approval: _____________________ Date: _________
"""
        
        return report
    
    def _format_section(self, data: Dict, section: str) -> str:
        """Format specific sections of the report"""
        if section == "identification":
            return f"""
Title: {data.get('title', 'Not specified')}
Type: {data.get('deviation_type', 'Not specified')}
Date of Occurrence: {data.get('date_of_occurrence', 'Not specified')}
Discovered By: {data.get('initiator', 'Not specified')}
Batch/Lot Number: {data.get('batch_lot_number', 'Not specified')}
Quantity Affected: {data.get('quantity_impacted', 'Not specified')}
"""
        elif section == "impact":
            return data.get('risk_assessment', 'Impact assessment pending')
        elif section == "root_cause":
            return data.get('root_cause_analysis', 'Root cause analysis pending')
        elif section == "capa":
            return data.get('corrective_actions', 'CAPA pending')
        else:
            return "Section not available"
    
    def validate_deviation_data(self, deviation_data: Dict) -> Dict[str, List[str]]:
        """Validate deviation data against regulatory requirements"""
        errors = []
        warnings = []
        
        # Required fields per regulations
        required_fields = [
            'title', 'deviation_type', 'date_of_occurrence', 
            'description', 'initiator', 'batch_lot_number'
        ]
        
        for field in required_fields:
            if not deviation_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Check for investigation requirements
        if deviation_data.get('priority') == 'Critical' and not deviation_data.get('immediate_actions'):
            warnings.append("Critical deviation should have immediate actions documented")
        
        # Search for specific validation requirements
        validation_reqs = self.search_with_context(
            f"deviation documentation requirements {deviation_data.get('deviation_type', '')}"
        )
        
        return {
            "errors": errors,
            "warnings": warnings,
            "regulatory_gaps": self._check_regulatory_gaps(deviation_data, validation_reqs)
        }
    
    def _check_regulatory_gaps(self, data: Dict, requirements: Dict) -> List[str]:
        """Check for gaps against regulatory requirements"""
        gaps = []
        
        # Extract specific requirements from search results
        req_categories = requirements.get('requirements', {})
        
        if 'investigation' in req_categories and not data.get('investigation_plan'):
            gaps.append("Investigation plan required per regulations")
        
        if 'risk_assessment' in req_categories and not data.get('risk_assessment'):
            gaps.append("Risk assessment required per regulations")
        
        return gaps