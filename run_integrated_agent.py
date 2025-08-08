# simplified_deviation_agent.py - Streamlined Pharmaceutical Deviation Reporting System

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import openai

# Add imports from your existing code
from cli import (
    vertexai, 
    GCP_PROJECT, 
    GCP_LOCATION,
    embedding_model,
    generative_model,
    generate_query_embedding,
    PersistentClient
)

from deviation_structures import (
    DeviationType, Priority, Status, Department,
    DeviationReport, DeviationInfo, HeaderInfo,
    Investigation, RiskAssessment, RootCauseAnalysis,
    AIGeneratedContent, DeviationDataValidator
)

from agent_tools import IntegratedRAGSystem
from report_generator import SimplifiedReportGenerator

class SimplifiedDeviationAgent:
    """
    Simplified agent for pharmaceutical deviation reporting
    """
    
    def __init__(self, openai_api_key: str, collection, embed_func, vertex_model):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.collection = collection
        self.embed_func = embed_func
        self.vertex_model = vertex_model
        
        # Initialize RAG system
        self.rag_system = IntegratedRAGSystem(collection, embed_func, vertex_model)
        
        # Valid deviation types and departments
        self.deviation_types = [
            "Equipment", "Process", "Procedure", "Documentation", 
            "Personnel", "Material", "Facility", "Formulation", 
            "Packaging", "Visual Inspection"
        ]
        
        self.departments = [
            "Manufacturing", "QA", "QC", "Warehouse", "R&D", 
            "Engineering", "Regulatory", "Quality"
        ]
        
        self.current_session = None
    
    def start_new_session(self) -> str:
        """Start a new deviation reporting session"""
        session_id = f"DEV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            "id": session_id,
            "start_time": datetime.now(),
            "stage": "greeting",
            "data": {}
        }
        
        return """
🏥 **Welcome to the Pharmaceutical Deviation Reporting System**

I'll help you create a comprehensive deviation report that complies with GMP requirements.

**Please provide your name:**
"""
    
    def process_input(self, user_input: str) -> str:
        """Process user input based on current stage"""
        
        if not self.current_session:
            return "No active session. Please start a new session."
        
        stage = self.current_session["stage"]
        
        if stage == "greeting":
            return self._handle_name(user_input)
        elif stage == "description":
            return self._handle_description(user_input)
        elif stage == "missing_info":
            return self._handle_missing_info(user_input)
        elif stage == "confirmation":
            return self._handle_confirmation(user_input)
        else:
            return "Unknown stage. Please start a new session."
    
    def _handle_name(self, name: str) -> str:
        """Handle name input and move to description"""
        self.current_session["data"]["initiator"] = name.strip()
        self.current_session["stage"] = "description"
        
        return f"""
Thank you, {name}!

**Please provide a detailed description of the deviation, including:**
- What happened?
- Batch/lot number (if applicable)
- Quantity impacted
- Date of occurrence
- Any other relevant details

*Please be as detailed as possible - this will help generate a complete report.*
"""
        # --- Helpers for safe JSON extraction & normalization ---
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Return the first valid top-level JSON object in the text."""
        import json, re
        if not text:
            return {}
        
        # Strip code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE|re.DOTALL).strip()
        
        # Try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # Fallback: find the first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

    def _coerce_extracted(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize fields to the expected shapes/values."""
        import re
        
        if not isinstance(data, dict):
            data = {}
        
        out = {
            "title": (data.get("title") or "").strip(),
            "deviation_type": (data.get("deviation_type") or "").strip(),
            "date_of_occurrence": (data.get("date_of_occurrence") or "").strip(),
            "batch_lot_number": (data.get("batch_lot_number") or "").strip(),
            "quantity_impacted": (data.get("quantity_impacted") or "").strip(),
            "department": (data.get("department") or "").strip(),
            "is_planned_deviation": data.get("is_planned_deviation")
        }
        
        # Canonicalize deviation type
        valid_types = {
            t.lower(): t for t in self.deviation_types
        }
        t = out["deviation_type"].lower()
        out["deviation_type"] = valid_types.get(t, out["deviation_type"])
        
        # Canonicalize department (map common synonyms)
        dept_map = {
            "manufacturing": "Manufacturing",
            "qa": "QA",
            "quality assurance": "QA",
            "qc": "QC",
            "quality control": "QC",
            "warehouse": "Warehouse",
            "r&d": "R&D",
            "rnd": "R&D",
            "engineering": "Engineering",
            "regulatory": "Regulatory",
            "quality": "Quality",
        }
        d_raw = out["department"].lower()
        
        # If user gave "QA/QC" or "QA, QC" pick the first valid
        split = [p.strip() for p in re.split(r"[\/,;]", out["department"])] if out["department"] else []
        chosen = None
        for part in split or [out["department"]]:
            key = part.strip().lower()
            if key in dept_map:
                chosen = dept_map[key]
                break
        if not chosen and d_raw in dept_map:
            chosen = dept_map[d_raw]
        out["department"] = chosen or out["department"]
        
        # Coerce planned deviation to boolean or None
        v = str(out["is_planned_deviation"]).strip().lower()
        if v in ("true", "yes", "y", "planned"):
            out["is_planned_deviation"] = True
        elif v in ("false", "no", "n", "unplanned"):
            out["is_planned_deviation"] = False
        elif v in ("", "none", "null", "n/a", "na"):
            out["is_planned_deviation"] = None
        
        return out

    def _is_missing(self, value: Any) -> bool:
        """Consistent 'missing' check for required fields."""
        if value is None:
            return True
        if isinstance(value, bool):
            return False
        s = str(value).strip().lower()
        return s in ("", "null", "none", "n/a", "na")

    def _coerce_extracted(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize fields to the expected shapes/values."""
        if not isinstance(data, dict):
            data = {}
        out = {
            "title": (data.get("title") or "").strip(),
            "deviation_type": (data.get("deviation_type") or "").strip(),
            "date_of_occurrence": (data.get("date_of_occurrence") or "").strip(),
            "batch_lot_number": (data.get("batch_lot_number") or "").strip(),
            "quantity_impacted": (data.get("quantity_impacted") or "").strip(),
            "department": (data.get("department") or "").strip(),
            "is_planned_deviation": data.get("is_planned_deviation")
        }

        # Canonicalize deviation type
        valid_types = {
            t.lower(): t for t in [
                "Equipment","Process","Procedure","Documentation","Personnel",
                "Material","Facility","Formulation","Packaging","Visual Inspection"
            ]
        }
        t = out["deviation_type"].lower()
        out["deviation_type"] = valid_types.get(t, out["deviation_type"])

        # Canonicalize department (map common synonyms)
        dept_map = {
            "manufacturing": "Manufacturing",
            "qa": "QA",
            "quality assurance": "QA",
            "qc": "QC",
            "quality control": "QC",
            "warehouse": "Warehouse",
            "r&d": "R&D",
            "rnd": "R&D",
            "engineering": "Engineering",
            "regulatory": "Regulatory",
            "quality": "Quality",
        }
        d_raw = out["department"].lower()
        # if user gave "QA/QC" or "QA, QC" pick the first valid
        split = [p.strip() for p in re.split(r"[\/,;]", out["department"])] if out["department"] else []
        chosen = None
        for part in split or [out["department"]]:
            key = part.strip().lower()
            if key in dept_map:
                chosen = dept_map[key]
                break
        if not chosen and d_raw in dept_map:
            chosen = dept_map[d_raw]
        out["department"] = chosen or out["department"]

        # Coerce planned deviation to 'true'/'false'/None
        v = str(out["is_planned_deviation"]).strip().lower()
        if v in ("true","yes","y","planned"):
            out["is_planned_deviation"] = True
        elif v in ("false","no","n","unplanned"):
            out["is_planned_deviation"] = False
        elif v in ("", "none", "null", "n/a", "na"):
            out["is_planned_deviation"] = None

        # Tidy date to plain string; validator will parse later
        out["date_of_occurrence"] = out["date_of_occurrence"]

        return out

    def _is_missing(self, value: Any) -> bool:
        """Consistent 'missing' check for required fields."""
        if value is None:
            return True
        if isinstance(value, bool):
            return False
        s = str(value).strip().lower()
        return s in ("", "null", "none", "n/a", "na")

    def _extract_deviation_info_with_context(self, description: str, rag_context: Dict) -> Dict[str, Any]:
        """Extract structured information using RAG context to improve accuracy"""
        
        # Get examples from RAG
        context_text = rag_context.get('formatted', '') if rag_context else ''
        
        prompt = f"""
    You are extracting information from a deviation description. 
    Use these examples from past reports to understand the terminology and format:

    PAST EXAMPLES:
    {context_text[:500]}

    CURRENT DEVIATION DESCRIPTION: 
    {description}

    Extract and return as JSON:
    {{
        "title": "Brief one sentence title summarizing the deviation (follow title format from examples)",
        "deviation_type": "One of: Equipment, Process, Procedure, Documentation, Personnel, Material, Facility, Formulation, Packaging, Visual Inspection",
        "date_of_occurrence": "Date in MM/DD/YYYY format if mentioned",
        "batch_lot_number": "Batch or lot number if mentioned (look for patterns like LOT#, Batch#, or similar)",
        "quantity_impacted": "Quantity impacted if mentioned (include units)",
        "department": "One of: Manufacturing, QA, QC, Warehouse, R&D, Engineering, Regulatory, Quality",
        "is_planned_deviation": "true or false if mentioned, otherwise null"
    }}

    Use the terminology and formats from the past examples. Only extract what is clearly stated. Use null for missing information.
    """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Extract and normalize JSON
            extracted = self._extract_json_from_text(response.choices[0].message.content)
            normalized = self._coerce_extracted(extracted)
            
            return normalized
            
        except Exception as e:
            print(f"Error extracting information: {e}")
            # Fallback to original method
            return self._extract_deviation_info(description)
    
    def _handle_description(self, description: str) -> str:
        """Handle deviation description and extract information"""
        self.current_session["data"]["description"] = description
        
        # First, do a quick RAG search to understand context
        print("\n🔍 Searching for similar past deviations...")
        initial_search = self.rag_system.search_with_context(
            description[:200],  # Use first 200 chars
            filters=None
        )
        
        # Extract information from description using AI with context
        extracted_info = self._extract_deviation_info_with_context(description, initial_search)
        self.current_session["data"]["extracted"] = extracted_info
        
        # Check what information is missing
        missing_fields = self._check_missing_fields(extracted_info)
        
        if missing_fields:
            self.current_session["stage"] = "missing_info"
            self.current_session["data"]["missing_fields"] = missing_fields
            self.current_session["data"]["current_field_index"] = 0
            
            return self._ask_for_missing_field(missing_fields[0])
        else:
            # All information available, generate report
            return self._generate_final_report()
    
    def _handle_missing_info(self, user_input: str) -> str:
        """Handle missing information collection with validation."""
        missing_fields = self.current_session["data"]["missing_fields"]
        current_index = self.current_session["data"]["current_field_index"]
        current_field = missing_fields[current_index]
        answer = user_input.strip()
        
        # Validate based on field type
        if current_field == "deviation_type":
            # Check if it's a valid deviation type
            valid_lower = [t.lower() for t in self.deviation_types]
            if answer.lower() not in valid_lower:
                return f"⚠️ '{answer}' is not a valid deviation type.\n\n**Valid options:** {', '.join(self.deviation_types)}\n\nPlease enter one of these."
        
        elif current_field == "department":
            # Check if it's a valid department
            valid_lower = [d.lower() for d in self.departments]
            if answer.lower() not in valid_lower:
                return f"⚠️ '{answer}' is not a valid department.\n\n**Valid options:** {', '.join(self.departments)}\n\nPlease enter one of these."
        
        elif current_field == "date_of_occurrence":
            # Validate date format
            from deviation_structures import DeviationDataValidator
            parsed_date = DeviationDataValidator.parse_date(answer)
            if parsed_date is None and answer.lower() not in ['n/a', 'na', 'not applicable']:
                return "⚠️ Please provide the date in MM/DD/YYYY format (e.g., 12/25/2024) or type 'N/A' if not applicable."
        
        elif current_field == "is_planned_deviation":
            # Validate yes/no answer
            if answer.lower() not in ["yes", "y", "no", "n", "true", "false"]:
                return "⚠️ Please answer 'yes' or 'no' - Is this a planned deviation?"
        
        elif current_field == "batch_lot_number":
            # Accept any non-empty input including N/A
            if not answer:
                return "⚠️ Please provide the batch/lot number or type 'N/A' if not applicable."
        
        elif current_field == "quantity_impacted":
            # Accept any non-empty input including N/A
            if not answer:
                return "⚠️ Please provide the quantity impacted or type 'N/A' if not applicable."
        
        # Store the validated answer
        self.current_session["data"]["extracted"][current_field] = answer
        
        # Move to next field or generate report
        current_index += 1
        if current_index < len(missing_fields):
            self.current_session["data"]["current_field_index"] = current_index
            return self._ask_for_missing_field(missing_fields[current_index])
        else:
            return self._generate_final_report()

    
    def _handle_confirmation(self, user_input: str) -> str:
        """Handle report confirmation and generation"""
        if user_input.lower() in ['yes', 'y', 'confirm', 'generate']:
            return self._create_deviation_reports()
        else:
            return "Report generation cancelled. You can start over or make modifications."
    
    def _extract_deviation_info(self, description: str) -> Dict[str, Any]:
        """Extract structured information from deviation description using AI"""
        
        prompt = f"""
    Extract the following information from this deviation description. If information is not clearly stated, leave it as null.

    Deviation Description: {description}

    Extract and return as JSON:
    {{
        "title": "Brief one sentence title summarizing the deviation",
        "deviation_type": "One of: Equipment, Process, Procedure, Documentation, Personnel, Material, Facility, Formulation, Packaging, Visual Inspection",
        "date_of_occurrence": "Date in MM/DD/YYYY format if mentioned",
        "batch_lot_number": "Batch or lot number if mentioned",
        "quantity_impacted": "Quantity impacted if mentioned",
        "department": "One of: Manufacturing, QA, QC, Warehouse, R&D, Engineering, Regulatory, Quality",
        "is_planned_deviation": "true or false if mentioned, otherwise null"
    }}

    Only extract what is clearly stated. Use null for missing information.
    """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Extract JSON from response using the helper method
            extracted = self._extract_json_from_text(response.choices[0].message.content)
            
            # Normalize the extracted data
            normalized = self._coerce_extracted(extracted)
            
            return normalized
            
        except Exception as e:
            print(f"Error extracting information: {e}")
            return {}


    
    def _check_missing_fields(self, extracted_info: Dict[str, Any]) -> list:
        """Check which required fields are missing"""
        required_fields = [
            "title", "deviation_type", "date_of_occurrence",
            "batch_lot_number", "quantity_impacted", "department",
            "is_planned_deviation"
        ]
        
        missing = []
        for field in required_fields:
            value = extracted_info.get(field)
            if self._is_missing(value):
                missing.append(field)
        
        return missing

    
    def _ask_for_missing_field(self, field: str) -> str:
        """Generate question for missing field"""
        questions = {
            "title": "**What is a brief title for this deviation?**\n(e.g., 'Incorrect mixing speed during batch 12345 formulation')",
            "deviation_type": f"**What type of deviation is this?**\nOptions: {', '.join(self.deviation_types)}",
            "date_of_occurrence": "**When did this deviation occur?** (MM/DD/YYYY format)",
            "batch_lot_number": "**What is the batch/lot number affected?**\n(If not applicable, type 'N/A')",
            "quantity_impacted": "**What quantity was impacted?**\n(e.g., '500 tablets', '2 liters', 'entire batch')",
            "department": f"**Which department is involved?**\nOptions: {', '.join(self.departments)}",
            "is_planned_deviation": "**Is this a planned deviation?** (yes/no)"
        }
        
        return questions.get(field, f"Please provide information for: {field}")
    
    def _search_similar_deviations(self, deviation_type: str, description: str, department: str = None) -> Dict[str, Any]:
        """Search RAG for similar past deviation reports to use as language templates"""
        
        # Build targeted search queries for past reports
        search_queries = [
            f"deviation report {deviation_type} {department or ''}",
            f"{deviation_type} deviation root cause CAPA",
            f"batch lot number {deviation_type} investigation",
            description[:200]  # Use first 200 chars of description
        ]
        
        similar_reports = []
        seen_chunks = set()  # Avoid duplicates
        
        for query in search_queries:
            try:
                results = self.rag_system.search_with_context(
                    query, 
                    filters={"document_type": "deviation_report"} if hasattr(self.rag_system, 'search_with_context') else None
                )
                
                # Extract unique chunks
                if results and 'chunks' in results:
                    for chunk in results['chunks'][:3]:  # Top 3 chunks per query
                        chunk_id = chunk.get('id', chunk.get('text', '')[:50])
                        if chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            similar_reports.append(chunk)
                
            except Exception as e:
                print(f"Search error: {e}")
                continue
        
        # Extract language patterns from similar reports
        language_patterns = self._extract_language_patterns(similar_reports)
        
        return {
            'similar_reports': similar_reports[:5],  # Keep top 5 most relevant
            'language_patterns': language_patterns,
            'formatted_context': self._format_deviation_context(similar_reports)
        }

    def _extract_language_patterns(self, reports: List[Dict]) -> Dict[str, List[str]]:
        """Extract common language patterns from past deviation reports"""
        
        patterns = {
            'opening_phrases': [],
            'root_cause_language': [],
            'capa_language': [],
            'impact_language': [],
            'investigation_language': []
        }
        
        for report in reports:
            text = report.get('text', '') if isinstance(report, dict) else str(report)
            
            # Extract opening phrases
            if "deviation was observed" in text.lower():
                start = text.lower().find("deviation was observed")
                patterns['opening_phrases'].append(text[max(0, start-50):start+100])
            
            # Extract root cause language
            if "root cause" in text.lower():
                start = text.lower().find("root cause")
                patterns['root_cause_language'].append(text[start:min(len(text), start+200)])
            
            # Extract CAPA language
            if "corrective action" in text.lower() or "capa" in text.lower():
                start = text.lower().find("corrective action" if "corrective action" in text.lower() else "capa")
                patterns['capa_language'].append(text[start:min(len(text), start+200)])
            
            # Extract impact assessment language
            if "impact" in text.lower() or "assessment" in text.lower():
                start = text.lower().find("impact")
                if start > 0:
                    patterns['impact_language'].append(text[start:min(len(text), start+200)])
            
            # Extract investigation language
            if "investigation" in text.lower():
                start = text.lower().find("investigation")
                patterns['investigation_language'].append(text[start:min(len(text), start+200)])
        
        # Keep unique patterns
        for key in patterns:
            patterns[key] = list(set(patterns[key]))[:3]  # Keep top 3 unique patterns
        
        return patterns

    def _format_deviation_context(self, reports: List[Dict]) -> str:
        """Format similar deviation reports for context"""
        
        if not reports:
            return ""
        
        context_parts = ["Similar Past Deviation Reports:\n"]
        
        for i, report in enumerate(reports[:3], 1):
            text = report.get('text', '') if isinstance(report, dict) else str(report)
            # Extract key sections
            preview = text[:300] if len(text) > 300 else text
            context_parts.append(f"\nExample {i}:\n{preview}...\n")
        
        return "\n".join(context_parts)

    def _generate_final_report(self) -> str:
        """Generate the final deviation report using AI and RAG"""
        
        # Search RAG system for relevant regulatory information
        search_query = f"{self.current_session['data']['extracted'].get('deviation_type', '')} deviation {self.current_session['data']['description'][:100]}"
        regulatory_context = self.rag_system.search_with_context(search_query, filters=None)
        
        # Generate comprehensive report using AI
        ai_report = self._generate_ai_report()
        
        # Show summary and ask for confirmation
        self.current_session["stage"] = "confirmation"
        self.current_session["data"]["ai_report"] = ai_report
        self.current_session["data"]["regulatory_context"] = regulatory_context
        
        extracted = self.current_session["data"]["extracted"]

        full_description = self.current_session["data"].get("description", "No description provided")

        
        return f"""
✅ **Deviation Report Ready**

**Summary of Information:**
• Title: {extracted.get('title', 'N/A')}
• Type: {extracted.get('deviation_type', 'N/A')}
• Date: {extracted.get('date_of_occurrence', 'N/A')}
• Batch/Lot: {extracted.get('batch_lot_number', 'N/A')}
• Quantity: {extracted.get('quantity_impacted', 'N/A')}
• Department: {extracted.get('department', 'N/A')}
• Planned: {extracted.get('is_planned_deviation', 'N/A')}

**Your Full Description:** 
{full_description}

**AI-Generated Report Preview:**
{ai_report[:1500]}...

**Type 'yes' to generate the complete reports (PDF, Word, HTML) or 'no' to cancel.**
"""
    
#     def _generate_ai_report(self) -> str:
#         """Generate comprehensive AI report using the specified prompt"""
        
#         description = self.current_session["data"]["description"]
#         extracted = self.current_session["data"]["extracted"]
        
#         # Get regulatory context
#         search_query = f"{extracted.get('deviation_type', '')} deviation CAPA root cause"
#         regulatory_context = self.rag_system.search_with_context(search_query, filters=None)
#         reg_text = regulatory_context.get('formatted', '')[:1000]
        
#         prompt = f"""
# You're given a brief description of a deviation that occurred during pharma manufacturing.

# Deviation Description: {description}

# Additional Information:
# - Batch/Lot: {extracted.get('batch_lot_number', 'Not specified')}
# - Quantity Impacted: {extracted.get('quantity_impacted', 'Not specified')}
# - Date of Occurrence: {extracted.get('date_of_occurrence', 'Not specified')}
# - Department: {extracted.get('department', 'Not specified')}
# - Deviation Type: {extracted.get('deviation_type', 'Not specified')}

# Relevant Regulatory Context:
# {reg_text}

# Based on the provided information, generate a clear and factual summary of the deviation. 
# Include the event timeline, equipment or process involved, batch or lot number if applicable, and how the issue was identified. 
# Also provide Root Cause Analysis, Impact Assessment, and Corrective and Preventive Action (CAPA) Plan.

# Be factual and concise. Use language consistent with GMP documentation.

# Format your response with clear sections:
# 1. **Deviation Summary**
# 2. **Event Timeline**
# 3. **Root Cause Analysis**
# 4. **Impact Assessment**
# 5. **Corrective and Preventive Action (CAPA) Plan**
# """
        
#         try:
#             response = self.openai_client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=1500,
#                 temperature=0.3
#             )
#             return response.choices[0].message.content
            
#         except Exception as e:
#             return f"Error generating AI report: {str(e)}"

    def _generate_ai_report(self) -> str:
        """Generate comprehensive AI report using past deviations as language templates"""
        
        description = self.current_session["data"]["description"]
        extracted = self.current_session["data"]["extracted"]
        
        # Search for similar past deviations if you have the RAG enhancement
        if hasattr(self, '_search_similar_deviations'):
            similar_deviations = self._search_similar_deviations(
                deviation_type=extracted.get('deviation_type', ''),
                description=description,
                department=extracted.get('department', '')
            )
            past_reports_context = similar_deviations.get('formatted_context', '')[:1000]
        else:
            past_reports_context = ""
        
        # Get regulatory context
        search_query = f"{extracted.get('deviation_type', '')} deviation CAPA root cause GMP requirements"
        regulatory_context = self.rag_system.search_with_context(search_query, filters=None)
        reg_text = regulatory_context.get('formatted', '')[:1000]
        
        prompt = f"""
You are documenting a deviation that has occurred in a pharmaceutical manufacturing facility.

DEVIATION INFORMATION:
Description: {description}
- Batch/Lot: {extracted.get('batch_lot_number', 'Not specified')}
- Quantity Impacted: {extracted.get('quantity_impacted', 'Not specified')}
- Date of Occurrence: {extracted.get('date_of_occurrence', 'Not specified')}
- Department: {extracted.get('department', 'Not specified')}
- Deviation Type: {extracted.get('deviation_type', 'Not specified')}

Relevant Regulatory Context:
{reg_text}

Previous Similar Deviations (for language reference):
{past_reports_context}

Generate a comprehensive deviation report with the following sections:

1. **Deviation Summary**
Write a clear, factual summary of what has occurred. Include the batch/lot number, quantity impacted, when it was identified, and by whom (if mentioned). Use past tense for what happened, present tense for current status. Maximum 3-4 sentences.

2. **Event Timeline**
Create a chronological timeline based ONLY on information provided in the description. Use past tense. Format as bullet points with dates/times if provided. Do not add assumed events.

3. **Root Cause Analysis**
Write in PRESENT TENSE. Be CONCISE and SPECIFIC to this deviation.
Based on the description, identify 3-4 most likely areas for investigation using Ishikawa categories:

Start with: "Root cause investigation focuses on the following areas based on the deviation details:"

Then list ONLY the relevant categories (pick only those that apply):
- If equipment is mentioned: "**Equipment**: [Specific equipment mentioned] requires review of [specific relevant checks based on the problem described]"
- If process issue: "**Process**: [Specific process step mentioned] needs evaluation of [specific parameters mentioned or implied by the deviation]"
- If human factors indicated: "**Personnel**: Review [specific action mentioned] for compliance with [relevant SOP implied by the deviation]"
- If material issue: "**Material**: [Specific material mentioned] requires verification of [specific concern based on deviation]"

End with ONE sentence: "Investigation will use [5-Why analysis/Fault Tree/Fishbone diagram] to identify the root cause."

Keep this section under 200 words. Be specific to what actually happened, not generic.

4. **Impact Assessment**
Write in PRESENT TENSE. Be CONCISE and SPECIFIC to this deviation.

Structure as 4-5 bullet points covering ONLY relevant impacts:
- **Product Quality**: [Specific quality concern based on the deviation - e.g., "Mixing speed deviation may affect content uniformity"]
- **Batch Disposition**: [Specific statement about the affected batch - e.g., "The batch requires additional testing for X"]
- **Risk Level**: Based on the specific deviation described, classify as Critical/Major/Minor with ONE sentence rationale
- **Other Batches**: ONLY if relevant - identify specific risk to other batches based on the deviation nature

Do not include generic statements. Each point must relate directly to the specific deviation described.
Keep this section under 200 words.

5. **Corrective and Preventive Action (CAPA) Plan**
Be SPECIFIC and ACTIONABLE based on the deviation described.

**Immediate Actions** (24-48 hours):
- List 2-3 specific actions directly addressing the deviation (e.g., "Quarantine batch X", "Stop equipment Y", "Sample for testing Z")

**Corrective Actions** (address this event):
- List 2-3 specific actions based on the deviation (e.g., "Reprocess batch using correct parameters", "Recalibrate equipment X", "Retrain operator on specific procedure")

**Preventive Actions** (prevent recurrence):
- List 2-3 specific long-term actions that directly prevent this type of deviation (e.g., "Add automated alarm for parameter X", "Revise SOP to include double verification", "Implement additional in-process control at step Y")

Each action must be directly related to the specific deviation. No generic statements.
Keep this section under 150 words.

CRITICAL INSTRUCTIONS:
- Be concise - entire report should be under 800 words
- Every statement must relate directly to the specific deviation described
- Do not include generic GMP statements that could apply to any deviation
- Do not invent information not provided in the description
- Use specific batch numbers, equipment names, parameters mentioned in the description
- If information is not provided, do not make assumptions
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            generated_report = response.choices[0].message.content
            
            # Store the similar deviations for reference if available
            if hasattr(self, '_search_similar_deviations'):
                self.current_session["data"]["similar_deviations"] = similar_deviations
            
            return generated_report
            
        except Exception as e:
            return f"Error generating AI report: {str(e)}"
    
    def _create_deviation_reports(self) -> str:
        """Create the actual deviation report files"""
        
        try:
            # Build deviation report object
            report = self._build_deviation_report_object()
            
            # Generate reports using your existing report generator
            generator = SimplifiedReportGenerator()
            results = generator.generate_all_formats(report)
            
            return f"""
🎉 **Reports Generated Successfully!**

Your deviation report has been created in multiple formats:

📄 **HTML Report:** {results.get('html', 'N/A')}
📄 **PDF Report:** {results.get('pdf', 'N/A')}
📄 **Word Report:** {results.get('word', 'N/A')}

The reports include:
✓ Complete deviation information
✓ Regulatory context and citations
✓ AI-generated analysis with Root Cause, Impact Assessment, and CAPA
✓ Professional GMP-compliant formatting

**Thank you for using the Deviation Reporting System!**

*Session completed. You can start a new session for another report.*
"""
            
        except Exception as e:
            return f"❌ Error generating reports: {str(e)}"
    
    def _build_deviation_report_object(self) -> DeviationReport:
        """Build the structured deviation report object"""
        
        report = DeviationReport()
        data = self.current_session["data"]
        extracted = data["extracted"]
        
        # Header information
        report.header.record_id = self.current_session["id"]
        report.header.initiator = data["initiator"]
        report.header.report_generated = datetime.now()
        
        # Deviation information
        report.deviation_info.title = extracted.get("title", "")
        report.deviation_info.description = data["description"]
        
        # Convert string to enum
        dev_type_str = extracted.get("deviation_type", "")
        report.deviation_info.deviation_type = DeviationDataValidator.validate_deviation_type(dev_type_str)
        
        # Parse date
        date_str = extracted.get("date_of_occurrence", "")
        report.deviation_info.date_of_occurrence = DeviationDataValidator.parse_date(date_str)
        
        # Other fields
        report.deviation_info.batch_lot_number = extracted.get("batch_lot_number", "")
        report.deviation_info.quantity_impacted = extracted.get("quantity_impacted", "")
        
        # Parse departments
        dept_str = extracted.get("department", "")
        report.deviation_info.department = DeviationDataValidator.parse_departments(dept_str)
        
        # Planned deviation
        planned_str = extracted.get("is_planned_deviation", "false")
        report.deviation_info.is_planned_deviation = str(planned_str).lower() in ['true', 'yes', 'y']
        
        # Set default values
        report.deviation_info.priority = Priority.MAJOR
        report.deviation_info.status = Status.IN_PROGRESS
        
        # AI-generated content
        report.ai_generated.enhanced_description = data.get("ai_report", "")
        
        # Basic investigation info
        report.investigation.assignee = data["initiator"]
        report.investigation.findings = "See AI-generated analysis for detailed findings."
        report.investigation.conclusion = "Analysis completed using AI-enhanced regulatory guidance."
        
        return report


def run_simplified_deviation_system():
    """Main function to run the simplified system"""
    
    print("🚀 Initializing Simplified Pharmaceutical Deviation System...")
    
    # Check environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        return
    
    # Initialize Vertex AI
    try:
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        print("✅ Vertex AI initialized")
    except Exception as e:
        print(f"❌ Error initializing Vertex AI: {e}")
        return
    
    # Connect to ChromaDB
    try:
        client = PersistentClient(path="./chroma")
        collection = client.get_collection(name="char-split-collection")
        print(f"✅ Connected to ChromaDB collection")
        print(f"📚 Documents in collection: {collection.count()}")
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        return
    
    # Initialize the simplified agent
    agent = SimplifiedDeviationAgent(
        openai_api_key=openai_api_key,
        collection=collection,
        embed_func=generate_query_embedding,
        vertex_model=generative_model
    )
    
    print("\n" + "="*60)
    print("SIMPLIFIED PHARMACEUTICAL DEVIATION REPORTING SYSTEM")
    print("="*60 + "\n")
    
    # Main interaction loop
    while True:
        try:
            # Start new session
            greeting = agent.start_new_session()
            print(greeting)
            
            # Process user inputs
            while True:
                user_input = input("\n👤 Your response: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    print("\n👋 Thank you for using the system. Goodbye!")
                    return
                
                if not user_input:
                    print("Please provide a response or type 'quit' to exit.")
                    continue
                
                print("\n🤖 Processing...")
                response = agent.process_input(user_input)
                print(f"\n{response}")
                
                # Check if session is complete
                if "Session completed" in response or "Thank you for using" in response:
                    break
            
            # Ask if they want to create another report
            another = input("\n🔄 Would you like to create another report? (yes/no): ")
            if another.lower() not in ['yes', 'y']:
                print("\n👋 Thank you for using the system. Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    run_simplified_deviation_system()