# run_integrated_agent.py - Main entry point for the integrated system

import os
import sys
from datetime import datetime
from pathlib import Path


# Add imports from code
from cli import (
    vertexai, 
    GCP_PROJECT, 
    GCP_LOCATION,
    embedding_model,
    generative_model,
    generate_query_embedding,
    PersistentClient
)

# Import the data structures 
from deviation_structures import (
    DeviationType, Priority, Status, Department,
    DeviationReport, DeviationInfo, HeaderInfo,
    Investigation, RiskAssessment, RootCauseAnalysis,
    AIGeneratedContent, DeviationDataValidator
)

from agent_tools import (
    IntegratedRAGSystem,
    execute_function_calls,
    get_deviation_analysis_tool
)

# Import report generator
from report_generator import EnhancedReportGenerator

# Import smolagents components
from smolagents import tool, ToolCallingAgent, LiteLLMModel
import openai

class IntegratedDeviationInterviewAgent:
    """
    Main agent that combines RAG search with intelligent interview flow
    """
    
    def __init__(self, openai_api_key: str, collection, embed_func, vertex_model):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.collection = collection
        self.embed_func = embed_func
        self.vertex_model = vertex_model
        
        # Initialize RAG system
        self.rag_system = IntegratedRAGSystem(collection, embed_func, vertex_model)
        
        # Session management
        self.sessions = {}
        
        # Interview flow configuration
        self.interview_stages = [
            "greeting", "basic_info", "description", 
            "regulatory_search", "analysis", "corrective_actions", 
            "review", "complete"
        ]
    
    def start_session(self, session_id: str = None) -> tuple[str, str]:
        """Start a new deviation interview session"""
        if not session_id:
            session_id = f"DEV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.sessions[session_id] = {
            "id": session_id,
            "start_time": datetime.now(),
            "stage": "greeting",
            "data": {
                "basic": {},
                "description": "",
                "regulatory_context": {},
                "analysis": {},
                "actions": {},
                "report": DeviationReport()
            }
        }
        
        greeting = """
        🏥 **Welcome to the Integrated Pharmaceutical Deviation Reporting System**

        I'll guide you through creating a comprehensive deviation report that:
        ✅ Complies with regulatory requirements (21 CFR Part 211, ICH Q7)
        ✅ Includes AI-enhanced analysis and risk assessment  
        ✅ Provides evidence-based corrective actions
        ✅ Generates professional reports in multiple formats

        Let's start by collecting some basic information.

        **Please provide your name (person initiating this report):**
        """
        
        return session_id, greeting
    
    def process_input(self, session_id: str, user_input: str) -> str:
        """Process user input based on current session stage"""
        
        if session_id not in self.sessions:
            return "Session not found. Please start a new session."
        
        session = self.sessions[session_id]
        stage = session["stage"]
        
        # Route to appropriate handler
        handlers = {
            "greeting": self._handle_greeting,
            "basic_info": self._handle_basic_info,
            "description": self._handle_description,
            "user_immediate_actions": self._handle_user_immediate_actions,
            "user_risk_assessment": self._handle_user_risk_assessment,
            "analysis": self._handle_analysis,
            "corrective_actions": self._handle_corrective_actions,
            "review": self._handle_review,
            "complete": self._handle_complete
        }
        
        handler = handlers.get(stage, self._handle_unknown)
        return handler(session, user_input)
    
    def _handle_greeting(self, session: dict, user_input: str) -> str:
        """Handle initial greeting and name collection"""
        session["data"]["basic"]["initiator"] = user_input
        session["stage"] = "basic_info"
        session["basic_info_step"] = 0
        
        return f"""
        Thank you, {user_input}!

        Now I need to collect some key information about the deviation.

        **What is the title/summary of this deviation?**
        (Be specific - e.g., "Incorrect mixing speed during batch 12345 formulation")
        """
    
    def _handle_basic_info(self, session: dict, user_input: str) -> str:
        """Collect basic deviation information step by step"""
        step = session.get("basic_info_step", 0)
        basic_data = session["data"]["basic"]
        
        questions = [
            ("type", "What type of deviation is this?\n(Options: Equipment, Process, Procedure, Documentation, Personnel, Material, Facility, Formulation)"),
            ("date", "When did this deviation occur? (MM/DD/YYYY format)"),
            ("batch", "What is the batch/lot number affected?"),
            ("quantity", "What quantity was impacted?"),
            ("department", "Which department(s) are involved? (e.g., Manufacturing, QA, QC)"),
            ("planned", "Is this a planned deviation? (yes/no)"),
            ("dummy", None)  # Last step - dummy field since we already have all data
        ]
        
        # Store current answer
        if step == 0:
            # First step is the title (asked in greeting)
            basic_data["title"] = user_input
        elif step > 0:
            # For subsequent steps, store based on previous question
            field_name = questions[step-1][0]
            basic_data[field_name] = user_input
        
        # Move to next question or stage
        if step < len(questions) - 1:
            session["basic_info_step"] = step + 1
            return questions[step][1]
        else:
            # All basic info collected, move to description
            session["stage"] = "description"
            return self._show_basic_summary(basic_data) + """
        Now, please provide a detailed description of the deviation:
        **What happened? Include all relevant details, observations, and circumstances.**
        """
    
    def _show_basic_summary(self, basic_data: dict) -> str:
        """Show summary of collected basic information"""
        return f"""
        ✅ **Basic Information Collected:**
        • Initiator: {basic_data.get('initiator', 'N/A')}
        • Title: {basic_data.get('title', 'N/A')}
        • Type: {basic_data.get('type', 'N/A')}
        • Date: {basic_data.get('date', 'N/A')}
        • Batch/Lot: {basic_data.get('batch', 'N/A')}
        • Quantity: {basic_data.get('quantity', 'N/A')}
        • Department(s): {basic_data.get('department', 'N/A')}
        • Planned: {basic_data.get('planned', 'N/A')}
        """
    def _handle_user_immediate_actions(self, session: dict, user_input: str) -> str:
        """Handle user's immediate actions input"""
        # Store user's original response
        session["data"]["user_immediate_actions"] = user_input
        session["stage"] = "user_risk_assessment"
        
        return """
        **Thank you. Your immediate actions have been recorded.**

        Now, please provide your initial risk assessment:
        **What is your assessment of the risk this deviation poses to product quality and patient safety?**
        """

    def _handle_user_risk_assessment(self, session: dict, user_input: str) -> str:
        """Handle user's risk assessment input"""
        # Store user's original response
        session["data"]["user_risk_assessment"] = user_input
        session["stage"] = "analysis"
        
        # Now generate AI enhancements
        return self._process_with_ai_enhancements(session)

    def _handle_analysis(self, session: dict, user_input: str) -> str:
        """Handle analysis confirmation and move to corrective actions"""
        if user_input.lower() in ['yes', 'y', 'correct', 'good']:
            # User accepts the analysis, move to corrective actions
            session["stage"] = "corrective_actions"
            return self._generate_corrective_actions(session)
        else:
            # User wants to add more details
            session["data"]["description"] += f"\n\nAdditional details: {user_input}"
            # Re-analyze with new information
            enhanced_analysis = self._generate_enhanced_analysis(session)
            session["data"]["analysis"]["enhanced_description"] = enhanced_analysis
            session["stage"] = "corrective_actions"
            return f"""
            **Thank you for the additional information. I've updated the analysis.**

            🤖 **Updated Analysis:**
            {enhanced_analysis}

            Moving on to corrective actions...

            {self._generate_corrective_actions(session)}
            """
        
    def _handle_description(self, session: dict, description: str) -> str:
        """Handle deviation description and trigger regulatory search"""
        session["data"]["description"] = description
        session["stage"] = "user_immediate_actions"
        
        # Search for relevant regulations
        print("🔍 Searching regulatory database...")
        
        # Build search query based on deviation type and description
        dev_type = session["data"]["basic"].get("type", "")
        search_query = f"{dev_type} deviation {description[:100]}"
        
        # Perform RAG search
        # ChromaDB doesn't support $contains, so we'll search without filters
        # or use exact matches if we know the book names
        search_results = self.rag_system.search_with_context(
            search_query,
            filters=None  # Remove the filter for now
        )
        print(f"DEBUG: search_results keys: {search_results.keys() if search_results else 'None'}")
        
        session["data"]["regulatory_context"] = search_results
        
        # Generate AI-enhanced analysis
        enhanced_analysis = self._generate_enhanced_analysis(session)
        session["data"]["analysis"]["enhanced_description"] = enhanced_analysis
        
        immediate_actions = self._generate_immediate_actions(session)
        session["data"]["analysis"]["immediate_actions"] = immediate_actions

        risk_assessment = self._generate_risk_assessment(session)
        session["data"]["analysis"]["risk_assessment"] = risk_assessment

        return """
        **Thank you for the detailed description.**

        Now I need to collect some additional information:

        **What immediate actions were taken when this deviation was discovered?**
        (Please describe any actions taken to contain the issue, preserve evidence, or prevent further impact. If no immediate actions were taken, please type "None" or explain why)
        """
    
    def _generate_enhanced_analysis(self, session: dict) -> str:
        """Generate enhanced analysis using AI and regulatory context"""
        description = session["data"]["description"]
        regulatory_context = session["data"]["regulatory_context"].get("formatted", "No regulatory context available")[:1000]
        
        prompt = f"""
        You are a pharmaceutical quality expert analyzing a deviation.

        Deviation: {description}

        Relevant Regulations:
        {regulatory_context}

        Provide a concise enhanced analysis including:
        1. Technical assessment with regulatory context
        2. Potential quality impact
        3. Regulatory compliance implications (very short)
        4. Key investigation points

        Be precise and factual. 
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis generation error: {str(e)}"
    
    def _handle_regulatory_search(self, session: dict, user_input: str) -> str:
        """Handle regulatory search confirmation"""
        if user_input.lower() in ['yes', 'y']:
            session["stage"] = "corrective_actions"
            return self._generate_corrective_actions(session)
        else:
            # User wants to add more details
            session["data"]["description"] += f"\n\nAdditional details: {user_input}"
            # Re-analyze with new information
            enhanced_analysis = self._generate_enhanced_analysis(session)
            session["data"]["analysis"]["enhanced_description"] = enhanced_analysis
            session["stage"] = "corrective_actions"
            return self._generate_corrective_actions(session)
        


    def _generate_immediate_actions(self, session: dict) -> str:
        """Generate immediate actions based on deviation"""
        description = session["data"]["description"]
        dev_type = session["data"]["basic"].get("type", "")
        batch = session["data"]["basic"].get("batch", "")
        
        prompt = f"""
        Based on this pharmaceutical deviation, suggest immediate actions (within 24 hours):

        Deviation: {description}
        Type: {dev_type}
        Batch: {batch}

        Provide 3-5 immediate actions that must be taken NOW to:
        1. Contain the issue
        2. Prevent further impact
        3. Preserve evidence
        4. Ensure product quality

        Format as a numbered list. Be specific and actionable.
        """ 
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return "1. Quarantine affected batch\n2. Document all observations\n3. Notify QA management\n4. Secure all related records"
    


    def _generate_risk_assessment(self, session: dict) -> str:
        """Generate risk assessment for the deviation"""
        description = session["data"]["description"]
        dev_type = session["data"]["basic"].get("type", "")
        quantity = session["data"]["basic"].get("quantity", "")
        
        prompt = f"""
        Perform a risk assessment for this pharmaceutical deviation:

        Deviation: {description}
        Type: {dev_type}
        Quantity Impacted: {quantity}

        Provide a brief risk assessment including:
        1. Product Quality Risk: (High/Medium/Low) with justification
        2. Patient Safety Risk: (High/Medium/Low) with justification  
        3. Regulatory Risk: (High/Medium/Low) with justification
        4. Overall Risk Level: (High/Medium/Low)
        5. Key risk factors (2-3 bullet points)

        Keep it concise and focused on actual risks.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return "Risk assessment error: " + str(e)
    

    def _generate_corrective_actions(self, session: dict) -> str:
        """Generate corrective actions based on deviation and regulations"""
        
        # Search for CAPA requirements
        capa_search = self.rag_system.search_with_context(
            f"corrective action preventive action {session['data']['basic'].get('type', '')} deviation"
        )
        
        # Generate AI suggestions with regulatory context
        description = session["data"]["description"]
        regulatory_reqs = capa_search['requirements']
        
        prompt = f"""
        Based on this pharmaceutical deviation and regulatory requirements, suggest corrective actions:

        Deviation: {description}

        Regulatory Requirements:
        {regulatory_reqs}

        Provide:
        1. Immediate corrective actions (3-5)
        2. Preventive actions (2-3) 
        3. Effectiveness verification methods
        4. Timeline recommendations

        Be specific and cite regulations where applicable.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            actions = response.choices[0].message.content
            session["data"]["actions"]["suggested"] = actions
        except Exception as e:
            actions = "Error generating actions: " + str(e)
        
        return f"""
        📋 **Regulatory CAPA Requirements:**
        {self._format_requirements(regulatory_reqs)}

        🔧 **Suggested Corrective & Preventive Actions:**
        {actions}

        **Do these actions look appropriate for your deviation?**
        (Type 'yes' to accept, or suggest modifications)
        """
    
    def _format_requirements(self, requirements: dict) -> str:
        """Format regulatory requirements for display"""
        formatted = []
        for category, items in requirements.items():
            if items:
                formatted.append(f"\n**{category.replace('_', ' ').title()}:**")
                for item in items[:3]:  # Show first 3
                    formatted.append(f"• {item}")
        return "\n".join(formatted) if formatted else "No specific requirements found."
    
    def _handle_corrective_actions(self, session: dict, user_input: str) -> str:
        """Handle corrective actions feedback"""
        if user_input.lower() in ['yes', 'y']:
            session["stage"] = "review"
            return self._prepare_final_review(session)
        else:
            # User wants modifications
            session["data"]["actions"]["user_feedback"] = user_input
            session["stage"] = "review"
            return self._prepare_final_review(session)
    
    def _process_with_ai_enhancements(self, session: dict) -> str:
        """Process all AI enhancements after collecting user inputs"""
        
        # Search for regulations
        print("🔍 Searching regulatory database...")
        dev_type = session["data"]["basic"].get("type", "")
        search_query = f"{dev_type} deviation {session['data']['description'][:100]}"
        
        search_results = self.rag_system.search_with_context(search_query, filters=None)
        session["data"]["regulatory_context"] = search_results
        
        # Generate AI enhancements
        enhanced_analysis = self._generate_enhanced_analysis(session)
        session["data"]["analysis"]["enhanced_description"] = enhanced_analysis
        
        immediate_actions = self._generate_immediate_actions(session)
        session["data"]["analysis"]["immediate_actions"] = immediate_actions
        
        risk_assessment = self._generate_risk_assessment(session)
        session["data"]["analysis"]["risk_assessment"] = risk_assessment
        
        return f"""
    🔍 **Regulatory Context Found:**
    {search_results.get('formatted', 'No regulatory context found')[:500]}...

    📝 **Your Original Description:**
    {session["data"]["description"][:200]}...

    🤖 **AI-Enhanced Analysis:**
    {enhanced_analysis}

    ⚡ **Your Immediate Actions:**
    {session["data"].get("user_immediate_actions", "None reported")}

    🤖 **AI Assessment of Immediate Actions:**
    {immediate_actions}

    ⚠️ **Your Risk Assessment:**
    {session["data"].get("user_risk_assessment", "None provided")}

    🤖 **AI Risk Assessment:**
    {risk_assessment}

    **Is this analysis accurate? Would you like to add any additional details?**
    (Type 'yes' to continue or provide additional information)
    """
    
    def _prepare_final_review(self, session: dict) -> str:
        """Prepare final review before report generation"""
        
        # Build complete deviation report object
        report = self._build_deviation_report(session)
        session["data"]["report"] = report
        
        return f"""
        ✅ **Deviation Report Ready for Generation**

        **Summary:**
        • Record ID: {report.header.record_id}
        • Title: {report.deviation_info.title}
        • Type: {report.deviation_info.deviation_type.value if report.deviation_info.deviation_type else 'N/A'}
        • Priority: {report.deviation_info.priority.value}
        • Regulatory context: ✓ Included
        • AI analysis: ✓ Complete
        • Corrective actions: ✓ Defined

        **Would you like to generate the formal reports now?**
        (Type 'yes' to generate PDF, Word, and HTML reports)
        """
    
    def _build_deviation_report(self, session: dict) -> DeviationReport:
        """Build complete deviation report from session data"""
        report = DeviationReport()
        basic = session["data"]["basic"]
        
        # Header
        report.header.record_id = session["id"]
        report.header.initiator = basic.get("initiator", "")
        
        # Deviation info
        report.deviation_info.title = basic.get("title", "")
        report.deviation_info.description = session["data"]["description"]
        report.deviation_info.deviation_type = DeviationDataValidator.validate_deviation_type(
            basic.get("type", "")
        )
        report.deviation_info.date_of_occurrence = DeviationDataValidator.parse_date(
            basic.get("date", "")
        )
        report.deviation_info.batch_lot_number = basic.get("batch", "")
        report.deviation_info.quantity_impacted = basic.get("quantity", "")
        report.deviation_info.department = DeviationDataValidator.parse_departments(
            basic.get("department", "")
        )
        report.deviation_info.is_planned_deviation = basic.get("planned", "").lower() in ['yes', 'y']
        
        # AI-generated content
        report.ai_generated.enhanced_description = session["data"]["analysis"].get(
            "enhanced_description", ""
        )

        # Store original user responses for display in report
        # These will be shown in grey boxes in the report
        # Store original user responses for display in report
        report.original_immediate_actions = session["data"].get("user_immediate_actions", "None provided")
        report.original_risk_assessment = session["data"].get("user_risk_assessment", "None provided")

        # Store AI assessments
        report.ai_generated.immediate_actions_summary = session["data"]["analysis"].get("immediate_actions", "")
        report.ai_generated.risk_assessment = session["data"]["analysis"].get("risk_assessment", "")
        
        # Collect ALL immediate actions (don't overwrite)
        all_immediate_actions = []
        
        # First, get immediate actions from the analysis phase
        if "immediate_actions" in session["data"]["analysis"]:
            actions_text = session["data"]["analysis"]["immediate_actions"]
            import re
            action_items = re.findall(r'\d+\.\s*([^\n]+)', actions_text)
            all_immediate_actions.extend(action_items)
        
        # Then add any from corrective actions (if they're different)
        actions_text = session["data"]["actions"].get("suggested", "")
        if actions_text and "Immediate" in actions_text:
            # Extract only immediate actions section
            immediate_section = actions_text.split("Immediate")[1].split("\n\n")[0] if "Immediate" in actions_text else ""
            more_actions = re.findall(r'\d+\.\s*([^\n]+)', immediate_section)
            # Add only if not duplicates
            for action in more_actions:
                if action not in all_immediate_actions:
                    all_immediate_actions.append(action)
        
        report.immediate_actions = all_immediate_actions[:5]  # Keep top 5
        
        # Add risk assessment with proper structure
        if "risk_assessment" in session["data"]["analysis"]:
            risk_text = session["data"]["analysis"]["risk_assessment"]
            report.risk_assessment.conclusion = risk_text
            report.risk_assessment.risk_description = "Based on deviation analysis"
            
            # Try to extract risk levels from the text
            if "Overall Risk Level:" in risk_text:
                overall_risk = risk_text.split("Overall Risk Level:")[1].split("\n")[0].strip()
                report.risk_assessment.risk_level = overall_risk
                
        return report
    
    def _handle_review(self, session: dict, user_input: str) -> str:
        """Handle final review and report generation"""
        if user_input.lower() in ['yes', 'y']:
            session["stage"] = "complete"
            
            # Generate reports
            report = session["data"]["report"]
            generator = EnhancedReportGenerator()
            
            try:
                results = generator.generate_all_formats(report)
                
                return f"""
                🎉 **Reports Generated Successfully!**

                Your deviation report has been created in multiple formats:

                📄 **HTML Report:** {results.get('html', 'N/A')}
                📄 **PDF Report:** {results.get('pdf', 'N/A')}
                📄 **Word Report:** {results.get('word', 'N/A')}

                The reports include:
                ✓ All collected information
                ✓ Regulatory context and citations
                ✓ AI-enhanced analysis
                ✓ Risk assessment
                ✓ Corrective and preventive actions
                ✓ Professional formatting

                **Thank you for using the Integrated Deviation Reporting System!**
                """
            except Exception as e:
                return f"Error generating reports: {str(e)}"
        else:
            return "Report generation cancelled. Type 'yes' to generate reports."
    
    def _handle_complete(self, session: dict, user_input: str) -> str:
        """Handle completed session"""
        return "This session is complete. Start a new session to create another report."
    
    def _handle_unknown(self, session: dict, user_input: str) -> str:
        """Handle unknown stage"""
        return "Unknown session state. Please start a new session."

# Main execution function
def run_integrated_deviation_system():
    """Main function to run the integrated system"""
    
    print("🚀 Initializing Integrated Pharmaceutical Deviation System...")
    
    # Check environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set: os.environ['OPENAI_API_KEY'] = 'your-api-key'")
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
        print("Please ensure you have run the RAG pipeline (chunk, embed, load)")
        return
    
    # Initialize the integrated agent
    agent = IntegratedDeviationInterviewAgent(
        openai_api_key=openai_api_key,
        collection=collection,
        embed_func=generate_query_embedding,
        vertex_model=generative_model
    )
    
    print("\n" + "="*60)
    print("PHARMACEUTICAL DEVIATION REPORTING SYSTEM")
    print("Powered by RAG + AI Agent Technology")
    print("="*60 + "\n")
    
    # Start interactive session
    session_id, greeting = agent.start_session()
    print(greeting)
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\n👤 Your response: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'stop']:
                print("\n👋 Thank you for using the system. Goodbye!")
                break
            
            if not user_input:
                print("Please provide a response or type 'quit' to exit.")
                continue
            
            print("\n🤖 Processing...")
            response = agent.process_input(session_id, user_input)
            print(f"\n{response}")
            
            # Check if session is complete
            if "session is complete" in response.lower():
                again = input("\n🔄 Would you like to create another report? (yes/no): ")
                if again.lower() in ['yes', 'y']:
                    session_id, greeting = agent.start_session()
                    print(f"\n{greeting}")
                else:
                    print("\n👋 Thank you for using the system. Goodbye!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Interview interrupted. Saving progress...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")
    
    return agent

if __name__ == "__main__":
    # Run the integrated system
    agent = run_integrated_deviation_system()