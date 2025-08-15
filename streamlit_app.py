import sys

# Enhanced SQLite fix for Streamlit Cloud
def ensure_sqlite_compatibility():
    """Ensure SQLite compatibility for ChromaDB"""
    try:
        # Check if we already have the right sqlite3
        import sqlite3
        version = sqlite3.sqlite_version_info
        if version >= (3, 35, 0):
            print(f"✅ SQLite {sqlite3.sqlite_version} is compatible")
            return True
    except:
        pass
    
    # Try to use pysqlite3-binary
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ Replaced sqlite3 with pysqlite3-binary")
        return True
    except ImportError:
        print("⚠ pysqlite3-binary not available")
        return False

# Apply the fix immediately
ensure_sqlite_compatibility()

import os
from pathlib import Path

# Add modules to path FIRST
sys.path.append('./modules')
sys.path.append('.')

import streamlit as st

# Modern CSS styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern color scheme - Light blue-gray theme */
    :root {
        --primary-blue-gray: #64748b;
        --light-blue-gray: #94a3b8;
        --very-light-blue-gray: #e2e8f0;
        --background-gray: #f8fafc;
        --text-dark: #1e293b;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --accent-blue: #3b82f6;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-left: 4px solid var(--success-color) !important;
        color: #065f46 !important;
        border-radius: 8px !important;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: rgba(148, 163, 184, 0.1) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-left: 4px solid var(--primary-blue-gray) !important;
        color: var(--text-dark) !important;
        border-radius: 8px !important;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.2) !important;
        border-left: 4px solid var(--warning-color) !important;
        color: #92400e !important;
        border-radius: 8px !important;
    }
    
    /* Error message styling - now blue-gray instead of red */
    .stException, [data-testid="stException"] {
        background-color: rgba(100, 116, 139, 0.1) !important;
        border: 1px solid rgba(100, 116, 139, 0.2) !important;
        border-left: 4px solid var(--primary-blue-gray) !important;
        color: var(--text-dark) !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background-color: rgba(100, 116, 139, 0.1) !important;
        border: 1px solid rgba(100, 116, 139, 0.2) !important;
        border-left: 4px solid var(--primary-blue-gray) !important;
        color: var(--text-dark) !important;
        border-radius: 8px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue-gray), var(--light-blue-gray)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(135deg, #475569, var(--primary-blue-gray)) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-blue), #2563eb) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 2px solid var(--primary-blue-gray) !important;
        color: var(--primary-blue-gray) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--very-light-blue-gray) !important;
        border-color: var(--light-blue-gray) !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success-color), #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.75rem !important;
        min-height: auto !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669, #047857) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1544g2n {
        background-color: var(--background-gray) !important;
        border-right: 1px solid var(--very-light-blue-gray) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: transparent !important;
        border: 1px solid var(--very-light-blue-gray) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Chat input styling */
    .stChatInput > div > div > input {
        border: 2px solid var(--very-light-blue-gray) !important;
        border-radius: 25px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.9rem !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: var(--primary-blue-gray) !important;
        box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.1) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--very-light-blue-gray) !important;
        border-radius: 8px !important;
        border: 1px solid var(--light-blue-gray) !important;
    }
    
    /* Divider styling */
    hr {
        border-color: var(--very-light-blue-gray) !important;
        opacity: 0.5 !important;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: var(--primary-blue-gray) !important;
    }
    
    /* Container styling */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Caption styling */
    .caption {
        color: var(--light-blue-gray) !important;
        font-size: 0.8rem !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: var(--very-light-blue-gray) !important;
        border: 1px solid var(--light-blue-gray) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

def is_cloud_environment():
    """Check if running on Streamlit Cloud"""
    try:
        # Check if we have secrets available (cloud environment)
        if hasattr(st, 'secrets') and len(st.secrets) > 0:
            return True
        return False
    except:
        return False

def setup_environment():
    """Setup environment for both local and cloud deployment"""
    
    if is_cloud_environment():
        st.info("🌐 Running on Streamlit Cloud")
        
        # Use Streamlit secrets for cloud deployment
        try:
            os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
            os.environ['GCP_PROJECT'] = st.secrets.get('GCP_PROJECT', 'rag-test-467013')
            
            # Handle GCP credentials for cloud
            if 'GCP_SERVICE_ACCOUNT_JSON' in st.secrets:
                import json
                import tempfile
                
                # Create temporary credentials file
                gcp_creds = st.secrets['GCP_SERVICE_ACCOUNT_JSON']
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    if isinstance(gcp_creds, str):
                        f.write(gcp_creds)
                    else:
                        json.dump(gcp_creds, f)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
            
            return True
                
        except Exception as e:
            st.error(f"Environment setup failed: {e}")
            return False
    else:
        st.info("💻 Running locally")
        
        # Use local environment variables
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            st.error("OPENAI_API_KEY environment variable not set")
            st.info("Set it with: `export OPENAI_API_KEY='your-key-here'`")
            return False
        
        gcp_project = os.getenv('GCP_PROJECT')
        if not gcp_project:
            st.warning("GCP_PROJECT not set, using default")
            os.environ['GCP_PROJECT'] = 'rag-test-467013'
        
        # Check GCP credentials file
        gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not gcp_creds or not os.path.exists(gcp_creds):
            st.error("GOOGLE_APPLICATION_CREDENTIALS not set or file not found")
            st.info("Set it with: `export GOOGLE_APPLICATION_CREDENTIALS='../secrets/cl-rag-docs.json'`")
            return False
        
        return True

# Import your modules after environment setup
if setup_environment():
    try:
        from modules.cli import (
            GCP_PROJECT, 
            GCP_LOCATION,
            embedding_model,
            generative_model,
            generate_query_embedding,
            PersistentClient  # Import from modules.cli instead of just 'cli'
        )
        
        from modules.run_integrated_agent import SimplifiedDeviationAgent
        from modules.deviation_structures import DeviationType, Priority, Status, Department
        
        # For cloud deployment, we'll use the original ChromaDB setup
        if is_cloud_environment():
            from data.init_chroma import get_chroma_setup
        
        MODULES_LOADED = True
        
    except ImportError as e:
        st.error(f"Module import failed: {e}")
        st.info("Make sure all modules are in the 'modules' directory or available in your environment")
        MODULES_LOADED = False
else:
    MODULES_LOADED = False

from typing import Dict, Any
from datetime import datetime

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    if 'chroma_collection' not in st.session_state:
        st.session_state.chroma_collection = None
    if 'initialization_error' not in st.session_state:
        st.session_state.initialization_error = None

def initialize_agent():
    """Initialize your deviation report agent"""
    if not MODULES_LOADED:
        st.error("Modules not loaded - cannot initialize agent")
        return False
        
    try:
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OPENAI_API_KEY not found")
            return False
        
        with st.spinner("Initializing Vertex AI..."):
            # Initialize Vertex AI
            import vertexai
            vertexai.init(project=os.getenv('GCP_PROJECT', 'rag-test-467013'), location=GCP_LOCATION)
        
        with st.spinner("Connecting to ChromaDB..."):
            if is_cloud_environment():
                # For cloud deployment
                client, collection = get_chroma_setup()
                if client is None or collection is None:
                    st.error("Failed to initialize ChromaDB")
                    return False
            else:
                # For local development
                client = PersistentClient(path="./chroma")
                collection = client.get_collection(name="char-split-collection")
            
            st.session_state.chroma_collection = collection
            doc_count = collection.count()
            st.success(f"Connected to ChromaDB ({doc_count} documents)")
        
        with st.spinner("Initializing AI Agent..."):
            # Initialize the simplified agent
            agent = SimplifiedDeviationAgent(
                openai_api_key=openai_api_key,
                collection=collection,
                embed_func=generate_query_embedding,
                vertex_model=generative_model
            )
            st.session_state.agent = agent
            st.session_state.agent_initialized = True
            
        st.success("Agent initialized successfully")
        return True
        
    except Exception as e:
        error_msg = f"Failed to initialize agent: {str(e)}"
        st.session_state.initialization_error = error_msg
        st.error(error_msg)
        return False

def display_chat_message(role: str, content: str):
    """Display a chat message"""
    with st.chat_message(role):
        st.write(content)

def get_generated_files():
    """Get list of generated report files with metadata"""
    files = []
    reports_dir = "./generated_reports"
    
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir, exist_ok=True)
        return files
    
    # Supported file types and their display names
    file_types = {
        '.pdf': 'PDF Report',
        '.html': 'HTML Report', 
        '.docx': 'Word Report',
        '.doc': 'Word Report'
    }
    
    try:
        for filename in os.listdir(reports_dir):
            file_path = os.path.join(reports_dir, filename)
            
            # Check if it's a file and has supported extension
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in file_types:
                    # Get file size in human readable format
                    file_size = get_file_size_human(file_path)
                    
                    # Get file modification time
                    mod_time = os.path.getmtime(file_path)
                    mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
                    
                    files.append({
                        'path': file_path,
                        'name': filename,
                        'type': file_types[file_ext],
                        'size': file_size,
                        'modified': mod_time_str,
                        'extension': file_ext
                    })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
        
    except Exception as e:
        st.error(f"Error scanning files: {str(e)}")
    
    return files

def get_file_size_human(file_path):
    """Convert file size to human readable format"""
    try:
        size_bytes = os.path.getsize(file_path)
        
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        size_index = 0
        
        while size_bytes >= 1024 and size_index < len(size_names) - 1:
            size_bytes /= 1024.0
            size_index += 1
        
        return f"{size_bytes:.1f} {size_names[size_index]}"
    except:
        return "Unknown"

def get_mime_type(file_extension):
    """Get MIME type for file extension"""
    mime_types = {
        'pdf': 'application/pdf',
        'html': 'text/html',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword'
    }
    return mime_types.get(file_extension, 'application/octet-stream')

def clear_generated_files():
    """Clear all generated report files"""
    reports_dir = "./generated_reports"
    
    if os.path.exists(reports_dir):
        try:
            for filename in os.listdir(reports_dir):
                file_path = os.path.join(reports_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            st.error(f"Error clearing files: {str(e)}")

def main():
    # Page configuration
    st.set_page_config(
        page_title="Deviation Report Generator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if not MODULES_LOADED:
        st.stop()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration and status
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Environment variables check
        st.markdown("#### 🔐 Environment Status")
        
        # Check environment setup
        openai_key = os.getenv('OPENAI_API_KEY')
        gcp_project = os.getenv('GCP_PROJECT', 'rag-test-467013')
        gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Create status indicators
        status_container = st.container()
        with status_container:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if openai_key:
                    st.success("✅")
                else:
                    st.error("❌")
            with col2:
                st.write("OpenAI API Key")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if gcp_creds and os.path.exists(gcp_creds):
                    st.success("✅")
                else:
                    st.error("❌")
            with col2:
                st.write("GCP Credentials")
            
            st.info(f"Project: {gcp_project}")
        
        if is_cloud_environment():
            st.info("🌐 Cloud Environment")
        else:
            st.info("💻 Local Environment")
        
        st.divider()
        
        # Agent status
        st.markdown("#### 🤖 Agent Status")
        if st.session_state.agent_initialized:
            st.success("Agent Ready")
            
            # Show ChromaDB info
            if st.session_state.chroma_collection:
                doc_count = st.session_state.chroma_collection.count()
                st.info(f"Knowledge Base: {doc_count} documents")
        else:
            st.warning("Agent Not Initialized")
            if st.button("Initialize Agent", type="primary", use_container_width=True):
                if initialize_agent():
                    st.rerun()
        
        st.divider()
        
        # Generated Files Download Section
        st.markdown("#### 📥 Generated Reports")
        
        # Check for generated files
        generated_files = get_generated_files()
        
        if generated_files:
            st.success(f"{len(generated_files)} file(s) available")
            
            for file_info in generated_files:
                file_path = file_info['path']
                file_name = file_info['name']
                file_type = file_info['type']
                file_size = file_info['size']
                
                # Create a clean file display
                with st.container():
                    st.markdown(f"**{file_type}**")
                    st.caption(f"{file_name} • {file_size}")
                    
                    try:
                        with open(file_path, "rb") as f:
                            file_data = f.read()
                            
                        # Get appropriate MIME type
                        mime_type = get_mime_type(file_info['extension'][1:])  # Remove the dot
                        
                        st.download_button(
                            label="Download",
                            data=file_data,
                            file_name=file_name,
                            mime=mime_type,
                            help=f"Download {file_type}",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error loading {file_name}")
                
                st.divider()
            
            # Clear files button
            if st.button("Clear All Files", type="secondary", use_container_width=True):
                clear_generated_files()
                st.success("Files cleared")
                st.rerun()
                
        else:
            st.info("No reports generated yet")
            st.caption("Reports will appear here after generation")
        
        st.divider()
        
        # Current session info
        if st.session_state.agent and hasattr(st.session_state.agent, 'current_session'):
            session = st.session_state.agent.current_session
            if session:
                st.markdown("#### 📈 Current Session")
                st.write(f"**ID:** {session['id']}")
                st.write(f"**Stage:** {session['stage']}")
                st.write(f"**Started:** {session['start_time'].strftime('%H:%M:%S')}")
        
        st.divider()
        
        # Reset button
        if st.button("Reset Session", type="secondary", use_container_width=True):
            if st.session_state.agent:
                st.session_state.agent.current_session = None
            st.session_state.messages = []
            st.rerun()
    
    # Main content area
    st.markdown("# Deviation Report Generator")
    st.markdown("### AI-powered deviation reporting with advanced document retrieval")
    
    if is_cloud_environment():
        st.info("🌐 Cloud Deployment Active")
    else:
        st.info("💻 Local Development Mode")
    
    # Check if agent is initialized
    if not st.session_state.agent_initialized:
        if st.session_state.initialization_error:
            st.error(f"Initialization failed: {st.session_state.initialization_error}")
        st.warning("Please initialize the agent using the sidebar before proceeding.")
        
        # Show setup instructions
        with st.expander("🛠️ Setup Instructions"):
            st.markdown("""
            **Environment Configuration:**
            
            ```bash
            export OPENAI_API_KEY='your-openai-api-key'
            export GOOGLE_APPLICATION_CREDENTIALS='./secrets/cl-rag-docs.json'
            export GCP_PROJECT='rag-test-467013'
            ```
            
            **Requirements:**
            - ChromaDB collection at `./chroma`
            - Collection name: `char-split-collection`
            - Valid GCP service account credentials
            """)
        return

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Check if we have an active session with the agent
    agent = st.session_state.agent
    current_session = getattr(agent, 'current_session', None)
    
    # If no active session, start one
    if not current_session:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start New Deviation Report", type="primary", use_container_width=True):
                with st.spinner("Starting new session..."):
                    greeting = agent.start_new_session()
                    st.session_state.messages.append({"role": "assistant", "content": greeting})
                    st.rerun()
        return
    
    # Handle ongoing conversation
    if prompt := st.chat_input("Type your response here...", key="chat_input"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Process with agent
        with st.spinner("Processing your request..."):
            try:
                response = agent.process_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
                # Check if session is complete and files were generated
                if "Reports Generated Successfully" in response:
                    st.balloons()
                    st.success("🎉 Reports completed successfully!")
                    
                    # Show a message directing users to the sidebar
                    st.info("📥 **Download your reports from the sidebar** ➡️")
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                display_chat_message("assistant", error_msg)
        
        st.rerun()
    
    # Show current session stage info
    if current_session:
        with st.expander("ℹ️ Session Details"):
            st.write(f"**Session ID:** {current_session['id']}")
            st.write(f"**Current Stage:** {current_session['stage']}")
            st.write(f"**Started:** {current_session['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if 'data' in current_session and current_session['data']:
                st.write("**Collected Data:**")
                data = current_session['data']
                
                if 'initiator' in data:
                    st.write(f"- Initiator: {data['initiator']}")
                if 'extracted' in data:
                    extracted = data['extracted']
                    for key, value in extracted.items():
                        if value and value != "":
                            st.write(f"- {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()