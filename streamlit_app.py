import sys
import os
from pathlib import Path

# Add modules to path FIRST
sys.path.append('./modules')
sys.path.append('.')

# Import SQLite fix BEFORE anything else
try:
    from modules.sqlite_fix import fix_sqlite
except ImportError:
    # Fallback inline fix
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ SQLite fixed (inline)")
    except ImportError:
        print("⚠️ SQLite fix not available")

import streamlit as st



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
            st.error(f"Error setting up cloud secrets: {e}")
            return False
    else:
        st.info("💻 Running locally")
        
        # Use local environment variables
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            st.error("❌ OPENAI_API_KEY environment variable not set locally")
            st.info("Set it with: `export OPENAI_API_KEY='your-key-here'`")
            return False
        
        gcp_project = os.getenv('GCP_PROJECT')
        if not gcp_project:
            st.warning("⚠️ GCP_PROJECT not set, using default")
            os.environ['GCP_PROJECT'] = 'rag-test-467013'
        
        # Check GCP credentials file
        gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not gcp_creds or not os.path.exists(gcp_creds):
            st.error("❌ GOOGLE_APPLICATION_CREDENTIALS not set or file not found")
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
        )
        
        from modules.run_integrated_agent import SimplifiedDeviationAgent
        from modules.deviation_structures import DeviationType, Priority, Status, Department
        
        # For local testing, we'll use the original ChromaDB setup
        if not is_cloud_environment():
            from cli import PersistentClient
        else:
            from data.init_chroma import get_chroma_setup
        
        MODULES_LOADED = True
        
    except ImportError as e:
        st.error(f"❌ Error importing modules: {e}")
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
        st.error("❌ Modules not loaded - cannot initialize agent")
        return False
        
    try:
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("❌ OPENAI_API_KEY not found")
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
                    st.error("❌ Failed to initialize ChromaDB")
                    return False
            else:
                # For local development
                client = PersistentClient(path="./chroma")
                collection = client.get_collection(name="char-split-collection")
            
            st.session_state.chroma_collection = collection
            doc_count = collection.count()
            st.success(f"✅ Connected to ChromaDB ({doc_count} documents)")
        
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
            
        st.success("✅ Agent initialized successfully!")
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
        st.title("🔧 Configuration")
        
        # Environment variables check
        st.subheader("🔑 Environment")
        
        # Check environment setup
        openai_key = os.getenv('OPENAI_API_KEY')
        gcp_project = os.getenv('GCP_PROJECT', 'rag-test-467013')
        gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if openai_key:
            st.success("✅ OpenAI API Key Set")
        else:
            st.error("❌ OpenAI API Key Missing")
        
        if gcp_creds and os.path.exists(gcp_creds):
            st.success("✅ GCP Credentials Set")
        else:
            st.error("❌ GCP Credentials Missing")
        
        st.info(f"📍 GCP Project: {gcp_project}")
        
        if is_cloud_environment():
            st.success("🌐 Cloud Environment")
        else:
            st.info("💻 Local Environment")
        
        st.divider()
        
        # Agent status
        if st.session_state.agent_initialized:
            st.success("✅ Agent Ready")
            
            # Show ChromaDB info
            if st.session_state.chroma_collection:
                doc_count = st.session_state.chroma_collection.count()
                st.info(f"📚 Knowledge Base: {doc_count} documents")
        else:
            st.warning("⚠️ Agent Not Initialized")
            if st.button("🚀 Initialize Agent"):
                if initialize_agent():
                    st.rerun()
        
        st.divider()
        
        # Generated Files Download Section
        st.subheader("📥 Generated Reports")
        
        # Check for generated files
        generated_files = get_generated_files()
        
        if generated_files:
            st.success(f"✅ {len(generated_files)} file(s) available")
            
            for file_info in generated_files:
                file_path = file_info['path']
                file_name = file_info['name']
                file_type = file_info['type']
                file_size = file_info['size']
                
                # Create a container for each file
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{file_type}**")
                        st.caption(f"{file_name} ({file_size})")
                    
                    with col2:
                        try:
                            with open(file_path, "rb") as f:
                                file_data = f.read()
                                
                            # Get appropriate MIME type
                            mime_type = get_mime_type(file_info['extension'][1:])  # Remove the dot
                            
                            st.download_button(
                                label="⬇️",
                                data=file_data,
                                file_name=file_name,
                                mime=mime_type,
                                help=f"Download {file_type} report",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"❌ Error loading {file_name}")
                    
                    st.divider()
            
            # Clear files button
            if st.button("🗑️ Clear All Files", use_container_width=True):
                clear_generated_files()
                st.success("Files cleared!")
                st.rerun()
                
        else:
            st.info("No reports generated yet")
            st.caption("Reports will appear here after generation")
        
        st.divider()
        
        # Current session info
        if st.session_state.agent and hasattr(st.session_state.agent, 'current_session'):
            session = st.session_state.agent.current_session
            if session:
                st.subheader("📈 Current Session")
                st.write(f"**ID:** {session['id']}")
                st.write(f"**Stage:** {session['stage']}")
                st.write(f"**Started:** {session['start_time'].strftime('%H:%M:%S')}")
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset Session", type="secondary"):
            if st.session_state.agent:
                st.session_state.agent.current_session = None
            st.session_state.messages = []
            st.rerun()
    
    # Main content area
    st.title("📊 Pharmacy Manufacturing Deviation Report Generator")
    
    if is_cloud_environment():
        st.markdown("🌐 **Cloud Deployment** - AI-powered deviation reporting with RAG")
    else:
        st.markdown("💻 **Local Development** - AI-powered deviation reporting with RAG")
    
    # Check if agent is initialized
    if not st.session_state.agent_initialized:
        if st.session_state.initialization_error:
            st.error(f"Initialization failed: {st.session_state.initialization_error}")
        st.warning("Please initialize the agent using the sidebar before proceeding.")
        
        # Show setup instructions
        with st.expander("🛠️ Setup Instructions"):
            st.markdown("""
            **For Local Development:**
            
            Set these environment variables:
            ```bash
            export OPENAI_API_KEY='your-openai-api-key'
            export GOOGLE_APPLICATION_CREDENTIALS='./secrets/cl-rag-docs.json'
            export GCP_PROJECT='rag-test-467013'
            ```
            
            **ChromaDB Setup:**
            - Ensure your ChromaDB collection exists at `./chroma`
            - Collection name should be `char-split-collection`
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
        if st.button("🚀 Start New Deviation Report", type="primary", use_container_width=True):
            with st.spinner("Starting new session..."):
                greeting = agent.start_new_session()
                st.session_state.messages.append({"role": "assistant", "content": greeting})
                st.rerun()
        return
    
    # Handle ongoing conversation
    if prompt := st.chat_input("Type your response here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Process with agent
        with st.spinner("🤖 Agent is processing..."):
            try:
                response = agent.process_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
                # Check if session is complete and files were generated
                if "Reports Generated Successfully" in response:
                    st.balloons()
                    st.success("🎉 Reports completed!")
                    
                    # Show a message directing users to the sidebar
                    st.info("📥 **Download your reports from the sidebar** ➡️")
                
            except Exception as e:
                error_msg = f"❌ Error processing input: {str(e)}"
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