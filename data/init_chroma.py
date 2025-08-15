# Initialize ChromaDB from backup data with SQLite fix

# Fix SQLite issue for Streamlit Cloud
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import chromadb
import os
import zipfile
import shutil
from pathlib import Path

def download_chroma_from_gcs():
    """Download ChromaDB backup from Google Cloud Storage"""
    try:
        from google.cloud import storage
        
        # Initialize GCS client
        client = storage.Client()
        bucket_name = st.secrets.get("GCS_BUCKET_NAME", f"{st.secrets['GCP_PROJECT']}-chroma-backup")
        bucket = client.bucket(bucket_name)
        
        # Download the ChromaDB backup
        blob = bucket.blob("chroma_backup.zip")
        blob.download_to_filename("chroma_backup.zip")
        
        return True
    except Exception as e:
        st.error(f"Error downloading ChromaDB from GCS: {e}")
        return False

def extract_chroma_data():
    """Extract ChromaDB data from backup"""
    try:
        # Check if local backup exists first
        if os.path.exists("data/chroma_backup.zip"):
            backup_file = "data/chroma_backup.zip"
        elif os.path.exists("chroma_backup.zip"):
            backup_file = "chroma_backup.zip"
        else:
            # Try to download from GCS
            if not download_chroma_from_gcs():
                return False
            backup_file = "chroma_backup.zip"
        
        # Extract the backup
        with zipfile.ZipFile(backup_file, 'r') as zip_ref:
            zip_ref.extractall("./chroma_data")
        
        st.success("✅ ChromaDB data extracted successfully")
        return True
        
    except Exception as e:
        st.error(f"Error extracting ChromaDB data: {e}")
        return False

def initialize_chroma_collection():
    """Initialize ChromaDB collection"""
    try:
        # Extract data if needed
        if not os.path.exists("./chroma_data"):
            if not extract_chroma_data():
                return None, None
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_data")
        
        # Get or create collection
        try:
            collection = client.get_collection(name="char-split-collection")
            doc_count = collection.count()
            st.success(f"✅ Connected to existing ChromaDB collection ({doc_count} documents)")
        except:
            # If collection doesn't exist, create an empty one
            collection = client.create_collection(name="char-split-collection")
            st.warning("⚠️ Created new empty ChromaDB collection")
        
        return client, collection
        
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None, None

# Cache the initialization to avoid repeated setup
@st.cache_resource
def get_chroma_setup():
    """Cached ChromaDB setup"""
    return initialize_chroma_collection()