#!/usr/bin/env python3
"""
Export ChromaDB data for cloud deployment
Run this script locally to prepare your data for upload
"""

import os
import shutil
import zipfile
from pathlib import Path

def export_chroma_data():
    """Export your local ChromaDB data"""
    
    # Source and destination paths
    source_path = "./chroma"  # Your local ChromaDB path
    temp_path = "./chroma_export"
    zip_path = "./data/chroma_backup.zip"
    
    print("📦 Exporting ChromaDB data for cloud deployment...")
    
    # Check if source exists
    if not os.path.exists(source_path):
        print(f"❌ Error: ChromaDB directory not found at {source_path}")
        print("Make sure you have your ChromaDB collection in the ./chroma directory")
        return False
    
    try:
        # Create export directory
        os.makedirs("./data", exist_ok=True)
        
        # Copy ChromaDB data to temporary location
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        shutil.copytree(source_path, temp_path)
        
        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create relative path for the zip
                    arc_path = os.path.relpath(file_path, temp_path)
                    zipf.write(file_path, arc_path)
        
        # Clean up temporary directory
        shutil.rmtree(temp_path)
        
        # Get file size
        file_size = os.path.getsize(zip_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✅ ChromaDB data exported successfully!")
        print(f"📁 File: {zip_path}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 100:
            print("⚠️  Warning: File is large (>100MB). Consider using Google Cloud Storage.")
            print("   See upload_to_gcs() function below.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting ChromaDB data: {e}")
        return False

def upload_to_gcs():
    """Upload ChromaDB backup to Google Cloud Storage (optional)"""
    try:
        from google.cloud import storage
        
        # Configuration
        bucket_name = f"{os.getenv('GCP_PROJECT', 'your-project')}-chroma-backup"
        source_file = "./data/chroma_backup.zip"
        
        if not os.path.exists(source_file):
            print("❌ No backup file found. Run export_chroma_data() first.")
            return False
        
        print(f"☁️  Uploading to Google Cloud Storage bucket: {bucket_name}")
        
        # Initialize client
        client = storage.Client()
        
        # Create bucket if it doesn't exist
        try:
            bucket = client.bucket(bucket_name)
            bucket.reload()  # Check if bucket exists
        except:
            bucket = client.create_bucket(bucket_name)
            print(f"✅ Created bucket: {bucket_name}")
        
        # Upload file
        blob = bucket.blob("chroma_backup.zip")
        blob.upload_from_filename(source_file)
        
        print(f"✅ Uploaded to gs://{bucket_name}/chroma_backup.zip")
        print(f"💡 Add this to your Streamlit secrets:")
        print(f'   GCS_BUCKET_NAME = "{bucket_name}"')
        
        return True
        
    except ImportError:
        print("❌ google-cloud-storage not installed")
        print("   Install with: pip install google-cloud-storage")
        return False
    except Exception as e:
        print(f"❌ Error uploading to GCS: {e}")
        return False

def check_file_sizes():
    """Check sizes of files to be uploaded"""
    print("📊 Checking file sizes for GitHub upload...")
    
    large_files = []
    
    # Check common directories
    paths_to_check = [
        "./data/chroma_backup.zip",
        "./chroma",
        "./docker-volumes",
        "./generated_reports"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path)
                size_mb = size / (1024 * 1024)
                print(f"📁 {path}: {size_mb:.2f} MB")
                
                if size_mb > 25:  # GitHub warning limit
                    large_files.append((path, size_mb))
            
            elif os.path.isdir(path):
                total_size = 0
                for root, dirs, files in os.walk(path):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))
                
                size_mb = total_size / (1024 * 1024)
                print(f"📁 {path}/: {size_mb:.2f} MB")
                
                if size_mb > 25:
                    large_files.append((path, size_mb))
    
    if large_files:
        print("\n⚠️  Large files detected (>25MB):")
        for file_path, size in large_files:
            print(f"   {file_path}: {size:.2f} MB")
        print("   Consider using Git LFS or Google Cloud Storage")
    
    return large_files

if __name__ == "__main__":
    print("🚀 ChromaDB Export Tool for Streamlit Cloud Deployment")
    print("=" * 60)
    
    # Check current file sizes
    check_file_sizes()
    
    print("\n1. Exporting ChromaDB data...")
    if export_chroma_data():
        print("\n2. Optional: Upload to Google Cloud Storage")
        upload_choice = input("   Upload to GCS? (y/n): ").lower().strip()
        
        if upload_choice == 'y':
            upload_to_gcs()
    
    print("\n✅ Export complete!")
    print("\nNext steps:")
    print("1. Add all files to your GitHub repository")
    print("2. Create your Streamlit Cloud app")
    print("3. Configure secrets in Streamlit Cloud")
    print("4. Deploy!")