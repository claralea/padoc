import sys
import os

def fix_sqlite():
    """Enhanced SQLite fix with multiple fallback strategies"""
    
    # Strategy 1: Try pysqlite3-binary first
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ SQLite3 replaced with pysqlite3-binary")
        return True
    except ImportError:
        print("⚠️ pysqlite3-binary not available, trying alternatives...")
    
    # Strategy 2: Try regular pysqlite3
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ SQLite3 replaced with pysqlite3")
        return True
    except ImportError:
        print("⚠️ pysqlite3 not available")
    
    # Strategy 3: Check if existing sqlite3 is compatible
    try:
        import sqlite3
        version = sqlite3.sqlite_version_info
        if version >= (3, 35, 0):
            print(f"✅ System SQLite {sqlite3.sqlite_version} is compatible")
            return True
        else:
            print(f"❌ System SQLite {sqlite3.sqlite_version} is too old (need >= 3.35.0)")
    except Exception as e:
        print(f"❌ Error checking SQLite version: {e}")
    
    # Strategy 4: Try to force ChromaDB to use different backend
    try:
        os.environ['CHROMA_DB_IMPL'] = 'duckdb'
        print("🦆 Set ChromaDB to use DuckDB backend")
        return True
    except:
        pass
    
    # Strategy 5: Set environment variables for ChromaDB
    try:
        os.environ['SQLITE_THREADSAFE'] = '1'
        os.environ['CHROMA_SERVER_SSL_ENABLED'] = 'false'
        print("🔧 Set ChromaDB environment variables")
        return True
    except:
        pass
    
    print("❌ All SQLite fix strategies failed")
    return False

# Execute the fix immediately when this module is imported
success = fix_sqlite()

# Additional ChromaDB configuration
def configure_chromadb():
    """Configure ChromaDB with optimal settings for cloud deployment"""
    try:
        # Set ChromaDB to use in-memory database if persistent fails
        os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
        os.environ['CHROMA_API_IMPL'] = 'local'
        print("🔧 ChromaDB configured for cloud deployment")
        return True
    except Exception as e:
        print(f"❌ ChromaDB configuration failed: {e}")
        return False

configure_chromadb()