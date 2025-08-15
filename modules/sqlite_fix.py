import sys

def ensure_sqlite_compatibility():
    """Ensure SQLite compatibility for ChromaDB - Must run before chromadb import"""
    try:
        # Try to use pysqlite3-binary first
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ Replaced sqlite3 with pysqlite3-binary")
        return True
    except ImportError:
        print("⚠ pysqlite3-binary not available, checking system sqlite3...")
        
    try:
        # Check if system sqlite3 is compatible
        import sqlite3
        version = sqlite3.sqlite_version_info
        if version >= (3, 35, 0):
            print(f"✅ System SQLite {sqlite3.sqlite_version} is compatible")
            return True
        else:
            print(f"❌ System SQLite {sqlite3.sqlite_version} is too old (need >= 3.35.0)")
            return False
    except Exception as e:
        print(f"❌ Error checking SQLite: {e}")
        return False

# Apply the fix immediately when module is imported
ensure_sqlite_compatibility()