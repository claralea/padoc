import sys

def fix_sqlite():
    """Fix SQLite version for ChromaDB compatibility"""
    try:
        # Try to import pysqlite3 and replace sqlite3
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ SQLite3 replaced with pysqlite3-binary for ChromaDB compatibility")
        return True
    except ImportError:
        print("⚠️ pysqlite3-binary not available, using system sqlite3")
        return False

# Execute the fix immediately when this module is imported
fix_sqlite()