import sys
import os
import logging
import glob
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from api.config import settings, langchain_service

def list_schema_files():
    """List all schema files in the tables directory"""
    tables_dir = "static/tables"
    if not os.path.exists(tables_dir):
        print(f"Tables directory not found: {tables_dir}")
        return
        
    print("\nAvailable schema files:")
    print("----------------------")
    
    # List text files
    txt_files = glob.glob(os.path.join(tables_dir, "*.txt"))
    if txt_files:
        print("\nText files:")
        for f in txt_files:
            print(f"  - {os.path.basename(f)}")
    
    # List JSON files
    json_files = glob.glob(os.path.join(tables_dir, "*.json"))
    if json_files:
        print("\nJSON files:")
        for f in json_files:
            print(f"  - {os.path.basename(f)}")
            
    print("\nTotal files:", len(txt_files) + len(json_files))

def refresh_schema():
    """Refresh the schema vector store"""
    try:
        print("Starting schema refresh...")
        
        # List available schema files
        list_schema_files()
        
        # Force refresh schema
        print("\nRefreshing vector store...")
        langchain_service.refresh_schema()
        
        print("✅ Schema refresh completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error refreshing schema: {e}")
        return False

if __name__ == "__main__":
    success = refresh_schema()
    sys.exit(0 if success else 1)
