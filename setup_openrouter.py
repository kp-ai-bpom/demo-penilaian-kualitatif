# setup_openrouter.py
import os
import sys
from pathlib import Path

def setup_openrouter():
    """Setup OpenRouter configuration"""
    
    print("🎯 SETUP OPENROUTER")
    print("=" * 50)
    
    # Check if config exists
    config_file = Path("config.py")
    if not config_file.exists():
        print("❌ config.py not found. Creating...")
        create_config_file()
    
    # Check API key
    from config import OPENROUTER_API_KEY
    
    if OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        print("\n❌ OPENROUTER API KEY NOT CONFIGURED")
        print("\n📝 Please follow these steps:")
        print("1. Go to: https://openrouter.ai/keys")
        print("2. Sign up/login (it's free)")
        print("3. Create a new API key")
        print("4. Edit config.py and replace 'your-openrouter-api-key-here'")
        print("5. Run this setup again")
        
        # Show config file location
        print(f"\n📍 Config file: {config_file.absolute()}")
        return False
    else:
        print("✅ OpenRouter API key is configured")
        print("🔧 Testing connection...")
        
        # Test connection
        try:
            from src.vector_store import setup_llm
            llm = setup_llm()
            response = llm.invoke("Hello, respond with 'OK'")
            print(f"✅ Connection test: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

def create_config_file():
    """Create config file for OpenRouter"""
    config_content = '''# config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories
for directory in [DATA_DIR, VECTOR_STORE_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# OpenRouter Configuration
OPENROUTER_API_KEY = "your-openrouter-api-key-here"  # ⚠️ REPLACE THIS!
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model names for OpenRouter
LLM_MODEL = "qwen/qwen-2.5-coder-7b-instruct:free"  # Free model

# Embedding model (OpenRouter doesn't support embeddings directly)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local embeddings

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
'''
    
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("✅ Created config.py")

if __name__ == "__main__":
    success = setup_openrouter()
    if success:
        print("\n🎉 OpenRouter setup completed successfully!")
        print("You can now run the notebooks:")
        print("1. 01_data_preparation.ipynb")
        print("2. 02_llm_extraction.ipynb")
        print("3. 03_rag_system.ipynb")
        print("4. 04_assessment_demo.ipynb")
    else:
        print("\n⚠️ Please configure your API key and run setup again.")