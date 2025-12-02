# import os
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS

# # Configuration
# BASE_URL = "https://openrouter.ai/api/v1"
# API_KEY = "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"
# LLM_MODEL = "mistralai/mistral-7b-instruct-v0.2"
# EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# def setup_llm():
#     """Setup LLM for chat"""
#     return ChatOpenAI(
#         base_url=BASE_URL,
#         api_key=API_KEY,
#         model=LLM_MODEL,
#         temperature=0.7,
#         max_tokens=512,
#         streaming=False
#         timeout=60  # Add timeout
#     )

# def setup_embedding_model():
#     """Setup embedding model with better error handling"""
#     try:
#         embeddings = OpenAIEmbeddings(
#             base_url=BASE_URL,
#             api_key=API_KEY,
#             model=EMBEDDING_MODEL,
#             timeout=30,
#             max_retries=2
#         )
        
#         # Test the embedding model
#         test_text = "test embedding"
#         test_embedding = embeddings.embed_query(test_text)
        
#         if test_embedding and len(test_embedding) > 0:
#             print(f"✅ Embedding model ready - dimension: {len(test_embedding)}")
#             return embeddings
#         else:
#             raise ValueError("Empty embedding received")
            
#     except Exception as e:
#         print(f"❌ Error setting up embedding model: {e}")
#         print("🔄 Using fallback embeddings...")
#         return setup_fallback_embeddings()

# def setup_fallback_embeddings():
#     """Fallback embeddings when main model fails"""
#     from langchain.embeddings import FakeEmbeddings
    
#     print("⚠️ Using fake embeddings for testing")
#     return FakeEmbeddings(size=768)

# def load_vector_store(vector_store_path):
#     """Load existing vector store"""
#     try:
#         embedding_model = setup_embedding_model()
#         if not os.path.exists(vector_store_path):
#             raise FileNotFoundError(f"Vector store not found: {vector_store_path}")
            
#         return FAISS.load_local(
#             vector_store_path,
#             embedding_model,
#             allow_dangerous_deserialization=True
#         )
#     except Exception as e:
#         print(f"❌ Error loading vector store: {e}")
#         return None
    

#     # sadasdas

# src/vector_store.py - VERSION DENGAN CONFIG
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

try:
    from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, EMBEDDING_MODEL
except ImportError:
    # Fallback config
    OPENROUTER_API_KEY = "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"  # ⚠️ GANTI INI!
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "qwen/qwen-2.5-coder-7b-instruct:free"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def setup_llm():
    """Setup LLM for chat dengan OpenRouter"""
    if OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        print("❌ Please set your OpenRouter API key in config.py")
        print("💡 Get free API key from: https://openrouter.ai/keys")
        raise ValueError("OpenRouter API key not configured")
    
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=512,
        streaming=False,
        timeout=60,
        headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "RAG Assessment System"
        }
    )

def setup_embedding_model():
    """Setup embedding model untuk OpenRouter compatibility"""
    try:
        # Coba Hugging Face embeddings dulu
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print(f"🔧 Loading embeddings: {EMBEDDING_MODEL}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test the embeddings
        test_text = "test embedding"
        test_result = embeddings.embed_query(test_text)
        
        if test_result and len(test_result) > 0:
            print(f"✅ Embeddings ready - dimension: {len(test_result)}")
            return embeddings
        else:
            raise ValueError("Empty embedding result")
            
    except Exception as e:
        print(f"❌ HuggingFace embeddings failed: {e}")
        print("🔄 Using fake embeddings for demo...")
        from langchain.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=384)

def load_vector_store(vector_store_path):
    """Load existing vector store"""
    try:
        embedding_model = setup_embedding_model()
        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(f"Vector store not found: {vector_store_path}")
            
        return FAISS.load_local(
            vector_store_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None