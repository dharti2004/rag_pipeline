import os
import shutil
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("API_KEY")
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def get_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None
_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        storage_path = "./qdrant_local"
        try:
            _qdrant_client = QdrantClient(path=storage_path)
        except Exception as e:
            if "already accessed by another instance" in str(e):
                print("Qdrant storage is locked. Please close other processes using it (e.g. another script or FastAPI server).")
                return None
            else:
                print(f"Error initializing Qdrant client: {e}")
                return None
    return _qdrant_client