"""
Application configuration
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Intelligent AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENV: str = os.getenv("ENV", "development")  # Environment variable for logger configuration
    
    
    # Google Drive API Credentials (from .env)
    GOOGLE_DRIVE_CLIENT_ID: Optional[str] = os.getenv("GOOGLE_DRIVE_CLIENT_ID")
    GOOGLE_DRIVE_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET")
    GOOGLE_DRIVE_PROJECT_ID: Optional[str] = os.getenv("GOOGLE_DRIVE_PROJECT_ID")
    GOOGLE_DRIVE_AUTH_URI: str = "https://accounts.google.com/o/oauth2/auth"
    GOOGLE_DRIVE_TOKEN_URI: str = "https://oauth2.googleapis.com/token"
    GOOGLE_DRIVE_AUTH_PROVIDER_X509_CERT_URL: str = "https://www.googleapis.com/oauth2/v1/certs"
    GOOGLE_DRIVE_REDIRECT_URIS: str = "http://localhost"  # Comma-separated or single URI
    
    # Google Drive API Configuration
    GOOGLE_DRIVE_TOKEN_FILE: Optional[str] = os.getenv("GOOGLE_DRIVE_TOKEN_FILE")
    GOOGLE_DRIVE_FOLDER_ID: Optional[str] = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    
    # MongoDB Configuration
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "central_acts")
    # MongoDB Atlas credentials (for secure connection)
    MONGO_USERNAME: Optional[str] = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD: Optional[str] = os.getenv("MONGO_PASSWORD")
    MONGO_CLUSTER_URL: Optional[str] = os.getenv("MONGO_CLUSTER_URL")
    MONGO_APP_NAME: Optional[str] = os.getenv("MONGO_APP_NAME")
    
    # OpenAI Configuration (Optional now)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Hugging Face Configuration (NEW)
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    HUGGINGFACE_EMBEDDING_MODEL: str = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Change default from:
    HUGGINGFACE_METADATA_MODEL: str = os.getenv("HUGGINGFACE_METADATA_MODEL", "google/flan-t5-base")
    # Already correct, just make sure .env has the right model
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "intelligent-ai-v2")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENV")

    # LLM Defaults (with safe defaults)
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    RAG_MAX_CONTEXT_CHARS: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
    RAG_MAX_SNIPPET_CHARS: int = int(os.getenv("RAG_MAX_SNIPPET_CHARS", "500"))
    SUMMARY_FULL_DOCUMENT_MODE: bool = os.getenv("SUMMARY_FULL_DOCUMENT_MODE", "false").lower() == "true"
    SUMMARY_MAX_CONTEXT_CHARS: int = int(os.getenv("SUMMARY_MAX_CONTEXT_CHARS", "8000"))
    SUMMARY_DOC_MAX_CHUNKS: int = int(os.getenv("SUMMARY_DOC_MAX_CHUNKS", "20"))
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # Changed to 384 for Hugging Face
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()