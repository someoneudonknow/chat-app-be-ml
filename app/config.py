import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    APP_NAME: str = os.environ.get("APP_NAME", "summarization_service")
    APP_HOST: str = os.environ.get("APP_HOST", "localhost")
    APP_PORT: int = os.environ.get("APP_PORT", 8000)
    APP_DEBUG: bool = os.environ.get("APP_DEBUG", True)
    APP_VERSION: str = os.environ.get("APP_VERSION", "0.1.0")

    LANGSMITH_TRACING: bool = os.environ.get("LANGSMITH_TRACING", False)
    LANGSMITH_ENDPOINT: str = os.environ.get(
        "LANGSMITH_ENDPOINT", "https://api.langsmith.com"
    )
    LANGSMITH_API_KEY: str = os.environ.get("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.environ.get("LANGSMITH_PROJECT", "default_project")

    AI_MODEL: str = os.environ.get("AI_MODEL", "llama3")

    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

    HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

    EMBEDDINGS_MODEL: str = os.environ.get("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")

    MONGODB_URI: str = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DB: str = os.environ.get("MONGODB_DB", "chat-app")

    RAG_CONTEXT_LIMIT: int = os.environ.get("RAG_CONTEXT_LIMIT", 30)
    RAG_RETRIEVAL_TOP_K: int = os.environ.get("RAG_RETRIEVAL_TOP_K", 5)
    RAG_CHUNK_SIZE: int = os.environ.get("RAG_CHUNK_SIZE", 1000)
    RAG_CHUNK_OVERLAP: int = os.environ.get("RAG_CHUNK_OVERLAP", 200)

    VECTOR_STORE_DIR: str = os.environ.get("VECTOR_STORE_DIR", "app/db/vectorstore")

    AWS_ACCESS_KEY: str = os.environ.get("AWS_ACCESS_KEY", "")
    AWS_SECRET_KEY: str = os.environ.get("AWS_SECRET_KEY", "")
    AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
    AWS_BUCKET_NAME: str = os.environ.get("S3_BUCKET_NAME", "chatapp-2025")

    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")

    REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT: int = os.environ.get("REDIS_PORT", 6379)


settings = Settings()
