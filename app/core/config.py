from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "EngageX AI Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    MONGODB_URI: str = "mongodb://localhost:27017/engagex"
    REDIS_URL: str = "redis://localhost:6379"
    
    # AI Models
    HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-medium"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GOOGLE_API_KEY: Optional[str] = None
    
    # Vector DB
    FAISS_INDEX_PATH: str = "./faiss_indexes"
    
    # File Upload
    MAX_FILE_SIZE: int = 50485760  # 50MB
    UPLOAD_DIR: str = "uploads"
    
    # Node.js Server
    NODE_SERVER_URL: str = "http://localhost:3000"
    NODE_SERVER_API_KEY: str = "test-api-key"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"

settings = Settings()