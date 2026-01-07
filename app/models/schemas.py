from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class KnowledgeBaseType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    URL = "url"
    FAQ = "faq"

class PersonalityTone(str, Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    FORMAL = "formal"

class KnowledgeBaseItem(BaseModel):
    type: KnowledgeBaseType
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class RAGConfig(BaseModel):
    knowledge_base: List[KnowledgeBaseItem]
    embedding_model: str = "text-embedding-ada-002"
    max_tokens: int = 4000
    temperature: float = 0.7

class ChatbotPersonality(BaseModel):
    tone: PersonalityTone = PersonalityTone.FRIENDLY
    language: str = "en"
    custom_instructions: Optional[str] = None

class ChatbotCreate(BaseModel):
    user_id: str
    name: str
    description: Optional[str] = None
    rag_config: RAGConfig
    personality: ChatbotPersonality

class ChatbotUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    rag_config: Optional[RAGConfig] = None
    personality: Optional[ChatbotPersonality] = None

class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    user_id: str
    chatbot_id: str

class ChatResponse(BaseModel):
    message: str
    confidence: float
    tokens_used: int
    processing_time: float

class NotificationOptimization(BaseModel):
    user_id: str
    customer_phone: str
    template: str
    business_context: str
    conversation_history: Optional[List[Dict]] = []

class OptimizedNotification(BaseModel):
    content: str
    optimal_time: datetime
    confidence: float
    reasoning: str

class BehaviorAnalysis(BaseModel):
    user_id: str
    days_back: int = 30

class BehaviorInsights(BaseModel):
    best_times: List[Dict[str, Any]]
    customer_behavior: Dict[str, Any]
    engagement_patterns: Dict[str, Any]

class FileUpload(BaseModel):
    filename: str
    content_type: str
    size: int

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None