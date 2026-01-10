from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# =========================
# Enums
# =========================

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


# =========================
# Core Shared Models
# =========================

class ConversationMessage(BaseModel):
    """
    Standard LLM-compatible conversation message
    """
    role: str  # "system" | "user" | "assistant"
    content: str


# =========================
# Knowledge Base & RAG
# =========================

class KnowledgeBaseItem(BaseModel):
    type: KnowledgeBaseType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    knowledge_base: List[KnowledgeBaseItem]
    embedding_model: str = "text-embedding-3-large"
    max_tokens: int = 4000
    temperature: float = 0.7


# =========================
# Chatbot Configuration
# =========================

class ChatbotPersonality(BaseModel):
    tone: PersonalityTone = PersonalityTone.FRIENDLY
    language: str = "en"
    custom_instructions: Optional[str] = None


class ChatbotCreate(BaseModel):
    chatbot_id: str
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


# =========================
# Chat API
# =========================

class ChatRequest(BaseModel):
    chatbot_id: str = Field(..., alias="chatbotId")
    message: str
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        alias="conversationHistory"
    )
    system_prompt: Optional[str] = Field(None, alias="systemPrompt")

    class Config:
        populate_by_name = True  # ✅ critical

class ChatResponse(BaseModel):
    """
    AI-generated response
    """
    message: str
    confidence: float = 0.0
    tokens_used: int
    processing_time: float


# =========================
# Notification Optimization
# =========================

class NotificationOptimization(BaseModel):
    user_id: str
    customer_phone: str
    template: str
    business_context: str
    conversation_history: List[ConversationMessage] = Field(default_factory=list)


class OptimizedNotification(BaseModel):
    content: str
    optimal_time: datetime
    confidence: float
    reasoning: str


# =========================
# Behavior Analysis
# =========================

class BehaviorAnalysis(BaseModel):
    user_id: str
    days_back: int = 30


class BehaviorInsights(BaseModel):
    best_times: List[Dict[str, Any]]
    customer_behavior: Dict[str, Any]
    engagement_patterns: Dict[str, Any]


# =========================
# File Processing
# =========================

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
