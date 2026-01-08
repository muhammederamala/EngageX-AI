from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import List
import asyncio

from app.models.schemas import (
    ChatbotCreate, ChatbotUpdate, ChatMessage, ChatResponse,
    NotificationOptimization, OptimizedNotification,
    BehaviorAnalysis, BehaviorInsights
)
from app.services.rag_service import rag_service
from app.services.ai_service import ai_service

router = APIRouter()

@router.post("/chatbots", response_model=dict)
async def create_chatbot(chatbot: ChatbotCreate):
    """Create a new chatbot with RAG knowledge base"""
    try:
        result = await rag_service.create_knowledge_base(
            chatbot_id=f"{chatbot.user_id}_{chatbot.name}",
            knowledge_items=chatbot.rag_config.knowledge_base
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/chatbots/{chatbot_id}")
async def update_chatbot(chatbot_id: str, chatbot: ChatbotUpdate):
    """Update chatbot knowledge base"""
    try:
        if chatbot.rag_config:
            result = await rag_service.create_knowledge_base(
                chatbot_id=chatbot_id,
                knowledge_items=chatbot.rag_config.knowledge_base
            )
            return {"success": True, "data": result}
        return {"success": True, "message": "No knowledge base updates"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(message: ChatMessage):
    """Generate AI response using RAG"""
    try:
        # Query knowledge base
        context = await rag_service.query_knowledge_base(
            chatbot_id=message.chatbot_id,
            query=message.message
        )
        
        # Generate response with default personality
        personality = {"tone": "friendly"}
        
        response = await rag_service.generate_response(
            query=message.message,
            context=context,
            conversation_history=message.conversation_history,
            personality=personality
        )
        
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notifications/optimize", response_model=OptimizedNotification)
async def optimize_notification(request: NotificationOptimization):
    """Optimize notification content and timing using AI"""
    try:
        result = await ai_service.optimize_notification(
            user_id=request.user_id,
            customer_phone=request.customer_phone,
            template=request.template,
            business_context=request.business_context,
            conversation_history=request.conversation_history
        )
        
        return OptimizedNotification(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/behavior/analyze", response_model=BehaviorInsights)
async def analyze_behavior(request: BehaviorAnalysis):
    """Analyze customer behavior patterns"""
    try:
        insights = await ai_service.analyze_customer_behavior(
            user_id=request.user_id,
            days_back=request.days_back
        )
        
        return BehaviorInsights(**insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/upload")
async def upload_knowledge_file(
    chatbot_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process knowledge base files"""
    try:
        # Basic file validation
        if file.size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.content_type == "application/pdf":
            # In production, implement PDF processing
            processed_content = content.decode('utf-8', errors='ignore')
        elif file.content_type == "text/plain":
            processed_content = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Create knowledge base item
        from app.models.schemas import KnowledgeBaseItem
        knowledge_item = KnowledgeBaseItem(
            type="text",
            content=processed_content,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size
            }
        )
        
        # Update knowledge base
        result = await rag_service.create_knowledge_base(
            chatbot_id=chatbot_id,
            knowledge_items=[knowledge_item]
        )
        
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "EngageX AI"}