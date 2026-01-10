from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import List

from app.models.schemas import (
    ChatbotCreate,
    ChatbotUpdate,
    ChatRequest,
    ChatResponse,
    NotificationOptimization,
    OptimizedNotification,
    BehaviorAnalysis,
    BehaviorInsights,
    KnowledgeBaseItem
)
from app.services.rag_service import rag_service
from app.services.ai_service import ai_service

router = APIRouter()


# ---------------------------------------------------------
# Chatbot Management
# ---------------------------------------------------------

@router.post("/chatbots", response_model=dict)
async def create_chatbot(chatbot: ChatbotCreate):
    """Create a new chatbot with RAG knowledge base"""
    chatbot_id = chatbot.chatbot_id
    try:
        result = await rag_service.create_knowledge_base(
            chatbot_id=chatbot_id,
            knowledge_items=chatbot.rag_config.knowledge_base
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/chatbots/{chatbot_id}", response_model=dict)
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
        print("❌ Chatbot update error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Chat (RAG + LLM)
# ---------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Generate AI response using RAG"""
    try:
        print(f"\n🔄 Chat request: {request.chatbot_id} - {request.message[:50]}...")
        
        # 1. Retrieve context from vector DB
        context = await rag_service.query_knowledge_base(
            chatbot_id=request.chatbot_id,
            query=request.message
        )

        # 2. Generate response with system prompt
        response = await rag_service.generate_response(
            query=request.message,
            context=context,
            conversation_history=request.conversation_history,
            system_prompt=getattr(request, 'system_prompt', None)
        )

        print(f"✅ Chat response generated: {response.get('message', '')[:50]}...")
        return ChatResponse(**response)

    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Notifications
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Behavior Analysis
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Knowledge Upload
# ---------------------------------------------------------

@router.post("/knowledge/upload", response_model=dict)
async def upload_knowledge_file(
    chatbot_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process knowledge base files"""
    try:
        if file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")

        content = await file.read()

        if file.content_type == "application/pdf":
            processed_content = content.decode("utf-8", errors="ignore")
        elif file.content_type == "text/plain":
            processed_content = content.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        knowledge_item = KnowledgeBaseItem(
            type="text",
            content=processed_content,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size
            }
        )

        result = await rag_service.create_knowledge_base(
            chatbot_id=chatbot_id,
            knowledge_items=[knowledge_item]
        )

        return {"success": True, "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "EngageX AI"}
