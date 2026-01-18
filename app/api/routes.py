from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List
import json

from app.models.schemas import (
    ChatbotCreate,
    ChatbotUpdate,
    ChatRequest,
    ChatResponse,
    NotificationOptimization,
    OptimizedNotification,
    BehaviorAnalysis,
    BehaviorInsights,
    KnowledgeBaseItem,
    KnowledgeBaseType
)
from app.services.rag_service import rag_service
from app.services.ai_service import ai_service
from app.services.rag_data_service import rag_data_service

router = APIRouter()


# ---------------------------------------------------------
# Chatbot Management
# ---------------------------------------------------------

@router.post("/chatbots", response_model=dict)
async def create_chatbot(chatbot: ChatbotCreate):
    """Create a new chatbot with RAG knowledge base"""
    chatbot_id = chatbot.chatbot_id
    try:
        # Handle structured data for menus
        # for item in chatbot.rag_config.knowledge_base:
        #     if item.type == KnowledgeBaseType.MENU and rag_data_service:
        #         try:
        #             menu_data = json.loads(item.content)
        #             await rag_data_service.store_structured_data(chatbot_id,item._id, menu_data)
        #             print(f"✅ Stored structured menu data for {chatbot_id}")
        #         except Exception as e:
        #             print(f"⚠️ Failed to store structured menu data: {e}")

        # Create initial knowledge base (Vector DB)
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
            # Re-create knowledge base entirely (PUT semantic) or Append?
            # Usually PUT replaces resource. So create_knowledge_base is correct for replacing config.
            # But if we have huge files uploaded before, they might be lost if we only send config?
            # The ChatbotUpdate schema has rag_config as Optional.
            # If provided, we update RAG. 
            
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
# ... (Keep existing Chat methods) ...
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
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Notifications and Behavior (Keep existing)
# ---------------------------------------------------------
@router.post("/notifications/optimize", response_model=OptimizedNotification)
async def optimize_notification(request: NotificationOptimization):
    return await ai_service.optimize_notification(
        request.user_id, request.customer_phone, request.template,
        request.business_context, request.conversation_history
    )

@router.post("/behavior/analyze", response_model=BehaviorInsights)
async def analyze_behavior(request: BehaviorAnalysis):
    return await ai_service.analyze_customer_behavior(request.user_id, request.days_back)


# ---------------------------------------------------------
# Menu Parsing (New)
# ---------------------------------------------------------

@router.post("/menu/parse", response_model=dict)
async def parse_menu_file(file: UploadFile = File(...)):
    """Parse menu file and return structured data for verification"""
    try:
        content = await file.read()
        text_content = content.decode("utf-8", errors="ignore")
        
        # Use RAG service to parse
        menu_data = await rag_service.parse_menu_content(text_content)
        
        return {
            "success": True, 
            "data": menu_data
        }
    except Exception as e:
        print(f"❌ Menu parse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------
# Knowledge Upload
# ---------------------------------------------------------

@router.post("/knowledge/upload", response_model=dict)
async def upload_knowledge_file(
    chatbot_id: str = Form(...),
    file_type: str = Form("text"),
    file: UploadFile = File(...)
):
    """Upload and process knowledge base files"""
    try:
        if file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")

        content = await file.read()

        if file.content_type == "application/pdf":
            # In a real app, use PyPDF2 or similar. 
            # For this demo, assuming text or handling raw bytes if parser supports it.
            # RAGService's split_text expects string.
            processed_content = content.decode("utf-8", errors="ignore") 
        else:
            processed_content = content.decode("utf-8", errors="ignore")

        # PARSE MENU IF TYPE IS MENU
        if file_type == "menu":
            print(f"🍽 Parsing menu for chatbot {chatbot_id}...")
            menu_data = await rag_service.parse_menu_content(processed_content)
            
            # Store structured data
            if rag_data_service:
                await rag_data_service.store_structured_data(chatbot_id, menu_data)
            
            # Also append structured text to RAG (for searching)
            # The rag_data_service stores it, but we also want it in the Vector DB?
            # Yes, let's add the raw text or the parsed text representation.
            # Using parsed text representation is better.
            processed_content = json.dumps(menu_data, indent=2)

        knowledge_item = KnowledgeBaseItem(
            type=file_type,
            content=processed_content,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size
            }
        )

        # Append to existing knowledge base
        result = await rag_service.add_knowledge_items(
            chatbot_id=chatbot_id,
            knowledge_items=[knowledge_item]
        )

        return {"success": True, "data": result}

    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "EngageX AI"}
