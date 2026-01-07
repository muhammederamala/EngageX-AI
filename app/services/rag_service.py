import os
import json
import pickle
from typing import List, Dict, Any
import asyncio

from app.core.config import settings
from app.models.schemas import KnowledgeBaseItem

class RAGService:
    def __init__(self):
        # Ensure directories exist
        os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        print("RAG Service initialized (simplified version)")

    async def create_knowledge_base(self, chatbot_id: str, knowledge_items: List[KnowledgeBaseItem]) -> Dict[str, Any]:
        """Create or update knowledge base for a chatbot"""
        try:
            documents = []
            
            for item in knowledge_items:
                if item.type == "pdf":
                    # Simple PDF processing - just store the content
                    documents.append(f"PDF Content: {item.content[:500]}")
                else:
                    # Process text content
                    chunks = self._split_text(item.content)
                    documents.extend(chunks)
            
            if not documents:
                return {"status": "error", "message": "No documents to process"}
            
            # Save documents to file (simple storage)
            documents_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")
            with open(documents_path, 'w') as f:
                json.dump(documents, f)
            
            return {
                "status": "success",
                "document_count": len(documents),
                "index_path": documents_path
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def query_knowledge_base(self, chatbot_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query the knowledge base for relevant information"""
        try:
            documents_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")
            
            if not os.path.exists(documents_path):
                return []
            
            # Load documents
            with open(documents_path, 'r') as f:
                documents = json.load(f)
            
            # Simple keyword matching
            query_lower = query.lower()
            results = []
            
            for i, doc in enumerate(documents):
                if any(word in doc.lower() for word in query_lower.split()):
                    results.append({
                        "content": doc,
                        "metadata": {"chunk_id": i},
                        "similarity": 0.8
                    })
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Query error: {e}")
            return []

    async def generate_response(self, query: str, context: List[Dict[str, Any]], 
                              conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate AI response"""
        try:
            # Build context string
            context_str = "\n".join([doc["content"] for doc in context[:2]])
            
            # Simple response generation
            if context_str:
                response_text = f"Based on the information I have, I can help you with '{query}'. Here's what I found: {context_str[:100]}..."
            else:
                response_text = f"I understand you're asking about '{query}'. I'm here to help! Could you provide more details?"
            
            return {
                "message": response_text[:200],
                "confidence": 0.8 if context_str else 0.5,
                "tokens_used": len(response_text.split()),
                "processing_time": 0
            }
            
        except Exception as e:
            print(f"Generation error: {e}")
            return {
                "message": "I'm having trouble processing your request right now.",
                "confidence": 0,
                "tokens_used": 0,
                "processing_time": 0
            }

    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks if chunks else [text]

rag_service = RAGService()