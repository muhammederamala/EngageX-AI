import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
import asyncio
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

from app.core.config import settings
from app.models.schemas import KnowledgeBaseItem

class RAGService:
    def __init__(self):
        # Ensure directories exist
        os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Initialize models
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Initialize Gemini
        print("Initializing Google Gemini...")
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY', 'your-api-key-here'))
        self.gemini_model = genai.GenerativeModel('gemini-flash-latest')
        
        print("RAG Service initialized with FAISS + Gemini")

    async def create_knowledge_base(self, chatbot_id: str, knowledge_items: List[KnowledgeBaseItem]) -> Dict[str, Any]:
        """Create or update knowledge base for a chatbot with FAISS vector store"""
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
            
            # Generate embeddings
            print(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_model.encode(documents)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            # Save FAISS index and documents
            index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
            docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")
            
            faiss.write_index(index, index_path)
            with open(docs_path, 'w') as f:
                json.dump(documents, f)
            
            return {
                "status": "success",
                "document_count": len(documents),
                "embedding_dimension": dimension,
                "index_path": index_path
            }
            
        except Exception as e:
            print(f"Knowledge base creation error: {e}")
            return {"status": "error", "message": str(e)}

    async def query_knowledge_base(self, chatbot_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the knowledge base using FAISS vector similarity"""
        try:
            index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
            docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")
            
            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                return []
            
            # Load FAISS index and documents
            index = faiss.read_index(index_path)
            with open(docs_path, 'r') as f:
                documents = json.load(f)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search similar documents
            scores, indices = index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(documents) and score > 0.2:  # Lowered threshold
                    results.append({
                        "content": documents[idx],
                        "metadata": {"chunk_id": int(idx), "rank": i + 1},
                        "similarity": float(score)
                    })
            
            return results
            
        except Exception as e:
            print(f"Query error: {e}")
            return []

    async def generate_response(self, query: str, context: List[Dict[str, Any]], 
                              conversation_history: List[Dict[str, str]], 
                              personality: Dict[str, str] = None) -> Dict[str, Any]:
        """Generate AI response using Google Gemini with RAG context"""
        try:
            # Build context string from retrieved documents
            context_str = "\n".join([doc["content"][:400] for doc in context[:3]])
            
            # Build conversation history
            history_str = ""
            if conversation_history:
                for msg in conversation_history[-3:]:
                    role = "Customer" if msg.get("sender") == "customer" else "Assistant"
                    history_str += f"{role}: {msg.get('content', '')}\n"
            
            # Create system prompt for Gemini
            system_prompt = f"""You are a helpful assistant for Kerala Spice House, an authentic Kerala restaurant. 
            
Restaurant Information:
{context_str}

Conversation History:
{history_str}

Instructions:
- Be friendly and conversational
- Use the restaurant information to answer questions accurately
- Mention specific prices, dishes, and details when relevant
- If asked about items not on the menu, suggest similar alternatives
- Keep responses concise but informative (2-3 sentences max)
- Sound like a knowledgeable restaurant staff member

Customer Question: {query}

Response:"""
            
            # Generate response using Gemini
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                system_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=200,
                    top_p=0.8,
                )
            )
            
            response_text = response.text.strip()
            
            # Calculate confidence based on context availability
            confidence = 0.95 if context_str else 0.7
            
            return {
                "message": response_text,
                "confidence": confidence,
                "tokens_used": len(response_text.split()),
                "processing_time": 0,
                "context_used": len(context) > 0
            }
            
        except Exception as e:
            print(f"Gemini generation error: {e}")
            # Fallback to template-based response
            return self._generate_fallback_response(query, context)
    
    def _generate_fallback_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback response when Gemini fails"""
        context_str = "\n".join([doc["content"][:300] for doc in context[:2]])
        
        if 'breakfast' in query.lower():
            response = "For breakfast, try our Appam with Chicken Stew ($15) - a traditional Kerala favorite!"
        elif 'biryani' in query.lower():
            response = "We don't have biryani, but our Ghee Rice ($9) and Coconut Rice ($8) are delicious alternatives!"
        elif context_str:
            response = f"Based on our menu: {context_str[:150]}... What else would you like to know?"
        else:
            response = "I'm here to help with Kerala Spice House! What would you like to know about our authentic Kerala cuisine?"
        
        return {
            "message": response,
            "confidence": 0.8,
            "tokens_used": len(response.split()),
            "processing_time": 0,
            "context_used": len(context) > 0
        }

    def _split_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]

rag_service = RAGService()