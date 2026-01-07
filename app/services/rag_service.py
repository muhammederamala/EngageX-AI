import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
import asyncio
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
        
        print("Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.HUGGINGFACE_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(settings.HUGGINGFACE_MODEL)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("RAG Service initialized with FAISS + DialoGPT")

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

    async def query_knowledge_base(self, chatbot_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
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
                if idx < len(documents) and score > 0.3:  # Similarity threshold
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
        """Generate AI response using DialoGPT with RAG context"""
        try:
            # Build context string from retrieved documents
            context_str = "\n".join([doc["content"][:200] for doc in context[:2]])
            
            # Build conversation history
            history_str = ""
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages
                    role = "Human" if msg.get("sender") == "customer" else "Assistant"
                    history_str += f"{role}: {msg.get('content', '')}\n"
            
            # Create prompt with context and personality
            tone = personality.get("tone", "friendly") if personality else "friendly"
            
            if context_str:
                prompt = f"Context: {context_str}\n\n{history_str}Human: {query}\nAssistant (respond in a {tone} tone):"
            else:
                prompt = f"{history_str}Human: {query}\nAssistant (respond in a {tone} tone):"
            
            # Generate response using DialoGPT
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            response_text = response[len(prompt):].strip()
            
            # Clean up response
            if not response_text:
                if context_str:
                    response_text = f"Based on the information I have: {context_str[:100]}..."
                else:
                    response_text = "I'm here to help! Could you provide more details?"
            
            # Calculate confidence based on context availability
            confidence = 0.8 if context_str else 0.6
            
            return {
                "message": response_text[:300],  # Limit response length
                "confidence": confidence,
                "tokens_used": len(outputs[0]),
                "processing_time": 0,
                "context_used": len(context) > 0
            }
            
        except Exception as e:
            print(f"Generation error: {e}")
            return {
                "message": "I'm having trouble processing your request right now. Please try again.",
                "confidence": 0,
                "tokens_used": 0,
                "processing_time": 0,
                "context_used": False
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