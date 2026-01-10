import os
import json
import asyncio
from typing import List, Dict, Any

import faiss
from google import genai
from google.genai.types import GenerateContentConfig
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.models.schemas import KnowledgeBaseItem
from app.services.rag_data_service import rag_data_service


class RAGService:
    def __init__(self):
        # Ensure directories exist
        os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # Initialize embedding model
        print("🔄 Loading embedding model:", settings.EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Initialize Gemini (NEW SDK)
        print("🔄 Initializing Google Gemini")
        print("Using API key present:", bool(settings.GOOGLE_API_KEY))

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)

        # ✅ USE A VALID MODEL
        self.model_name = "gemini-2.5-flash"

        print("✅ RAG Service initialized (FAISS + Gemini)\n")

    # ------------------------------------------------------------------
    # KNOWLEDGE BASE CREATION
    # ------------------------------------------------------------------
    async def create_knowledge_base(
        self,
        chatbot_id: str,
        knowledge_items: List[KnowledgeBaseItem]
    ) -> Dict[str, Any]:
        try:
            print("\n========== KB CREATION START ==========")
            print("Chatbot ID:", chatbot_id)

            documents: List[str] = []

            for item in knowledge_items:
                print("Processing KB item type:", item.type)

                if item.type == "pdf":
                    documents.append(item.content)
                else:
                    documents.extend(self._split_text(item.content))

            if not documents:
                print("❌ No documents generated")
                return {"status": "error", "message": "No documents to process"}

            print("Total document chunks:", len(documents))

            embeddings = self.embedding_model.encode(documents)
            dimension = embeddings.shape[1]

            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype("float32"))

            index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
            docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")

            faiss.write_index(index, index_path)
            with open(docs_path, "w") as f:
                json.dump(documents, f)

            print("✅ KB saved")
            print("========== KB CREATION END ==========\n")

            return {
                "status": "success",
                "document_count": len(documents),
                "embedding_dimension": dimension,
            }

        except Exception as e:
            print("❌ Knowledge base creation error:", e)
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # QUERY KNOWLEDGE BASE
    # ------------------------------------------------------------------
    async def query_knowledge_base(
        self,
        chatbot_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            print("\n========== RAG QUERY START ==========")
            print("Query:", query)

            index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
            docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")

            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                print("❌ FAISS index or docs missing")
                return []

            index = faiss.read_index(index_path)
            with open(docs_path, "r") as f:
                documents = json.load(f)

            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding.astype("float32"), top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0.2:
                    results.append({
                        "content": documents[idx],
                        "similarity": float(score)
                    })

            print("Total RAG matches:", len(results))
            print("========== RAG QUERY END ==========\n")
            return results

        except Exception as e:
            print("❌ Query error:", e)
            return []

    # ------------------------------------------------------------------
    # GENERATE RESPONSE (GEMINI + RAG)
    # ------------------------------------------------------------------
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
        system_prompt: str = None,
    ) -> Dict[str, Any]:

        try:
            print("\n========== RESPONSE GENERATION START ==========")

            # Build context from documents
            context_str = "\n\n".join(
                doc["content"][:400] for doc in context[:3]
            )

            history_str = ""
            for msg in conversation_history[-3:]:
                role = "Customer" if msg["sender"] == "customer" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

            base_prompt = """
You are a restaurant assistant for "Kerala Spice House".

CRITICAL RULES:
- Answer using ONLY the provided CONTEXT.
- If information is not in the CONTEXT, reply exactly:
"I'm sorry, I don't have that information right now."
- Do NOT invent menu items or prices.
- Keep responses short and polite (1–3 sentences).
"""

            if system_prompt:
                base_prompt += f"\n{system_prompt}"

            final_prompt = f"""
{base_prompt}

CONTEXT:
{context_str}

CHAT HISTORY:
{history_str}

USER QUESTION:
{query}

ANSWER:
"""

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=final_prompt,
                config=GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                    top_p=0.8,
                )
            )

            # ✅ CORRECT RESPONSE ACCESS
            text = response.candidates[0].content.parts[0].text.strip()

            print("✅ Gemini response:", text)
            print("========== RESPONSE GENERATION END ==========\n")

            return {
                "message": text,
                "confidence": 0.9 if context_str else 0.6,
                "tokens_used": len(text.split()),
                "processing_time": 0.0,
                "context_used": bool(context_str),
            }

        except Exception as e:
            print("❌ Gemini generation error:", e)
            return {
                "message": "I'm sorry, I don't have that information right now.",
                "confidence": 0.4,
                "tokens_used": 10,
                "processing_time": 0.0,
                "context_used": False,
            }

    # ------------------------------------------------------------------
    # TEXT SPLITTER
    # ------------------------------------------------------------------
    def _split_text(
        self,
        text: str,
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> List[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]


# Singleton
rag_service = RAGService()
