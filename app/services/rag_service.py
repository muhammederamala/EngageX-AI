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

            # Process structured data and store in MongoDB
            for item in knowledge_items:
                print("Processing KB item type:", item.type)
                
                if item.type == "text":
                    try:
                        # Try to parse as JSON for structured data
                        structured_data = json.loads(item.content)
                        await rag_data_service.store_structured_data(chatbot_id, structured_data)
                        print("✅ Stored structured data in MongoDB")
                    except json.JSONDecodeError:
                        print("⚠️ Not JSON, treating as plain text")
                        # Fallback to plain text processing
                        pass

            # Get all searchable texts for embedding
            searchable_items = await rag_data_service.get_all_searchable_texts(chatbot_id)
            
            if not searchable_items:
                print("❌ No searchable items found")
                return {"status": "error", "message": "No searchable content"}

            # Create embeddings for searchable texts
            texts = [item["text"] for item in searchable_items]
            item_ids = [item["id"] for item in searchable_items]
            
            print(f"Creating embeddings for {len(texts)} items")
            embeddings = self.embedding_model.encode(texts)
            dimension = embeddings.shape[1]

            # Create FAISS index with item IDs
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype("float32"))

            # Save FAISS index and ID mapping
            index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
            ids_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}_ids.json")

            faiss.write_index(index, index_path)
            with open(ids_path, "w") as f:
                json.dump(item_ids, f)

            print("✅ KB saved with metadata approach")
            print("========== KB CREATION END ==========\n")

            return {
                "status": "success",
                "document_count": len(searchable_items),
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
            ids_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}_ids.json")

            if not os.path.exists(index_path) or not os.path.exists(ids_path):
                print("❌ FAISS index or IDs missing")
                return []

            # Load FAISS index and ID mapping
            index = faiss.read_index(index_path)
            with open(ids_path, "r") as f:
                item_ids = json.load(f)

            # Search for similar items
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding.astype("float32"), top_k)

            # Get matching item IDs
            matching_ids = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0.3:  # Higher threshold for structured data
                    matching_ids.append(item_ids[idx])
                    print(f"✅ Match: {item_ids[idx]} (score: {score:.3f})")

            # Retrieve full data from MongoDB
            if matching_ids:
                full_items = await rag_data_service.get_items_by_ids(chatbot_id, matching_ids)
                results = [{"content": item, "id": item.get("id", "")} for item in full_items]
                print(f"✅ Retrieved {len(results)} full items from DB")
            else:
                results = []
                print("❌ No matches found")

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

            # Build context from structured data
            context_items = []
            for doc in context[:3]:
                item_data = doc["content"]
                if "name" in item_data and "price" in item_data:
                    # Menu item
                    context_items.append(f"{item_data['name']} - ${item_data['price']} - {item_data.get('description', '')}")
                elif "name" in item_data and "type" in item_data:
                    # Restaurant info
                    context_items.append(f"{item_data['name']} - {item_data['type']} - {item_data.get('description', '')}")
                else:
                    # Fallback
                    context_items.append(str(item_data)[:200])
            
            context_str = "\n".join(context_items)

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

        return chunks


# Singleton
rag_service = RAGService()
