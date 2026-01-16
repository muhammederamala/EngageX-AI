import os
import json
import asyncio
from typing import List, Dict, Any

import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from google import genai
from google.genai.types import GenerateContentConfig

from app.core.config import settings
from app.models.schemas import KnowledgeBaseItem


class RAGService:
    def __init__(self):
        # -------------------------------------------------
        # CONFIG
        # -------------------------------------------------
        self.llm_provider = settings.LLM_PROVIDER  # "hf" | "gemini" | "simple"

        os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # -------------------------------------------------
        # EMBEDDINGS
        # -------------------------------------------------
        print("🔄 Loading embedding model:", settings.EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("✅ Embedding model loaded")

        # -------------------------------------------------
        # HUGGING FACE
        # -------------------------------------------------
        self.tokenizer = None
        self.model = None

        if self.llm_provider == "hf":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                self.model.eval()
            except Exception as e:
                print("❌ HF init failed:", e)
                self.llm_provider = "simple"

        # -------------------------------------------------
        # GEMINI
        # -------------------------------------------------
        self.gemini_client = None
        self.gemini_model = "gemini-2.5-flash"

        if self.llm_provider == "gemini":
            try:
                self.gemini_client = genai.Client(
                    api_key=settings.GOOGLE_API_KEY
                )
            except Exception as e:
                print("❌ Gemini init failed:", e)
                self.llm_provider = "simple"

        print(f"✅ RAG Service initialized (LLM = {self.llm_provider})\n")

    # ------------------------------------------------------------------
    # CREATE KNOWLEDGE BASE
    # ------------------------------------------------------------------
    async def create_knowledge_base(
        self,
        chatbot_id: str,
        knowledge_items: List[KnowledgeBaseItem]
    ) -> Dict[str, Any]:

        documents: List[str] = []

        for item in knowledge_items:
            documents.extend(self._split_text(item.content))

        if not documents:
            return {"status": "empty"}

        embeddings = self.embedding_model.encode(documents)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))

        index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
        docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")

        faiss.write_index(index, index_path)
        json.dump(documents, open(docs_path, "w"))

        return {
            "status": "success",
            "document_count": len(documents),
            "embedding_dimension": embeddings.shape[1],
        }

    async def add_knowledge_items(
        self,
        chatbot_id: str,
        knowledge_items: List[KnowledgeBaseItem]
    ) -> Dict[str, Any]:
        """Append items to existing knowledge base"""
        
        # Load existing documents
        docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")
        existing_docs = []
        if os.path.exists(docs_path):
            existing_docs = json.load(open(docs_path))
            
        # Add new documents
        new_docs = []
        for item in knowledge_items:
            new_docs.extend(self._split_text(item.content))
            
        all_docs = existing_docs + new_docs
        
        if not all_docs:
            return {"status": "empty"}

        # Re-index everything (simplest approach)
        embeddings = self.embedding_model.encode(all_docs)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))

        index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
        
        faiss.write_index(index, index_path)
        json.dump(all_docs, open(docs_path, "w"))

        return {
            "status": "success",
            "document_count": len(all_docs),
            "added_count": len(new_docs)
        }

    # ------------------------------------------------------------------
    # QUERY KNOWLEDGE BASE
    # ------------------------------------------------------------------
    async def query_knowledge_base(
        self,
        chatbot_id: str,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:

        index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
        docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")

        index = faiss.read_index(index_path)
        documents = json.load(open(docs_path))

        enriched_query = f"About EngageX platform and services: {query}"

        query_embedding = self.embedding_model.encode([enriched_query])
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding.astype("float32"), top_k)

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < 0.15:
                continue
            matches.append({
                "index": int(idx),
                "similarity": float(score),
                "content": documents[idx],
            })

        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "query_embedding": query_embedding[0],
            "matches": matches,
        }

    # ------------------------------------------------------------------
    # GENERATE RESPONSE
    # ------------------------------------------------------------------
    async def generate_response(
        self,
        query: str,
        context: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
    ) -> Dict[str, Any]:

        if not system_prompt:
            raise ValueError("system_prompt is required")

        self._debug_embeddings(query, context)

        context_str = "\n\n".join(
            m["content"][:400] for m in context.get("matches", [])[:3]
        )

        if self.llm_provider == "gemini":
            return await self._generate_gemini(
                query, context_str, conversation_history, system_prompt
            )

        if self.llm_provider == "hf":
            return await self._generate_hf(
                query, context_str, system_prompt
            )

        return self._generate_simple(query, context_str)

    # ------------------------------------------------------------------
    # GEMINI
    # ------------------------------------------------------------------
    async def _generate_gemini(
        self,
        query: str,
        context_str: str,
        history: List[Dict[str, str]],
        system_prompt: str
    ) -> Dict[str, Any]:

        chat_history = "\n".join(
            f"{'User' if m['sender'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-3:]
        )

        prompt = f"""{system_prompt}

CONTEXT:
{context_str}

CHAT HISTORY:
{chat_history}

QUESTION:
{query}

ANSWER:"""

        print("Prompt length", len(prompt))

        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=self.gemini_model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200,
                top_p=0.8,
            )
        )

        print("🤖 Gemini response:", response)

        text = response.candidates[0].content.parts[0].text.strip()

        return {
            "message": self._post_process(text),
            "confidence": 0.9,
            "tokens_used": len(text.split()),
            "processing_time": 0.0,
            "context_used": bool(context_str),
        }

    # ------------------------------------------------------------------
    # HUGGING FACE
    # ------------------------------------------------------------------
    async def _generate_hf(
        self,
        query: str,
        context_str: str,
        system_prompt: str
    ) -> Dict[str, Any]:

        prompt = f"""{system_prompt}

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = await asyncio.to_thread(
                self.model.generate,
                **inputs,
                max_new_tokens=200,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = text.split("ANSWER:")[-1].strip()

        return {
            "message": self._post_process(answer),
            "confidence": 0.75,
            "tokens_used": len(answer.split()),
            "processing_time": 0.0,
            "context_used": bool(context_str),
        }

    # ------------------------------------------------------------------
    # SIMPLE FALLBACK
    # ------------------------------------------------------------------
    def _generate_simple(self, query: str, context_str: str) -> Dict[str, Any]:
        if context_str:
            return {
                "message": context_str[:200] + "...",
                "confidence": 0.3,
                "tokens_used": 20,
                "processing_time": 0.0,
                "context_used": True,
            }
        return {
            "message": "I'm sorry, I don't have that information right now.",
            "confidence": 0.2,
            "tokens_used": 10,
            "processing_time": 0.0,
            "context_used": False,
        }

    # ------------------------------------------------------------------
    # POST PROCESS (CRITICAL)
    # ------------------------------------------------------------------
    def _post_process(self, text: str) -> str:
        if not text or len(text.split()) < 5:
            return "I'm sorry, I don't have that information right now."

        text = text.strip()

        # Remove trailing ellipsis artifacts from Gemini
        text = text.rstrip(". ").rstrip("…")

        # Try to keep up to 2 complete sentences
        text = text.rstrip("…").strip()

        if not text.endswith(('.', '!', '?')):
            text += '.'
    
        return text

    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------
    def _debug_embeddings(self, query: str, context: Dict[str, Any]):
        print("\n========== 🔍 RAG DEBUG ==========")
        print("Query:", query)

        emb = context.get("query_embedding")
        matches = context.get("matches", [])

        if emb is not None:
            print("Embedding shape:", emb.shape)

        print("Valid Matches:", len(matches))

        for i, m in enumerate(matches):
            print(f"\n--- Match {i + 1} ---")
            print("Similarity:", round(m["similarity"], 4))
            print("Preview:", m["content"][:150])

        print("========== 🔍 END ==========\n")

    # ------------------------------------------------------------------
    # MENU PARSING
    # ------------------------------------------------------------------
    async def parse_menu_content(self, content: str) -> Dict[str, Any]:
        """Parse raw menu text into structured JSON using LLM"""
        
        prompt = """
        You are an advanced AI Menu Parser. Your task is to extract restaurant menu information from the provided text and convert it into a structured JSON object.

        I need you to identify:
        1. **Restaurant Info**: Name, type (cuisine), and a brief description.
        2. **Menu Items**: Grouped by Category. For EACH item, you MUST extract:
           - **name**: The full name of the dish.
           - **price**: The numeric price (clean up currency symbols).
           - **description**: Any ingredients or details described next to the item.
           - **dietary**: An array of tags like ["veg", "non-veg", "spicy", "gluten-free"] if mentioned or inferable.
           - **id**: A generated unique slug (e.g., "chicken-curry").

        JSON STRUCTURE:
        {
            "restaurant": {
                "name": "Restaurant Name",
                "type": "Cuisine",
                "description": "..."
            },
            "menu": {
                "Starters": [
                    {
                        "id": "starter-1",
                        "name": "Item Name",
                        "description": "Delicious ingredients...",
                        "price": 120.00,
                        "currency": "INR",
                        "dietary": ["veg"]
                    }
                ]
            }
        }

        IMPORTANT:
        - If a price is "120", treat it as 120.00.
        - Capture descriptions accurately.
        - Ensure valid JSON output.
        - If no categories are explicit, use "General".

        MENU TEXT:
        """ + content[:15000]  # Increased context limit
        
        try:
            if self.llm_provider == "gemini":
                response = await asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=self.gemini_model,
                    contents=prompt,
                    config=GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                text = response.candidates[0].content.parts[0].text.strip()
                
            elif self.llm_provider == "hf":
                # HF might struggle with complex JSON extraction, but we try
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                with torch.no_grad():
                    outputs = await asyncio.to_thread(
                        self.model.generate,
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.1
                    )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # primitive extraction
                if "{" in text:
                    text = text[text.find("{"):text.rfind("}")+1]
            
            else:
                # Simple mock for testing without LLM
                return {
                    "restaurant": {"name": "Unknown", "type": "General"},
                    "menu": {}
                }

            # Clean up markdown if present
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
                
            return json.loads(text)
            
        except Exception as e:
            print(f"❌ Menu parsing failed: {e}")
            # Return empty structure
            return {"restaurant": {}, "menu": {}}

    # ------------------------------------------------------------------
    # TEXT SPLIT
    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> List[str]:
        sections = text.split("--------------------------------------------------")
        return [
            f"[SECTION {i + 1}] {s.strip()}"
            for i, s in enumerate(sections)
            if len(s.split()) >= 30
        ]


# Singleton
rag_service = RAGService()
