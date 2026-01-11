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
        try:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            print("✅ Embedding model loaded")
        except Exception as e:
            print("❌ Embedding model failed:", e)
            self.embedding_model = None

        # -------------------------------------------------
        # HUGGING FACE LLM
        # -------------------------------------------------
        self.tokenizer = None
        self.model = None

        if self.llm_provider == "hf":
            try:
                print("🔄 Loading HuggingFace model: distilgpt2")
                self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                self.model.eval()
                print("✅ HuggingFace model loaded")
            except Exception as e:
                print("❌ HF model failed:", e)
                self.tokenizer = None
                self.model = None
                self.llm_provider = "simple"

        # -------------------------------------------------
        # GEMINI LLM
        # -------------------------------------------------
        self.gemini_client = None
        self.gemini_model = "gemini-2.5-flash"

        if self.llm_provider == "gemini":
            try:
                print("🔄 Initializing Gemini client")
                self.gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
                
                # List available models
                try:
                    models = self.gemini_client.models.list()
                    print("Available models:")
                    for model in models:
                        print(f"  - {model.name}")
                except Exception as e:
                    print(f"Could not list models: {e}")
                
                print("✅ Gemini client ready")
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
            if item.type == "pdf":
                documents.append(item.content)
            else:
                documents.extend(self._split_text(item.content))

        if not documents:
            return {"status": "error", "message": "No documents"}

        embeddings = self.embedding_model.encode(documents)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))

        index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
        docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")

        faiss.write_index(index, index_path)
        with open(docs_path, "w") as f:
            json.dump(documents, f)

        return {
            "status": "success",
            "document_count": len(documents),
            "embedding_dimension": embeddings.shape[1],
        }

    # ------------------------------------------------------------------
    # QUERY KB
    # ------------------------------------------------------------------
    async def query_knowledge_base(
        self,
        chatbot_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.index")
        docs_path = os.path.join(settings.FAISS_INDEX_PATH, f"{chatbot_id}.json")

        if not os.path.exists(index_path):
            return []

        index = faiss.read_index(index_path)
        documents = json.load(open(docs_path))

        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding.astype("float32"), top_k)

        return [
            {"content": documents[idx], "similarity": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0 and score > 0.2
        ]

    # ------------------------------------------------------------------
    # GENERATE RESPONSE
    # ------------------------------------------------------------------
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
        system_prompt: str = None,
    ) -> Dict[str, Any]:

        context_str = "\n\n".join(doc["content"][:400] for doc in context[:3])

        if self.llm_provider == "hf":
            return await self._generate_hf(query, context_str)

        if self.llm_provider == "gemini":
            return await self._generate_gemini(query, context_str, conversation_history, system_prompt)

        return self._generate_simple(context_str)

    # ------------------------------------------------------------------
    # HUGGING FACE
    # ------------------------------------------------------------------
    async def _generate_hf(self, query: str, context_str: str) -> Dict[str, Any]:
        if not self.model or not self.tokenizer:
            return self._generate_simple(context_str)

        prompt = f"""
You are a helpful assistant.
Answer ONLY from the context.

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:
""".strip()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = await asyncio.to_thread(
                self.model.generate,
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = text.split("ANSWER:")[-1].strip()

        if len(answer) < 5:
            answer = "I'm sorry, I don't have that information right now."

        return {
            "message": answer,
            "confidence": 0.75,
            "tokens_used": len(answer.split()),
            "processing_time": 0.0,
            "context_used": bool(context_str),
        }

    # ------------------------------------------------------------------
    # GEMINI
    # ------------------------------------------------------------------
    async def _generate_gemini(
        self,
        query: str,
        context_str: str,
        conversation_history: List[Dict[str, str]],
        system_prompt: str
    ) -> Dict[str, Any]:

        history = "\n".join(
            f"{'User' if m['sender']=='customer' else 'Assistant'}: {m['content']}"
            for m in conversation_history[-3:]
        )

        base_prompt = """
You are an AI assistant.
Rules:
- Use ONLY the context
- If answer not in context say:
"I'm sorry, I don't have that information right now."
"""

        if system_prompt:
            base_prompt += f"\n{system_prompt}"

        prompt = f"""
                {base_prompt}

                CONTEXT:
                {context_str}

                CHAT HISTORY:
                {history}

                QUESTION:
                {query}

                ANSWER:
                """

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

        text = response.candidates[0].content.parts[0].text.strip()

        return {
            "message": text,
            "confidence": 0.9,
            "tokens_used": len(text.split()),
            "processing_time": 0.0,
            "context_used": bool(context_str),
        }

    # ------------------------------------------------------------------
    # SIMPLE FALLBACK
    # ------------------------------------------------------------------
    def _generate_simple(self, context_str: str) -> Dict[str, Any]:
        if context_str:
            return {
                "message": context_str.split("\n")[0],
                "confidence": 0.4,
                "tokens_used": 15,
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
    # TEXT SPLIT
    # ------------------------------------------------------------------
    def _split_text(self, text: str, chunk_size=300, overlap=50) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)
            if words[i:i + chunk_size]
        ]


# Singleton
rag_service = RAGService()
