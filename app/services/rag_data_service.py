from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

class RAGDataService:
    def __init__(self, mongodb_uri: str):
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client.engagex
        self.rag_collection = self.db.rag_data
        
    async def store_structured_data(self, chatbot_id: str, data: Dict[str, Any]) -> bool:
        """Store structured RAG data with metadata for embedding search"""
        try:
            # Clear existing data for this chatbot
            await self.rag_collection.delete_many({"chatbot_id": chatbot_id})
            
            documents = []
            
            # Process restaurant info
            if "restaurant" in data:
                restaurant = data["restaurant"]
                documents.append({
                    "chatbot_id": chatbot_id,
                    "type": "restaurant_info",
                    "id": "restaurant_main",
                    "searchable_text": f"{restaurant.get('name', '')} {restaurant.get('type', '')} {restaurant.get('description', '')}",
                    "data": restaurant,
                    "created_at": datetime.utcnow()
                })
            
            # Process menu items
            if "menu" in data:
                menu = data["menu"]
                for category, items in menu.items():
                    for item in items:
                        searchable_text = f"{item.get('name', '')} {item.get('description', '')} {category} {' '.join(item.get('dietary', []))}"
                        documents.append({
                            "chatbot_id": chatbot_id,
                            "type": "menu_item",
                            "category": category,
                            "id": item.get("id", ""),
                            "searchable_text": searchable_text,
                            "data": item,
                            "created_at": datetime.utcnow()
                        })
            
            if documents:
                await self.rag_collection.insert_many(documents)
                print(f"✅ Stored {len(documents)} structured documents for chatbot {chatbot_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error storing structured data: {e}")
            return False
    
    async def get_items_by_ids(self, chatbot_id: str, item_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve full item data by IDs"""
        try:
            cursor = self.rag_collection.find({
                "chatbot_id": chatbot_id,
                "id": {"$in": item_ids}
            })
            
            items = []
            async for doc in cursor:
                items.append(doc["data"])
            
            return items
            
        except Exception as e:
            print(f"❌ Error retrieving items: {e}")
            return []
    
    async def get_all_searchable_texts(self, chatbot_id: str) -> List[Dict[str, Any]]:
        """Get all searchable texts with their IDs for embedding"""
        try:
            cursor = self.rag_collection.find(
                {"chatbot_id": chatbot_id},
                {"id": 1, "searchable_text": 1, "type": 1, "category": 1}
            )
            
            texts = []
            async for doc in cursor:
                texts.append({
                    "id": doc["id"],
                    "text": doc["searchable_text"],
                    "type": doc.get("type", ""),
                    "category": doc.get("category", "")
                })
            
            return texts
            
        except Exception as e:
            print(f"❌ Error getting searchable texts: {e}")
            return []

rag_data_service = None

async def init_rag_data_service(mongodb_uri: str):
    global rag_data_service
    rag_data_service = RAGDataService(mongodb_uri)
    return rag_data_service