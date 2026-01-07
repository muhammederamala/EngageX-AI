from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None
    database = None

db = Database()

redis_client = None

async def init_db():
    """Initialize database connections"""
    global redis_client
    
    # MongoDB
    db.client = AsyncIOMotorClient(settings.MONGODB_URI)
    db.database = db.client.engagex
    
    # Redis
    redis_client = redis.from_url(settings.REDIS_URL)
    
    print("Database connections initialized")

async def close_db():
    """Close database connections"""
    if db.client:
        db.client.close()
    if redis_client:
        await redis_client.close()

def get_database():
    return db.database

def get_redis():
    return redis_client