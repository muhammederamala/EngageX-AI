# EngageX AI Service

FastAPI microservice for AI-powered features including RAG chatbots, behavior analysis, and notification optimization.

## Features

- **RAG Chatbots**: Vector-based knowledge retrieval with ChromaDB
- **Behavior Analysis**: Customer interaction pattern analysis
- **Notification Optimization**: AI-powered content and timing optimization
- **File Processing**: PDF and text document processing
- **Vector Embeddings**: OpenAI embeddings for semantic search

## Tech Stack

- **Framework**: FastAPI
- **Vector DB**: ChromaDB
- **AI**: OpenAI GPT-4, LangChain
- **Database**: MongoDB (Motor)
- **Cache**: Redis
- **ML**: scikit-learn, pandas, numpy

## Quick Start

### Prerequisites

- Python 3.11+
- MongoDB
- Redis
- OpenAI API Key

### Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the service:
```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
python main.py
```

## API Endpoints

### Chatbots
- `POST /api/v1/chatbots` - Create chatbot with knowledge base
- `PUT /api/v1/chatbots/{chatbot_id}` - Update chatbot
- `POST /api/v1/chat` - Generate AI response
- `POST /api/v1/knowledge/upload` - Upload knowledge files

### AI Services
- `POST /api/v1/notifications/optimize` - Optimize notification
- `POST /api/v1/behavior/analyze` - Analyze customer behavior

### Health
- `GET /api/v1/health` - Service health check

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `MONGODB_URI` | MongoDB connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `NODE_SERVER_URL` | Node.js server URL | Yes |

## Docker

```bash
# Build image
docker build -t engagex-ai .

# Run container
docker run -p 8000:8000 --env-file .env engagex-ai
```

## Integration

This service integrates with the Node.js server via HTTP APIs. The Node.js server handles:
- User authentication
- WhatsApp messaging
- Notification scheduling (Bull queues)
- Database operations

The AI service handles:
- Knowledge base management
- AI response generation
- Behavior analysis
- Content optimization