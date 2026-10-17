import os

from dotenv import load_dotenv
from upstash_redis import Redis

# Load local .env before reading credentials. In deployment, environment variables
# provided by the hosting platform take precedence.
load_dotenv()

redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

# Redis caching is optional: the RAG pipeline still works if credentials are
# not configured. Cache helpers in rag_service.py also fail open if Redis errors.
redis = Redis(url=redis_url, token=redis_token) if redis_url and redis_token else None
