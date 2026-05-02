import os
from dotenv import load_dotenv
load_dotenv()

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN not found in .env")
    return token


def get_pinecone_api_key() -> str:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env")
    return api_key


def get_pinecone_index_name() -> str:
    return os.getenv("PINECONE_INDEX_NAME", "admin-docs")


def get_pinecone_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", "admin_docs")


def get_pinecone_cloud() -> str:
    return os.getenv("PINECONE_CLOUD", "aws")


def get_pinecone_region() -> str:
    return os.getenv("PINECONE_REGION", "us-east-1")


def get_pinecone_dimension() -> int:
    raw = os.getenv("PINECONE_DIMENSION", "384")
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError("PINECONE_DIMENSION must be an integer") from e


def get_pinecone_metric() -> str:
    # Pinecone expects one of: 'cosine', 'dotproduct', 'euclidean'
    return os.getenv("PINECONE_METRIC", "cosine")