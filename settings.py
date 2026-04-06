import os
from dotenv import load_dotenv
load_dotenv()

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN not found in .env")
    return token