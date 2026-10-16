import re
import uuid

import math

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from settings import (
    get_gemini_api_key,
    get_gemini_embedding_model,
    get_gemini_llm_model,
    get_pinecone_dimension,
    get_pinecone_api_key,
    get_pinecone_cloud,
    get_pinecone_index_name,
    get_pinecone_metric,
    get_pinecone_namespace,
    get_pinecone_region,
)

# =========================================================
# CONFIG
# =========================================================

SIMILARITY_THRESHOLD = 0.70
TOP_K = 8
FINAL_CONTEXT_DOCS = 3

EMBEDDING_DIMENSION = get_pinecone_dimension()
EMBEDDING_MODEL = get_gemini_embedding_model()
MODEL_NAME = get_gemini_llm_model()

gemini_client = genai.Client(api_key=get_gemini_api_key())

# =========================================================
# EMBEDDINGS (Gemini)
# =========================================================

class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper around the Gemini embedding API."""

    BATCH_SIZE = 100

    def __init__(self, model: str, dimension: int):
        self.model = model
        self.dimension = dimension

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        # Gemini only returns unit-length vectors at the full default
        # size, so normalize when using a reduced dimension.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            res = gemini_client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimension,
                ),
            )
            out.extend(self._normalize(list(e.values)) for e in res.embeddings)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


embeddings = GeminiEmbeddings(EMBEDDING_MODEL, EMBEDDING_DIMENSION)

print(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION} dims)")
print(f"LLM model: {MODEL_NAME}")

# =========================================================
# VECTORSTORE CACHE
# =========================================================

_vectorstore: PineconeVectorStore | None = None

# =========================================================
# TEXT SPLITTER
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
)

# =========================================================
# PROMPT
# =========================================================

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are **MFD AI Tutor**, a precise mutual-fund education assistant for Market for Dummies (MFD).

Your purpose is to help users understand mutual funds, SIPs, NAV, returns, risk, diversification, and other financial concepts using the provided context from MFD's YouTube lecture transcripts.

You MUST answer ONLY from the provided context.

**Rules:**

1. Use only information contained in the retrieved lecture passages and accompanying metadata. Do not use external knowledge or unsupported assumptions.
2. If partial information exists, provide the supported information and clearly state what is missing.
3. Reasoning over concepts, numerical values, examples, calculations, and relationships explicitly supported by the context is allowed.
4. Do NOT fabricate facts, numerical values, examples attributed to lecturers, lecture titles, timestamps, or source links.
5. Keep answers concise, clear, and beginner-friendly. Explain technical terms when the context supports the explanation.
6. When relevant source metadata is available, cite the lecture title and timestamp. Include a clickable video link when a valid URL is provided.
7. Distinguish between information explicitly stated in the lectures and conclusions derived through straightforward reasoning from that information.
8. Do not provide investment recommendations, fund selections, buy/sell signals, guaranteed return predictions, or personalized financial advice. MFD is an educational paper-trading platform, not an investment advisory service.
9. If the context does not contain enough information to answer the question, do not guess or fill gaps using your pretrained knowledge.
10. If absolutely no relevant information exists in the provided context, reply exactly:

Answer not found in MFD lecture materials please try a different query

Context:
{context}

Question:
{question}

Answer:
""",
)

# =========================================================
# QUERY CLEANING
# =========================================================

def clean_query(query: str) -> str:

    query = query.lower().strip()

    query = re.sub(r"\s+", " ", query)

    return query

# =========================================================
# PINECONE HELPERS
# =========================================================

def _pinecone_index_host(
    pc: Pinecone,
    index_name: str,
) -> str:

    desc = pc.describe_index(index_name)

    host = getattr(desc, "host", None)

    if host:
        return host

    if isinstance(desc, dict):
        return desc.get("host")

    raise RuntimeError(
        f"Unable to determine host for index: {index_name}"
    )


def _ensure_pinecone_index(
    pc: Pinecone,
    index_name: str,
) -> None:

    metric = get_pinecone_metric()
    cloud = get_pinecone_cloud()
    region = get_pinecone_region()

    existing_indexes = set(
        pc.list_indexes().names()
    )

    if index_name in existing_indexes:
        return

    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric=metric,
        spec=ServerlessSpec(
            cloud=cloud,
            region=region,
        ),
    )

# =========================================================
# VECTORSTORE
# =========================================================

def _get_vectorstore() -> PineconeVectorStore:

    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    api_key = get_pinecone_api_key()

    index_name = get_pinecone_index_name()

    namespace = get_pinecone_namespace()

    pc = Pinecone(api_key=api_key)

    _ensure_pinecone_index(
        pc,
        index_name,
    )

    host = _pinecone_index_host(
        pc,
        index_name,
    )

    _vectorstore = PineconeVectorStore(
        index_name=index_name,
        host=host,
        pinecone_api_key=api_key,
        namespace=namespace,
        embedding=embeddings,
    )

    return _vectorstore

# =========================================================
# INDEX DOCUMENT
# =========================================================

def index_admin_document(
    text: str,
    link: str,
) -> dict:

    print("\n=========== RAW DOCUMENT TEXT ===========\n")
    print(text)

    text = text.strip()

    if not text:
        raise ValueError("No valid text found")

    # Small docs remain whole
    if len(text) < 2500:
        chunks = [text]

    else:
        chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError(
            "No valid text found"
        )

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    metadatas = [
        {
            "link": link,
        }
        for _ in chunks
    ]

    vectorstore = _get_vectorstore()

    vectorstore.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids,
    )

    print("\n=========== CHUNKS INDEXED ===========")

    for i, chunk in enumerate(chunks):

        print(f"\n--- CHUNK {i+1} ---")
        print(chunk)

    return {
        "message": "Document indexed successfully",
        "chunks_added": len(chunks),
        "link": link,
    }

# =========================================================
# ANSWER QUERY
# =========================================================

def answer_user_query(query: str) -> dict:

    vectorstore = _get_vectorstore()

    cleaned_query = clean_query(query)

    print("\n=========== ORIGINAL QUERY ===========")
    print(query)

    print("\n=========== CLEANED QUERY ===========")
    print(cleaned_query)

    # =====================================================
    # RETRIEVAL
    # =====================================================

    results = vectorstore.similarity_search_with_score(
        cleaned_query,
        k=TOP_K,
    )

    print("\n=========== RETRIEVED RESULTS ===========")

    docs = []

    for i, (doc, score) in enumerate(results):

        print(f"\n--- RESULT {i+1} ---")
        print(f"SIMILARITY SCORE: {score}")

        print(doc.page_content)

        # cosine similarity
        # higher = better
        if score >= SIMILARITY_THRESHOLD:
            docs.append(doc)

    print("\n=========== FILTERED DOC COUNT ===========")
    print(len(docs))

    # fallback:
    # if threshold removed everything,
    # keep top 2 anyway
    if not docs and results:

        print("\n=========== USING FALLBACK DOCS ===========")

        docs = [
            doc
            for doc, _ in results[:2]
        ]

    if not docs:

        return {
            "ans": (
                "Answer not found in uploaded "
                "documents please try a different query"
            ),
            "link": None,
            "sources": [],
        }

    # =====================================================
    # LIMIT CONTEXT
    # =====================================================

    docs = docs[:FINAL_CONTEXT_DOCS]

    # =====================================================
    # CONTEXT BUILDING
    # =====================================================

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    print("\n=========== FINAL CONTEXT ===========")
    print(context)

    final_prompt = qa_prompt.format(
        context=context,
        question=query,
    )

    # =====================================================
    # LLM CALL
    # =====================================================

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=final_prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a precise retrieval-based "
                "college assistant."
            ),
            temperature=0.05,
            max_output_tokens=1024,
        ),
    )

    answer = (response.text or "").strip()

    print("\n=========== FINAL ANSWER ===========")
    print(answer)

    # =====================================================
    # CLEANUP
    # =====================================================

    if not answer:

        answer = (
            "Answer not found in uploaded "
            "documents please try a different query"
        )

    reference_link = (
        docs[0].metadata or {}
    ).get("link")

    return {
        "ans": answer,
        "link": reference_link,
        # "sources": [
        #     {
        #         "content": doc.page_content,
        #         "metadata": doc.metadata,
        #     }
        #     for doc in docs
        # ],
    }
