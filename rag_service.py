# import uuid

# from huggingface_hub import InferenceClient
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pinecone import Pinecone, ServerlessSpec

# from settings import (
#     get_hf_token,
#     get_pinecone_api_key,
#     get_pinecone_cloud,
#     get_pinecone_dimension,
#     get_pinecone_index_name,
#     get_pinecone_metric,
#     get_pinecone_namespace,
#     get_pinecone_region,
# )

# HF_TOKEN = get_hf_token()

# # Embedding model
# embeddings = HuggingFaceEndpointEmbeddings(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
#     huggingfacehub_api_token=HF_TOKEN,
# )


# _vectorstore: PineconeVectorStore | None = None


# def _pinecone_index_host(pc: Pinecone, index_name: str) -> str:
#     desc = pc.describe_index(index_name)
#     host = getattr(desc, "host", None)
#     if host:
#         return host
#     if isinstance(desc, dict) and desc.get("host"):
#         return str(desc["host"])
#     status = getattr(desc, "status", None)
#     raise RuntimeError(
#         "Unable to determine Pinecone index host (index may still be provisioning). "
#         f"index_name={index_name!r} status={status!r}"
#     )


# def _ensure_pinecone_index(pc: Pinecone, index_name: str) -> None:
#     metric = get_pinecone_metric()
#     dimension = get_pinecone_dimension()
#     cloud = get_pinecone_cloud()
#     region = get_pinecone_region()

#     try:
#         names = set(pc.list_indexes().names())
#     except Exception:
#         indexes = pc.list_indexes()
#         names = set(getattr(indexes, "names", lambda: [])())

#     if index_name in names:
#         return

#     pc.create_index(
#         name=index_name,
#         dimension=dimension,
#         metric=metric,
#         spec=ServerlessSpec(cloud=cloud, region=region),
#     )


# def _get_vectorstore() -> PineconeVectorStore:
#     global _vectorstore
#     if _vectorstore is not None:
#         return _vectorstore

#     api_key = get_pinecone_api_key()
#     index_name = get_pinecone_index_name()
#     namespace = get_pinecone_namespace()

#     pc = Pinecone(api_key=api_key)
#     _ensure_pinecone_index(pc, index_name)
#     host = _pinecone_index_host(pc, index_name)

#     _vectorstore = PineconeVectorStore(
#         index_name=index_name,
#         host=host,
#         pinecone_api_key=api_key,
#         namespace=namespace,
#         embedding=embeddings,
#     )
#     return _vectorstore

# # Hugging Face
# llm_client = InferenceClient(
#     provider="featherless-ai",
#     api_key=HF_TOKEN,
# )

# # Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=100,
# )

# # Prompt
# # qa_prompt = PromptTemplate(
# #     input_variables=["context", "question"],
# #     template="""
# # You are a strict retrieval assistant.

# # Rules:
# # Dont even try to answer questions that are out of the scope of the documents, however if you encounter such a query then simply reply like this and then simply stop, no need to follow the rest of the prompt:
# # Answer not found in uploaded documents please try a different query
# # 1. Use only the provided context.
# # 2. Do not use outside knowledge.
# # 3. If the answer is not explicitly present in the context, reply exactly:
# # Answer not found in uploaded documents please try a different query
# # 4. Do not guess, infer, or partially answer.
# # Context:
# # {context}

# # Question:
# # {question}

# # Answer:
# # """,
# # )

# qa_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a college document assistant.

# Answer ONLY using the provided context.

# Rules:
# 1. Use only information present in the context.
# 2. Simple reasoning based on tables, ranges, schedules, and mappings is allowed.
# 3. Keep answers concise and accurate.
# 4. If the answer truly cannot be found in the context, reply exactly:
# Answer not found in uploaded documents please try a different query
# 5. Do not invent information.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """,
# )


# def index_admin_document(text: str, link: str) -> dict:
#     chunks = text_splitter.split_text(text)

#     if not chunks:
#         raise ValueError("No valid text found")

#     ids = [str(uuid.uuid4()) for _ in chunks]
#     metadatas = [{"link": link} for _ in chunks]

#     vectorstore = _get_vectorstore()

#     vectorstore.add_texts(
#         texts=chunks,
#         metadatas=metadatas,
#         ids=ids,
#     )

#     return {
#         "message": "Document indexed successfully",
#         "chunks_added": len(chunks),
#         "link": link,
#     }


# def answer_user_query(query: str) -> dict:
#     vectorstore = _get_vectorstore()
#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 4},
#     )

#     docs = retriever.invoke(query)

#     if not docs:
#         return {
#             "ans": "Answer not found in uploaded documents please try a different query",
#             "link": None,
#         }

#     context = "\n\n".join(doc.page_content for doc in docs)

#     final_prompt = qa_prompt.format(
#         context=context,
#         question=query,
#     )

#     response = llm_client.chat.completions.create(
#         model="mistralai/Mistral-7B-Instruct-v0.2",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You answer only using the supplied document context."
#             },
#             {
#                 "role": "user",
#                 "content": final_prompt
#             }
#         ],
#         max_tokens=512,
#     )

#     answer = response.choices[0].message.content.strip()
#     reference_link = (docs[0].metadata or {}).get("link")

#     return {
#         "ans": answer,
#         "link": reference_link,
#     }

import re
import uuid

from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from settings import (
    get_hf_token,
    get_pinecone_api_key,
    get_pinecone_cloud,
    get_pinecone_index_name,
    get_pinecone_metric,
    get_pinecone_namespace,
    get_pinecone_region,
)

HF_TOKEN = get_hf_token()

# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)

EMBEDDING_DIMENSION = len(
    embeddings.embed_query("test")
)

print(f"Embedding Dimension: {EMBEDDING_DIMENSION}")

# =========================================================
# VECTORSTORE CACHE
# =========================================================

_vectorstore: PineconeVectorStore | None = None

# =========================================================
# TEXT SPLITTER
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=250,
)

# =========================================================
# PROMPT
# =========================================================

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a college document assistant.

You MUST answer ONLY from the provided context.

Rules:
1. Use only information present in the context.
2. Simple reasoning based on tables, schedules,
   roll-number ranges, mappings, and structured
   data is allowed.
3. Do NOT use outside knowledge.
4. Keep answers concise and accurate.
5. If the answer truly does not exist in the context,
   reply exactly:
Answer not found in uploaded documents please try a different query
6. Do not fabricate information.

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
    """
    Convert noisy student queries into
    retrieval-friendly queries.
    """

    query = query.lower()

    # Extract roll number
    roll_match = re.search(r"\b\d+\b", query)

    if roll_match:
        roll = roll_match.group()
        return f"roll number {roll} lab exam timing"

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

    _ensure_pinecone_index(pc, index_name)

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
# LLM
# =========================================================

llm_client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)

# =========================================================
# INDEX DOCUMENT
# =========================================================

def index_admin_document(
    text: str,
    link: str,
) -> dict:

    print("\n=========== RAW DOCUMENT TEXT ===========\n")
    print(text)

    # Small notices/schedules should NOT be chunked
    if len(text) < 3000:
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

    # Debug similarity scores
    results = vectorstore.similarity_search_with_score(
        cleaned_query,
        k=5,
    )

    print("\n=========== RETRIEVED RESULTS ===========")

    docs = []

    for i, (doc, score) in enumerate(results):

        print(f"\n--- RESULT {i+1} ---")
        print(f"Score: {score}")
        print(doc.page_content)

        docs.append(doc)

    if not docs:
        return {
            "ans": (
                "Answer not found in uploaded documents "
                "please try a different query"
            ),
            "link": None,
            "sources": [],
        }

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = qa_prompt.format(
        context=context,
        question=query,
    )

    response = llm_client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a retrieval-based college "
                    "assistant. Answer ONLY using "
                    "the provided document context."
                ),
            },
            {
                "role": "user",
                "content": final_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=512,
    )

    answer = (
        response.choices[0]
        .message.content
        .strip()
    )

    reference_link = (
        docs[0].metadata or {}
    ).get("link")

    if (
        not answer
        or "Answer not found" in answer
    ):
        answer = (
            "Answer not found in uploaded documents "
            "please try a different query"
        )

    return {
        "ans": answer,
        "link": reference_link,
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ],
    }
