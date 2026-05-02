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
    get_pinecone_dimension,
    get_pinecone_index_name,
    get_pinecone_metric,
    get_pinecone_namespace,
    get_pinecone_region,
)

HF_TOKEN = get_hf_token()

# Embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)


_vectorstore: PineconeVectorStore | None = None


def _pinecone_index_host(pc: Pinecone, index_name: str) -> str:
    desc = pc.describe_index(index_name)
    host = getattr(desc, "host", None)
    if host:
        return host
    if isinstance(desc, dict) and desc.get("host"):
        return str(desc["host"])
    status = getattr(desc, "status", None)
    raise RuntimeError(
        "Unable to determine Pinecone index host (index may still be provisioning). "
        f"index_name={index_name!r} status={status!r}"
    )


def _ensure_pinecone_index(pc: Pinecone, index_name: str) -> None:
    metric = get_pinecone_metric()
    dimension = get_pinecone_dimension()
    cloud = get_pinecone_cloud()
    region = get_pinecone_region()

    try:
        names = set(pc.list_indexes().names())
    except Exception:
        indexes = pc.list_indexes()
        names = set(getattr(indexes, "names", lambda: [])())

    if index_name in names:
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def _get_vectorstore() -> PineconeVectorStore:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    api_key = get_pinecone_api_key()
    index_name = get_pinecone_index_name()
    namespace = get_pinecone_namespace()

    pc = Pinecone(api_key=api_key)
    _ensure_pinecone_index(pc, index_name)
    host = _pinecone_index_host(pc, index_name)

    _vectorstore = PineconeVectorStore(
        index_name=index_name,
        host=host,
        pinecone_api_key=api_key,
        namespace=namespace,
        embedding=embeddings,
    )
    return _vectorstore

# Hugging Face
llm_client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)

# Prompt
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a strict retrieval assistant.

Rules:
Dont even try to answer questions that are out of the scope of the documents, however if you encounter such a query then simply reply like this and then simply stop, no need to follow the rest of the prompt:
Answer not found in uploaded documents please try a different query
1. Use only the provided context.
2. Do not use outside knowledge.
3. If the answer is not explicitly present in the context, reply exactly:
Answer not found in uploaded documents please try a different query
4. Do not guess, infer, or partially answer.
Context:
{context}

Question:
{question}

Answer:
""",
)


def index_admin_document(text: str, link: str) -> dict:
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError("No valid text found")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"link": link} for _ in chunks]

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


def answer_user_query(query: str) -> dict:
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    docs = retriever.invoke(query)

    if not docs:
        return {
            "ans": "Answer not found in uploaded documents please try a different query",
            "link": None,
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = qa_prompt.format(
        context=context,
        question=query,
    )

    response = llm_client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {
                "role": "system",
                "content": "You answer only using the supplied document context."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    reference_link = (docs[0].metadata or {}).get("link")

    return {
        "ans": answer,
        "link": reference_link,
    }