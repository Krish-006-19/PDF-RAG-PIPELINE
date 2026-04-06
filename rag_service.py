import uuid

from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import get_hf_token

HF_TOKEN = get_hf_token()

# Embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN,
)

# Vector DB
vectorstore = Chroma(
    collection_name="admin_docs",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# Hugging Face
llm_client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
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


def _maybe_persist() -> None:
    persist_fn = getattr(vectorstore, "persist", None)
    if callable(persist_fn):
        persist_fn()
        return

    client = getattr(vectorstore, "_client", None)
    client_persist = getattr(client, "persist", None)
    if callable(client_persist):
        client_persist()


def index_admin_document(text: str, link: str) -> dict:
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError("No valid text found")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"link": link} for _ in chunks]

    vectorstore.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids,
    )

    _maybe_persist()

    return {
        "message": "Document indexed successfully",
        "chunks_added": len(chunks),
        "link": link,
    }


def answer_user_query(query: str) -> dict:
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