from fastapi import FastAPI, HTTPException

from rag_service import answer_user_query, index_admin_document
from schemas import AdminDocument, UserQuery
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    # "http://localhost:5173",
    # "http://127.0.0.1:5173",
    "https://market-for-dummies.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

# @app.post("/transcript")
# def upload_document(payload: AdminDocument):
#     try:
#         return index_admin_document(text=payload.text, link=payload.link)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def ask_question(payload: UserQuery):
    try:
        return answer_user_query(query=payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
