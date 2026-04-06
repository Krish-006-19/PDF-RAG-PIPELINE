from fastapi import FastAPI, HTTPException

from rag_service import answer_user_query, index_admin_document
from schemas import AdminDocument, UserQuery

app = FastAPI()

@app.post("/admin/upload")
def upload_document(payload: AdminDocument):
    try:
        return index_admin_document(text=payload.text, link=payload.link)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def ask_question(payload: UserQuery):
    try:
        return answer_user_query(query=payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))