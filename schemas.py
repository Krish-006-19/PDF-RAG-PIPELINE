from pydantic import BaseModel

class AdminDocument(BaseModel):
    text: str


class UserQuery(BaseModel):
    query: str