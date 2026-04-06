from pydantic import BaseModel

class AdminDocument(BaseModel):
    text: str
    link: str


class UserQuery(BaseModel):
    query: str