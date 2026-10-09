from fastapi import FastAPI
from pydantic import BaseModel
from rag_test import retriver, rag_agent

app = FastAPI()


class Query(BaseModel):
    question: str
    role: str = "general"


@app.post("/chat")
def chat(q: Query):
    docs = retriver(q.question, q.role)
    return {"answer": rag_agent(q.question, docs)}
