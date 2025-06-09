from fastapi import FastAPI, Request
from pydantic import BaseModel
from backend/rag_utils import retrieve_relevant_chunks, ask_with_context

app = FastAPI()

class ChatRequest(BaseModel):
    question: str


@app.post("/ask")
def chat_endpoint(req: ChatRequest):
    context_chunks = retrieve_relevant_chunks(req.question, "avengers-bot")
    response = ask_with_context(req.question, context_chunks)
    return {"answer": response}
