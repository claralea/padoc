from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    chunk_type: str = "recursive-split"

class ChatRequest(BaseModel):
    query: str
    chunk_type: str = "recursive-split"

@app.post("/query")
def query_vector_db(req: QueryRequest):
    command = [
        "python", "cli.py",
        "--query",
        "--chunk_type", req.chunk_type,
        "--input_query", req.query
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return {"output": result.stdout.strip()}

@app.post("/chat")
def chat_with_llm(req: ChatRequest):
    command = [
        "python", "cli.py",
        "--chat",
        "--chunk_type", req.chunk_type,
        "--input_query", req.query
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return {"output": result.stdout.strip()}

@app.get("/")
def health_check():
    return {"status": "ok"}
