import os
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retrieval.rag_chain import ask, ask_stream

# Create the FastAPI app
app = FastAPI(
    title="SmartDocs AI",
    description="Ask questions from your documents",
    version="1.0"
)

# Allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: str = Header(...)):
    valid_key = os.getenv("SMARTDOCS_API_KEY")
    if x_api_key != valid_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return x_api_key

class QuestionRequest(BaseModel):
    question: str

# Route 1: Health check
@app.get("/health")
def health():
    return {"status": "SmartDocs AI is running"}

# Route 2: Ask a question
@app.post("/ask", dependencies=[Depends(verify_api_key)])
def ask_question(request: QuestionRequest):
    result = ask(request.question)
    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"]
    }

@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
def ask_stream_endpoint(request: QuestionRequest):
    return StreamingResponse(
        ask_stream(request.question),
        media_type="text/plain",
    )
