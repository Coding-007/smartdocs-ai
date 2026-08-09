import os
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from retrieval.rag_chain import ask, ask_stream
from memory.memory import get_history, clear_history

load_dotenv()

# Create the FastAPI app
app = FastAPI(
    title="SmartDocs AI",
    description="Ask questions from your documents",
    version="2.0"
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
    session_id: str = "default"

@app.get("/health")
def health():
    return {"status": "SmartDocs AI is running"}

@app.post("/ask", dependencies=[Depends(verify_api_key)])
def ask_question(request: QuestionRequest):
    result = ask(request.question, request.session_id)
    return result

@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
def ask_stream_endpoint(request: QuestionRequest):
    return StreamingResponse(
        ask_stream(request.question, request.session_id),
        media_type="text/plain"
    )

@app.get("/history/{session_id}", dependencies=[Depends(verify_api_key)])
def get_session_history(session_id: str):
    history = get_history(session_id)
    return {"session_id": session_id, "messages": history}

@app.delete("/history/{session_id}", dependencies=[Depends(verify_api_key)])
def clear_session_history(session_id: str):
    clear_history(session_id)
    return {"message": f"Session {session_id} cleared"}
