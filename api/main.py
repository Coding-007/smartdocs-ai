import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from pydantic import BaseModel
from retrieval.rag_chain import ask

# Create the FastAPI app
app = FastAPI(
    title="SmartDocs AI",
    description="Ask questions from your documents",
    version="1.0"
)

class QuestionRequest(BaseModel):
    question: str

# Route 1: Health check
@app.get("/health")
def health():
    return {"status": "SmartDocs AI is running"}

# Route 2: Ask a question
@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = ask(request.question)
    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"]
    }
