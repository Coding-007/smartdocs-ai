# SmartDocs AI

A production-ready RAG (Retrieval Augmented Generation) system 
that lets you ask natural language questions over any document library.

## Tech stack
- OpenAI API — embeddings + GPT-4o
- Pinecone — vector database
- LangChain — document loading and chunking
- FastAPI — REST API
- Docker — containerized Python environment

## How it works
1. Upload PDF documents
2. Documents are split into chunks and converted to vectors
3. Vectors are stored in Pinecone
4. User asks a question via API
5. Question is embedded and matched against stored vectors
6. GPT-4o answers using only retrieved document chunks

## Run locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API endpoints
- `GET /health` — health check
- `POST /ask` — ask a question