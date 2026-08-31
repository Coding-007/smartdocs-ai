# SmartDocs AI 🤖📄

A production-ready **RAG (Retrieval Augmented Generation)** system that lets you ask natural language questions over any document library — with real conversation memory, cost tracking, and a live chat interface.

**🔗 Live demo:** `https://your-railway-domain.up.railway.app/app/index.html`  
**📚 API docs:** `https://your-railway-domain.up.railway.app/docs`

---

## What it does

Upload PDFs or text documents, then ask questions in plain English. SmartDocs AI retrieves the most relevant passages, reranks them for accuracy, and generates a grounded answer — always citing its source and never hallucinating facts outside your documents.

---

## Features

- 🔍 **RAG pipeline** — semantic search over your own documents, not generic AI knowledge
- 🎯 **HuggingFace reranker** — cross-encoder re-scores retrieved chunks for higher accuracy
- 💬 **Conversation memory** — ask follow-up questions naturally, with automatic query rewriting for vague references ("tell me more about it")
- ⚡ **Streaming responses** — answers appear word-by-word like ChatGPT
- 🔐 **API key authentication** — all endpoints protected
- 📊 **LLMOps logging** — every query tracked with cost, latency, and token usage
- 🎨 **Chat UI** — clean, responsive interface built with Bootstrap + jQuery
- 🐳 **Dockerized** — fully containerized, deployed on Railway

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Pinecone (serverless) |
| Reranker | HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Orchestration | LangChain |
| Backend | FastAPI |
| Frontend | HTML, Bootstrap 5, jQuery |
| Deployment | Docker, Railway |

---

## Architecture