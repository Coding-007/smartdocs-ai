import os
import json
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
from vectordb.pinecone_store import init_pinecone, query_pinecone
from reranker.reranker import rerank

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_question(question: str) -> list:
    # Convert user question into a vector
    response = client.embeddings.create(
        input=[question],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_context(question: str):
    # Step 1: Load question vector
    question_vector = embed_question(question)

    # Step 2: Get top 10 from Pinecone
    index = init_pinecone()
    matches = query_pinecone(index, question_vector, top_k=10)

    # Step 3: Rerank → keep best 3
    best_matches = rerank(question, matches, top_n=3)

    # Step 4: Build context
    context = ""
    sources = []
    for match in best_matches:
        context += match.metadata["text"] + "\n\n"
        sources.append(match.metadata.get("source", "unknown"))

    return context, list(set(sources))

def ask(question: str) -> dict:
    print(f"\nQuestion: {question}")
    context, sources = get_context(question)

    system_prompt = """You are SmartDocs AI.
        Answer ONLY using the context provided below.
        If the answer is not in the context, say:
        'I could not find that in the documents.'
        Always mention the source at the end."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": sources
    }

def ask_stream(question: str):
    # Same pipeline but streams words as they arrive
    context, sources = get_context(question)

    system_prompt = """You are SmartDocs AI.
    Answer ONLY using the context provided below.
    If the answer is not in the context say:
    'I could not find that in the documents.'
    Always mention the source at the end."""

    # stream=True tells OpenAI to send words one by one
    stream = client.chat.completions.create(
        model="gpt-4o",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    # Yield each word chunk as it arrives
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            yield delta