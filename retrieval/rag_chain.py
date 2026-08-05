import json
import os
import warnings
warnings.filterwarnings("ignore")
from http.client import responses

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

def ask(question: str) -> dict:
    print(f"\nQuestion: {question}")

    # Step 1: Embed the question
    question_vector = embed_question(question)
    #with open("retrieval/test_question_vector.json", "r") as f:
        #question_vector = json.load(f)

    # Step 2: Search Pinecone for similar chunks
    index = init_pinecone()
    matches = query_pinecone(index, question_vector, top_k=3)

    # Step 3: Rerank → keep best 3
    best_matches = rerank(question, matches, top_n=3)

    # Step 4: Build context from matched chunks
    context = ""
    sources = []
    for match in matches:
        context += match.metadata["text"] + "\n\n"
        sources.append(match.metadata.get("source", "unknown"))

     # Step 4: Send question + context to GPT
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

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": list(set(sources))
    }