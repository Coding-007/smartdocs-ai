import os
import json
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
from vectordb.pinecone_store import init_pinecone, query_pinecone
from reranker.reranker import rerank
from memory.memory import get_history, add_to_history

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_question(question: str) -> list:
    # Convert user question into a vector
    response = client.embeddings.create(
        input=[question],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def rewrite_question(question: str, history: list) -> str:
    # If no history, nothing to rewrite
    if not history:
        return question

    recent = history[-4:]  # last 2 turns
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

    rewrite_prompt = f"""Given this conversation history:
        {history_text}
        
        Rewrite this follow-up question into a standalone question that makes sense without the history.
        Only output the rewritten question, nothing else.
        
        Follow-up question: {question}
        Standalone question:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper model for this simple task
        messages=[{"role": "user", "content": rewrite_prompt}]
    )

    rewritten = response.choices[0].message.content.strip()
    print(f"Rewritten: '{question}' → '{rewritten}'")
    return rewritten

def get_context(question: str, history: list = None):
    # Rewrite vague follow-ups into standalone questions
    search_question = rewrite_question(question, history or [])

    # Step 1: Load question vector
    question_vector = embed_question(search_question)

    # Step 2: Get top 10 from Pinecone
    index = init_pinecone()
    matches = query_pinecone(index, question_vector, top_k=10)

    # Step 3: Rerank → keep best 3
    best_matches = rerank(search_question, matches, top_n=3)

    # Step 4: Build context
    context = ""
    sources = []
    for match in best_matches:
        context += match.metadata["text"] + "\n\n"
        sources.append(match.metadata.get("source", "unknown"))

    return context, list(set(sources))

def ask(question: str, session_id: str = 'default') -> dict:
    print(f"\nQuestion: {question}")

    history = get_history(session_id)

    # Get this session's history
    history = get_history(session_id)

    context, sources = get_context(question, history)

    system_prompt = """You are SmartDocs AI.
        Answer ONLY using the context provided below.
        If the answer is not in the context, say:
        'I could not find that in the documents.'
        Always mention the source at the end."""

    # Build messages — history + current question
    messages = [
        {"role": "system", "content": system_prompt + f"\n\nContext:\n{context}"}
    ]

    # Add previous conversation turns
    messages.extend(history)

    # Add current question
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    answer = response.choices[0].message.content

    # Save this turn to memory
    add_to_history(session_id, "user", question)
    add_to_history(session_id, "assistant", answer)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "session_id": session_id
    }

def ask_stream(question: str, session_id: str = "default"):
    print(f"\nQuestion: {question}")

    history = get_history(session_id)

    context, sources = get_context(question, history)

    system_prompt = """You are SmartDocs AI.
    Answer ONLY using the context provided below.
    If the answer is not in the context say:
    'I could not find that in the documents.'
    Always mention the source at the end."""

    messages = [
        {"role": "system", "content": system_prompt + f"\n\nContext:\n{context}"}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    # stream=True tells OpenAI to send words one by one
    stream = client.chat.completions.create(
        model="gpt-4o",
        stream=True,
        messages=messages
    )

    # Yield each word chunk as it arrives
    full_answer = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            full_answer += delta
            yield delta

    # Save to memory after streaming completes
    add_to_history(session_id, "user", question)
    add_to_history(session_id, "assistant", full_answer)