import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# OpenAI pricing (per 1M tokens) — update if prices change
GPT4O_INPUT_COST = 2.50 # $ per 1M input tokens
GPT4O_OUTPUT_COST = 10.00 # $ per 1M output tokens
EMBEDDING_COST = 0.02  # $ per 1M tokens

LOG_FILE = "logs/queries.jsonl"

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = (prompt_tokens / 1_000_000) * GPT4O_INPUT_COST
    output_cost = (completion_tokens / 1_000_000) * GPT4O_OUTPUT_COST
    return round(input_cost + output_cost, 6)

def log_query(
    question: str,
    answer: str,
    sources: list,
    session_id: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
):
    cost = calculate_cost(prompt_tokens, completion_tokens)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost
    }

    # Append to JSONL file — one JSON object per line
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Logged query — {latency_ms:.0f}ms, ${cost:.6f}")
    return entry

def get_stats() -> dict:
    # Read all logs and compute summary stats
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {"total_queries": 0, "total_cost": 0, "avg_latency_ms": 0}

    if not lines:
        return {"total_queries": 0, "total_cost": 0, "avg_latency_ms": 0}

    entries = [json.loads(line) for line in lines]

    total_cost = sum(e["cost_usd"] for e in entries)
    avg_latency = sum(e["latency_ms"] for e in entries) / len(entries)
    total_tokens = sum(e["prompt_tokens"] + e["completion_tokens"] for e in entries)

    return {
        "total_queries": len(entries),
        "total_cost_usd": round(total_cost, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "total_tokens": total_tokens
    }