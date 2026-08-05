import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import CrossEncoder

# Load free HuggingFace cross-encoder model
# Downloads automatically on first run (~80MB)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question: str, matches: list, top_n: int = 3) -> list:
    if not matches:
        return []

    # Pair the question with each chunk text
    pairs = [
        (question, match.metadata["text"])
        for match in matches
    ]

    # Score each pair — cross-encoder reads both together
    scores = model.predict(pairs)

    # Zip scores with matches and sort best first
    scored = sorted(
        zip(scores, matches),
        key=lambda x: x[0],
        reverse=True
    )

    # Return only top N matches
    top_matches = [match for _, match in scored[:top_n]]

    print(f"Reranked {len(matches)} chunks → kept top {top_n}")
    for i, (score, match) in enumerate(scored[:top_n]):
        print(f"  {i + 1}. score: {round(float(score), 3)} — {match.metadata.get('source', '?')}")

    return top_matches