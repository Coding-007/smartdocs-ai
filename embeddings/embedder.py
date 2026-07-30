import json
import os
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

def get_embeddings(chunks: list) -> list:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract just the text from each chunk
    texts = [chunk.page_content for chunk in chunks]

    # # #open-ai api and create vectors
    # response = client.embeddings.create(
    #     input = texts,
    #     model = "text-embedding-3-small"
    # )
    #
    # vectors = []
    # for i, embedding in enumerate(response.data):
    #     vectors.append({
    #         'id': f"{chunks[i].metadata.get('source', 'doc').replace('/', '_')}_{i}",
    #         'values': embedding.embedding,
    #         'metadata': {
    #             'text': texts[i],
    #             'source': chunks[i].metadata.get('source', 'unknown'),
    #             'chunk_id': chunks[i].metadata.get('chunk_id', i),
    #         }
    #     })
    #
    # # #Save to JSON file
    # with open("embeddings/test2_vectors.json", "w") as f:
    #     json.dump(vectors, f)

    with open("embeddings/test2_vectors.json", "r") as file:
        vectors = json.load(file)

    return vectors

# --- Execution ---
#embeddings = get_embeddings(chunks)

