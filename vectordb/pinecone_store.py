import os
import warnings
warnings.filterwarnings("ignore")

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def init_pinecone(index_name: str = "smartdocs"):
    # Connect to Pinecone using your API key
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Pinecone index {index_name} created")
    else:
        print(f"Pinecone index {index_name} already exists")

    return pc.Index(index_name)

def upsert_vectors(index, vectors: list = None):
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded {len(vectors)} vectors to Pinecone")

def query_pinecone(index, query_vector: list, top_k: int = 3):
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches

