import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(docs_path: str) -> list:

    all_docs = []

    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)

        if filename.endswith(".pdf"):
            if filename == 'qna.pdf':
                continue
            loader = PyPDFLoader(filepath)
            all_docs.extend(loader.load())
        elif filename.endswith(".txt"):
            print('add text file section!')

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(all_docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

    return chunks

# --- Execution ---
#load_documents('docs')