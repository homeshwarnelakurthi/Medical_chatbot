"""One-time script: embed the PDFs in ``data/`` and upsert them into Pinecone.

Run after placing your medical PDF(s) in the data/ folder:

    python store_index.py
"""

import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    EMBEDDING_DIMENSION,
    download_embeddings,
    filter_to_minimal_docs,
    load_pdf_files,
    text_split,
)

INDEX_NAME = "medical-chatbot"


def main() -> None:
    load_dotenv()
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY is not set. Add it to your .env file.")

    print("Loading PDFs from data/ ...")
    extracted_docs = load_pdf_files("data/")
    if not extracted_docs:
        raise FileNotFoundError(
            "No PDF files found in data/. Place your medical PDF(s) there first."
        )

    minimal_docs = filter_to_minimal_docs(extracted_docs)
    text_chunks = text_split(minimal_docs)
    print(f"Split {len(extracted_docs)} pages into {len(text_chunks)} chunks.")

    embeddings = download_embeddings()

    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    print("Embedding chunks and upserting to Pinecone (this can take a while) ...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print(f"Done. Indexed {len(text_chunks)} chunks into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
