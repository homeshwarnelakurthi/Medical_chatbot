"""Helper utilities for loading, splitting and embedding medical documents."""

from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # output size of all-MiniLM-L6-v2


def load_pdf_files(data_path: str) -> List[Document]:
    """Load every PDF file inside ``data_path`` as LangChain documents."""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only page content and the source path in metadata.

    Pinecone stores metadata with every vector, so stripping the extra
    PDF metadata keeps the index payload small.
    """
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")},
        )
        for doc in docs
    ]


def text_split(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks sized for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(documents)


def download_embeddings() -> HuggingFaceEmbeddings:
    """Download (or load from cache) the sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
