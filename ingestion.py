"""
Loads guideline PDFs, chunks them, embeds them, and stores them in a
persistent Chroma vector store.

Usage:
    1. Drop your guideline PDFs (Dietary Guidelines, WHO fact sheets, etc.)
       into data/guidelines/
    2. Run: python src/ingestion.py
    3. This populates data/chroma_db/ — reusable across runs, no need to
       re-embed unless source documents change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

GUIDELINES_DIR = Path(__file__).parent.parent / "data" / "guidelines"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "nutrition_guidelines"

# ~500 tokens ≈ 2000 chars at a rough 4 chars/token estimate. Overlap keeps
# context from being cut mid-thought at chunk boundaries.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def load_documents(guidelines_dir: Path = GUIDELINES_DIR):
    """Load every PDF in the guidelines directory. Each page becomes one
    Document with source metadata carried through automatically by PyPDFLoader."""
    docs = []
    pdf_paths = sorted(guidelines_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {guidelines_dir}. Add your guideline documents "
            "there first (see the data source links from the project plan)."
        )

    for pdf_path in pdf_paths:
        print(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks, persist_dir: Path = CHROMA_DIR):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )
    print(f"Stored {len(chunks)} chunks in Chroma at {persist_dir}")
    return vector_store


def run_ingestion():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    documents = load_documents()
    chunks = chunk_documents(documents)
    build_vector_store(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
