"""
Loads guideline PDFs AND web pages, chunks them, embeds them, and stores
them in a persistent Chroma vector store.

Usage:
    1. Drop your guideline PDFs (Dietary Guidelines, WHO fact sheets, etc.)
       into data/guidelines/
    2. List any HTML guideline pages (WHO fact sheet pages, CDC sleep page,
       etc.) one per line in data/guidelines/urls.txt
    3. Run: python src/ingestion.py
    4. This populates data/chroma_db/ — reusable across runs, no need to
       re-embed unless source documents change.
"""

import os
from pathlib import Path

import sys

import pandas as pd
from pathlib import Path
from langchain_core.documents import Document

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd

load_dotenv()

GUIDELINES_DIR = Path(__file__).parent.parent / "Nutrition RAG Project" / "RAG_Data"
CHROMA_DIR = Path(__file__).parent.parent / "Nutrition RAG Project" / "chroma_db"
COLLECTION_NAME = "nutrition_guidelines"
URLS_FILE = GUIDELINES_DIR / "urls.txt"
folder = Path(r"C:\Users\dprad\Documents\Nutrition RAG Project\RAG_Data")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# ~500 tokens ≈ 2000 chars at a rough 4 chars/token estimate. Overlap keeps
# context from being cut mid-thought at chunk boundaries.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200






def load_pdf_documents(guidelines_dir: Path = GUIDELINES_DIR):
    """Load every PDF in the guidelines directory. Each page becomes one
    Document with source metadata carried through automatically by PyPDFLoader."""
    docs = []
    pdf_paths = sorted(guidelines_dir.glob("*.pdf"))

    for pdf_path in pdf_paths:
        print(f"Loading PDF: {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    return docs


# def load_web_documents(urls_file: Path = URLS_FILE):
#     """Load HTML guideline pages listed one-per-line in urls.txt. Blank lines
#     and lines starting with # are ignored, so you can comment out sources."""
#     if not urls_file.exists():
#         return []

#     urls = [
#         line.strip()
#         for line in urls_file.read_text().splitlines()
#         if line.strip() and not line.strip().startswith("#")
#     ]

#     if not urls:
#         return []

#     print(f"Loading {len(urls)} web page(s) from urls.txt...")
#     # WebBaseLoader fetches and strips HTML into plain text via BeautifulSoup.
#     # bs_kwargs can be tuned per-site if a page has heavy nav/ad boilerplate
#     # you want stripped before chunking — left default here for simplicity.
#     loader = WebBaseLoader(web_paths=urls)
#     docs = loader.load()

#     # WebBaseLoader sets metadata["source"] to the URL already, which keeps
#     # citations in agent.py consistent with the PDF path (Path(...).name
#     # on a URL just returns the last path segment, which is fine for display).
#     return docs


def load_documents(guidelines_dir: Path = GUIDELINES_DIR):
    """Load every PDF plus every URL in urls.txt from the guidelines directory."""
    docs = load_pdf_documents(guidelines_dir)
    #docs.extend(load_web_documents(guidelines_dir / "urls.txt"))
    docs.extend(load_excel_documents())

    if not docs:
        raise FileNotFoundError(
            f"No PDFs found in {guidelines_dir} and no URLs found in "
            f"{guidelines_dir / 'urls.txt'}. Add at least one guideline "
            "source before running ingestion (see the data source links "
            "from the project plan)."
        )

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

def delete_vector_store(persist_dir: Path = CHROMA_DIR):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(persist_dir)
        )
    vector_store.delete_collection()
    return vector_store

def load_excel_documents():
    "Load every excel file"
    docs=[]

    for file in folder.glob("*.csv"):
        print(file.stem)
        df = pd.read_csv(file)
        for idx,row in df.iterrows():
            text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
            print(text)
            docs.append(Document(page_content=text,
            metadata={"source": file.stem, "row": idx, "type": "csv"}))   
    return docs

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
    #run_ingestion()
    #print(GUIDELINES_DIR)
    delete_vector_store()
    run_ingestion()


    
    # print("=== PYTHON EXECUTABLE BEING USED ===")
    # print(sys.executable)
    # print("====================================")
