"""
Hybrid retriever: combines BM25 (keyword/exact-term matching) with vector
similarity search, fused via LangChain's EnsembleRetriever (reciprocal rank
fusion under the hood).

Why hybrid: nutrition guidelines mix exact terminology ("vitamin D",
"2,000 mg sodium") with conceptual questions ("what should I eat to sleep
better"). Vector search alone tends to miss exact-term queries; BM25 alone
misses paraphrased/conceptual ones.
"""

from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from ingestion import CHROMA_DIR, COLLECTION_NAME, load_documents, chunk_documents

# Weighting: slightly favor semantic search, since most nutritionist queries
# are conceptual ("what should I eat for more energy") rather than exact-term
# lookups. Tune this during Week 3 evaluation.
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def build_hybrid_retriever(k: int = 5, chroma_dir: Path = CHROMA_DIR):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # BM25Retriever needs the raw documents in memory (it's not persisted
    # like Chroma) — re-chunk from source PDFs each run. For a larger corpus,
    # consider pickling the chunked docs instead of re-parsing PDFs.
    documents = load_documents()
    chunks = chunk_documents(documents)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
    )
    return hybrid_retriever


if __name__ == "__main__":
    retriever = build_hybrid_retriever()
    results = retriever.invoke("how much sodium should an adult have per day")
    for doc in results:
        print(f"--- {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')}) ---")
        print(doc.page_content[:200], "...\n")
