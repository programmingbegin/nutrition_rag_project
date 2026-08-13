"""
Simple RAG chatbot: ChromaDB (retrieval) + OpenAI (generation)

Prereqs:
    pip install chromadb openai

Set your API key as an environment variable before running:
    export OPENAI_API_KEY="sk-..."   (Mac/Linux)
    set OPENAI_API_KEY="sk-..."      (Windows cmd)
What
Usage:
    python chroma_chatbot.py
"""
from dotenv import load_dotenv
import os
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from openai import OpenAI
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your setup
# ---------------------------------------------------------------------------
#"C:\Users\dprad\Documents\Nutrition RAG Project\chroma_db\chroma.sqlite3"
CHROMA_PATH = Path(__file__).parent.parent/ "Nutrition RAG Project" / "chroma_db"         # path to your persisted Chroma DB
COLLECTION_NAME = "nutrition_guidelines"    # name of the collection you created
EMBEDDING_MODEL = "text-embedding-3-small"  # must match what you used when indexing
CHAT_MODEL = "gpt-4o-mini"           # swap for gpt-4o, gpt-4.1, etc.
N_RESULTS = 4                        # how many chunks to retrieve per question

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
client = OpenAI()  # reads OPENAI_API_KEY from env automatically

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL,
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# list_of_collections = chroma_client.list_collections()

# for c in list_of_collections:
#     print(c.name)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef
)


# ---------------------------------------------------------------------------
# CORE RAG LOGIC
# ---------------------------------------------------------------------------
def retrieve_context(question: str, n_results: int = N_RESULTS) -> list[str]:
    """Query Chroma for the most relevant chunks to the question."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )
    documents = results.get("documents", [[]])[0]
    return documents


def build_prompt(question: str, context_chunks: list[str]) -> list[dict]:
    """Build the messages payload for the chat model."""
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    print("Program is building prompt")
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the context provided below. If the answer isn't in the context, say "
        "you don't know rather than making something up. Be concise.\n\n"
        f"CONTEXT:\n{context_text}"
    )
    print("Program is building prompt return")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def ask(question: str) -> str:
    context_chunks = retrieve_context(question)
    messages = build_prompt(question, context_chunks)
    print("Program is here")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    print("Program is here")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# SIMPLE CLI LOOP
# ---------------------------------------------------------------------------
def main():
    print(f"Chatbot ready. Collection: '{COLLECTION_NAME}'. Type 'exit' to quit.\n")
    while True:
        question = input("You:").strip()
        print("Reached after input")
        if question.lower() in ("exit", "quit"):
            break
        answer = ask(question)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()