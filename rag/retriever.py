"""
FAISS Retriever Loaders — Load persisted vector stores for retrieval.

Provides functions to load the Speech and Feeding FAISS indexes
and return LangChain retrievers for use in the RAG pipeline.
"""

from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORES_DIR = BASE_DIR / "vector_stores"

SPEECH_INDEX_DIR = VECTOR_STORES_DIR / "speech_index"
FEEDING_INDEX_DIR = VECTOR_STORES_DIR / "feeding_index"

# ── Embedding Model (must match ingestion) ─────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-2"

# ── Retrieval Config ───────────────────────────────────────────────────
DEFAULT_K = 5  # Number of top documents to retrieve


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the embeddings instance (must match the one used for ingestion)."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def _load_faiss_index(index_dir: Path) -> FAISS:
    """Load a persisted FAISS index from disk.

    Args:
        index_dir: Path to the saved FAISS index directory.

    Returns:
        A FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index directory doesn't exist.
    """
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}. "
            f"Run 'python -m rag.ingest' first to build the vector stores."
        )

    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def load_speech_retriever(k: int = DEFAULT_K):
    """Load the Speech FAISS vector store and return a retriever.

    Args:
        k: Number of top documents to retrieve per query.

    Returns:
        A LangChain retriever configured for the Speech vector store.
    """
    vector_store = _load_faiss_index(SPEECH_INDEX_DIR)
    return vector_store.as_retriever(search_kwargs={"k": k})


def load_feeding_retriever(k: int = DEFAULT_K):
    """Load the Feeding FAISS vector store and return a retriever.

    Args:
        k: Number of top documents to retrieve per query.

    Returns:
        A LangChain retriever configured for the Feeding vector store.
    """
    vector_store = _load_faiss_index(FEEDING_INDEX_DIR)
    return vector_store.as_retriever(search_kwargs={"k": k})


def load_speech_vectorstore() -> FAISS:
    """Load the Speech FAISS vector store directly.

    Returns:
        The Speech FAISS vector store instance.
    """
    return _load_faiss_index(SPEECH_INDEX_DIR)


def load_feeding_vectorstore() -> FAISS:
    """Load the Feeding FAISS vector store directly.

    Returns:
        The Feeding FAISS vector store instance.
    """
    return _load_faiss_index(FEEDING_INDEX_DIR)
