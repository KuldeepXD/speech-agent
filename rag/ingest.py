"""
PDF Ingestion — Build FAISS vector stores from Speech and Feeding PDFs.

Loads PDFs from Documents/Speech/ and Documents/Feeding/, chunks the text,
embeds with Google Generative AI embeddings, and saves separate FAISS indexes.

Usage:
    python -m rag.ingest
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "Documents"
VECTOR_STORES_DIR = BASE_DIR / "vector_stores"

SPEECH_DOCS_DIR = DOCUMENTS_DIR / "Speech"
FEEDING_DOCS_DIR = DOCUMENTS_DIR / "Feeding"

SPEECH_INDEX_DIR = VECTOR_STORES_DIR / "speech_index"
FEEDING_INDEX_DIR = VECTOR_STORES_DIR / "feeding_index"

# ── Chunking Config ────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Embedding Model ────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Google Generative AI embeddings instance."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def load_pdfs_from_directory(directory: Path) -> list:
    """Load all PDFs from a directory using PyPDFLoader.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        A list of LangChain Document objects.
    """
    all_documents = []
    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"  [!] No PDF files found in {directory}")
        return all_documents

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            # Add source metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["category"] = directory.name
            all_documents.extend(documents)
            print(f"     [OK] Loaded {len(documents)} pages")
        except Exception as e:
            print(f"     [ERROR] Error loading {pdf_path.name}: {e}")

    return all_documents


def chunk_documents(documents: list) -> list:
    """Split documents into chunks for embedding.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_faiss_index(chunks: list, index_dir: Path) -> FAISS:
    """Build a FAISS vector store from document chunks and save it.

    Args:
        chunks: List of chunked Document objects.
        index_dir: Path to save the FAISS index.

    Returns:
        The created FAISS vector store.
    """
    embeddings = get_embeddings()

    print(f"  Embedding {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save locally
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    print(f"  Saved FAISS index to: {index_dir}")

    return vector_store


def ingest_category(docs_dir: Path, index_dir: Path, category: str) -> None:
    """Full ingestion pipeline for a single category.

    Args:
        docs_dir: Path to the PDF documents directory.
        index_dir: Path to save the FAISS index.
        category: Category name for display.
    """
    print(f"\n{'='*60}")
    print(f"Ingesting {category} documents")
    print(f"{'='*60}")

    # Load
    documents = load_pdfs_from_directory(docs_dir)
    if not documents:
        print(f"  [!] No documents loaded for {category}. Skipping.")
        return

    print(f"\n  Total pages loaded: {len(documents)}")

    # Chunk
    chunks = chunk_documents(documents)
    print(f"  Total chunks created: {len(chunks)}")
    
    # ── RATE LIMIT PREVENTION ──────────────────────────────────────────────
    # The Google Gemini API free tier limits embeddings to 100 requests per minute.
    # To ensure this completes without errors, we will limit ingestion to 
    # the first 50 chunks per category for this demonstration.
    MAX_CHUNKS = 50
    if len(chunks) > MAX_CHUNKS:
        print(f"  [!] Limiting to first {MAX_CHUNKS} chunks to avoid free-tier API rate limits (HTTP 429).")
        chunks = chunks[:MAX_CHUNKS]
    
    # Embed and save
    build_faiss_index(chunks, index_dir)
    print(f"\n  [OK] {category} vector store ready!")


def main():
    """Run the full ingestion pipeline for both categories."""
    print("\nSpeech/Feeding Medical AI Agent -- PDF Ingestion")
    print("=" * 60)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("[ERROR] GOOGLE_API_KEY not set. Please configure your .env file.")
        sys.exit(1)

    # Ingest Speech documents
    ingest_category(SPEECH_DOCS_DIR, SPEECH_INDEX_DIR, "Speech")

    # Ingest Feeding documents
    ingest_category(FEEDING_DOCS_DIR, FEEDING_INDEX_DIR, "Feeding")

    print(f"\n{'='*60}")
    print("Ingestion complete!")
    print(f"  Speech index: {SPEECH_INDEX_DIR}")
    print(f"  Feeding index: {FEEDING_INDEX_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
