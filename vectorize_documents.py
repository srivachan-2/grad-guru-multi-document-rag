import os
import shutil
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOCAL_DB_DIR = BASE_DIR / "vector_db_dir"
DEFAULT_COLLECTION_NAME = "grad_guru_rag"
COLLECTION_NAME = DEFAULT_COLLECTION_NAME
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Substring patterns for sensitive personal documents that must never be indexed or exposed
SENSITIVE_PATTERNS = [
    "cv",
    "resume",
    "certificate",
    "offer",
    "vachan",
    "dhurishett",
    "codevita",
]

# Explicitly empty set: demo documents are completely removed from the default knowledge base
DEMO_DOCUMENTS = set()

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "LOCAL_DB_DIR",
    "COLLECTION_NAME",
    "DEFAULT_COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "DEMO_DOCUMENTS",
    "SENSITIVE_PATTERNS",
    "is_sensitive_document",
    "get_session_documents",
    "get_indexed_pdfs",
    "clear_session_data",
    "clear_all_documents",
    "reset_to_demo_documents",
    "get_db_dir",
    "DB_DIR",
    "build_session_vectorstore",
    "load_session_vectorstore",
    "build_vectorstore",
    "load_vectorstore",
]

def is_sensitive_document(filename: str) -> bool:
    """Return True if the filename matches sensitive personal document patterns."""
    name_lower = Path(filename).name.lower()
    return any(p in name_lower for p in SENSITIVE_PATTERNS)

def get_session_documents(docs_dir: Path) -> list[Path]:
    """
    Scans docs_dir strictly for valid, non-sensitive PDF documents.
    Ensures documents are not silently loaded from an unintended directory.
    """
    resolved_dir = Path(docs_dir).resolve()
    if not resolved_dir.exists():
        return []

    valid_pdfs = []
    for pdf_path in resolved_dir.glob("*.pdf"):
        # Prevent traversal or loading from subdirectories
        if pdf_path.resolve().parent != resolved_dir:
            continue
        # Exclude personal CVs / sensitive files
        if is_sensitive_document(pdf_path.name):
            continue
        valid_pdfs.append(pdf_path)

    return sorted(valid_pdfs, key=lambda p: p.name.lower())

def get_indexed_pdfs(data_dir: Path = DATA_DIR) -> list[Path]:
    """Backward compatibility alias for scanning a directory."""
    return get_session_documents(data_dir)

def clear_session_data(session_dir: Path, collection_name: str = None) -> None:
    """
    Safely clears all documents and deletes the Chroma collection for a session.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        return

    docs_dir = session_dir / "documents"
    if docs_dir.exists():
        for doc_file in docs_dir.glob("*"):
            try:
                if doc_file.is_file():
                    doc_file.unlink(missing_ok=True)
            except Exception:
                pass

    # Delete collection from Chroma if possible
    if collection_name:
        try:
            db_dir = session_dir / "chroma_db"
            if db_dir.exists() and any(db_dir.iterdir()):
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                vs = Chroma(
                    collection_name=collection_name,
                    persist_directory=str(db_dir),
                    embedding_function=embeddings,
                )
                vs.delete_collection()
                if hasattr(vs, "_client"):
                    try:
                        vs._client.reset()
                    except Exception:
                        pass
                del vs
        except Exception:
            pass

    import gc
    gc.collect()

    try:
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception:
        pass

def clear_all_documents(data_dir: Path = DATA_DIR) -> int:
    """Removes all PDF documents from the directory for clean reset."""
    resolved_dir = Path(data_dir).resolve()
    if not resolved_dir.exists():
        return 0
    removed = 0
    for item in resolved_dir.iterdir():
        if item.is_file() and item.name.lower().endswith(".pdf"):
            try:
                item.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed

def reset_to_demo_documents(data_dir: Path = DATA_DIR) -> int:
    """Alias for clear_all_documents to support legacy calls."""
    return clear_all_documents(data_dir)

def get_db_dir() -> Path:
    """Resolve the Chroma database path."""
    env_dir = os.getenv("CHROMA_DB_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return LOCAL_DB_DIR

DB_DIR = get_db_dir()

def build_session_vectorstore(
    docs_dir: Path,
    db_dir: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    status_callback=None
):
    """
    Extracts text from PDFs in docs_dir, creates chunks and embeddings,
    and updates the persistent Chroma vector store in db_dir.
    """
    docs_dir = Path(docs_dir)
    db_dir = Path(db_dir)
    pdfs = get_session_documents(docs_dir)
    if not pdfs:
        raise RuntimeError("No PDF files found to index. Please upload at least one valid text-based PDF.")

    if status_callback:
        status_callback(f"Scanning {len(pdfs)} PDF file(s)...")

    documents = []
    success_files = []
    scanned_or_empty_files = []
    failed_files = []

    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            loaded_docs = loader.load()

            # Check whether pages contain readable text
            text_pages = [doc for doc in loaded_docs if doc.page_content.strip()]
            if text_pages:
                documents.extend(text_pages)
                success_files.append(pdf.name)
            else:
                scanned_or_empty_files.append(pdf.name)
        except Exception as exc:
            failed_files.append((pdf.name, str(exc)))

    if not documents:
        error_msg = "No readable text could be extracted from the uploaded PDF(s)."
        if scanned_or_empty_files:
            error_msg += f" The following file(s) appear to be scanned images or empty: {', '.join(scanned_or_empty_files)}."
        if failed_files:
            error_msg += f" Failed to read: {', '.join(f[0] for f in failed_files)}."
        raise RuntimeError(error_msg)

    if status_callback:
        status_callback(f"Loaded {len(documents)} page(s) with text from {len(success_files)} file(s).")
        if scanned_or_empty_files:
            status_callback(f"Notice: {len(scanned_or_empty_files)} scanned/image-only PDF(s) skipped ({', '.join(scanned_or_empty_files)}).")
        status_callback("Splitting text into semantic chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    if status_callback:
        status_callback(f"Generating embeddings for {len(chunks)} chunks...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing collection before rebuilding
    if db_dir.exists() and any(db_dir.iterdir()):
        try:
            existing_vs = Chroma(
                collection_name=collection_name,
                persist_directory=str(db_dir),
                embedding_function=embeddings,
            )
            existing_vs.delete_collection()
        except Exception:
            pass

    if status_callback:
        status_callback("Writing vector embeddings to database...")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(db_dir),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to create vector store: {exc}")

    return vectorstore, len(success_files), len(chunks), scanned_or_empty_files, failed_files

def load_session_vectorstore(db_dir: Path, collection_name: str = DEFAULT_COLLECTION_NAME):
    """Loads an existing Chroma vector store from db_dir."""
    db_dir = Path(db_dir)
    if not db_dir.exists() or not any(db_dir.iterdir()):
        raise RuntimeError("Vector database not found. Please upload and index documents first.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )

def build_vectorstore(status_callback=None):
    """Backward compatibility fallback using default directories."""
    return build_session_vectorstore(DATA_DIR, LOCAL_DB_DIR, DEFAULT_COLLECTION_NAME, status_callback)

def load_vectorstore():
    """Backward compatibility fallback using default directories."""
    return load_session_vectorstore(LOCAL_DB_DIR, DEFAULT_COLLECTION_NAME)
