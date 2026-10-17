import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOCAL_DB_DIR = BASE_DIR / "vector_db_dir"
COLLECTION_NAME = "grad_guru_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_db_dir() -> Path:
    """
    Resolve the Chroma persistent directory dynamically.
    Priority:
    1. Explicit environment variable CHROMA_DB_DIR if configured.
    2. Linux / Streamlit Community Cloud: /tmp/chroma_db (writable container filesystem).
    3. Local development: ./vector_db_dir in the project root.
    4. Fallback: OS temp directory if candidate directory is not writable.
    """
    env_dir = os.getenv("CHROMA_DB_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir)
    elif os.name != "nt" and Path("/tmp").exists():
        # Default for Streamlit Cloud and Linux containers
        candidate = Path("/tmp/chroma_db")
    else:
        # Default for local development (Windows / standard local dev)
        candidate = LOCAL_DB_DIR

    # Ensure candidate parent or directory is writable; fallback to tempfile if read-only
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        test_file = candidate / ".write_test"
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        return candidate
    except (PermissionError, OSError):
        fallback = Path(tempfile.gettempdir()) / "chroma_db"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

# Backward compatibility alias
DB_DIR = get_db_dir()

def build_vectorstore(status_callback=None):
    """
    Extracts text from PDFs in DATA_DIR, creates chunks and embeddings,
    and updates the persistent Chroma vector store.
    """
    DATA_DIR.mkdir(exist_ok=True)
    pdfs = sorted(list(DATA_DIR.glob("*.pdf")), key=lambda p: p.name.lower())
    if not pdfs:
        raise RuntimeError("No PDF files found in data/ folder. Please place or upload text-based PDFs before indexing.")

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

    # If absolutely no readable text could be extracted from any PDF
    if not documents:
        error_msg = "No readable text could be extracted from the PDF(s)."
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
    db_dir = get_db_dir()

    # Rebuild safety: clear only our collection if the vectorstore exists.
    # We do NOT remove unrelated directories or delete user documents.
    if db_dir.exists() and any(db_dir.iterdir()):
        try:
            existing_vs = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(db_dir),
                embedding_function=embeddings
            )
            existing_vs.delete_collection()
        except Exception:
            # Collection may not exist yet or was empty, ignore
            pass

    if status_callback:
        status_callback(f"Writing vector embeddings to Chroma database ({db_dir})...")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(db_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create Chroma vector store in '{db_dir}': {exc}. "
            "Please ensure the directory is writable or configure CHROMA_DB_DIR."
        )

    return vectorstore, len(success_files), len(chunks), scanned_or_empty_files, failed_files

def load_vectorstore():
    """
    Loads the persistent Chroma vector store using the dynamically resolved db directory
    and collection name matching build_vectorstore.
    """
    db_dir = get_db_dir()
    if not db_dir.exists() or not any(db_dir.iterdir()):
        raise RuntimeError(
            f"Vector database not found at '{db_dir}'. Please click 'Build / Rebuild Index' in the sidebar first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )

