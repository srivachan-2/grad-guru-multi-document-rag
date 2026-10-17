"""Build a persistent Chroma vector store from PDFs in ./data."""
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "vector_db_dir"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vectorstore(status_callback=None):
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

    # Clear previous vector store to prevent duplicate chunks on rebuild
    if DB_DIR.exists():
        try:
            existing_vs = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
            existing_vs.delete_collection()
        except Exception:
            pass
        shutil.rmtree(DB_DIR, ignore_errors=True)

    if status_callback:
        status_callback("Writing vector embeddings to Chroma database...")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(DB_DIR),
        )
    except Exception as exc:
        # Clean up corrupted directory if embedding storage fails
        if DB_DIR.exists():
            shutil.rmtree(DB_DIR, ignore_errors=True)
        raise RuntimeError(f"Failed to create Chroma vector store: {exc}")

    return vectorstore, len(success_files), len(chunks), scanned_or_empty_files, failed_files

def load_vectorstore():
    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        raise RuntimeError(
            "Vector database not found. Please click 'Build / Rebuild Index' in the sidebar first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
