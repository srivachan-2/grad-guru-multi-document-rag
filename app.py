import os
from pathlib import Path
import streamlit as st

# MUST be the very first Streamlit command executed
st.set_page_config(
    page_title="Multi-Document RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
try:
    from vectorize_documents import (
        build_vectorstore,
        load_vectorstore,
        DATA_DIR,
        get_db_dir,
        COLLECTION_NAME,
        get_indexed_pdfs,
        is_sensitive_document,
        reset_to_demo_documents,
        DEMO_DOCUMENTS,
    )
except ImportError:
    import importlib
    import vectorize_documents
    try:
        importlib.reload(vectorize_documents)
    except Exception:
        pass
    try:
        from vectorize_documents import (
            build_vectorstore,
            load_vectorstore,
            DATA_DIR,
            get_db_dir,
            COLLECTION_NAME,
            get_indexed_pdfs,
            is_sensitive_document,
            reset_to_demo_documents,
            DEMO_DOCUMENTS,
        )
    except ImportError:
        from vectorize_documents import (
            build_vectorstore,
            load_vectorstore,
            DATA_DIR,
            get_db_dir,
            COLLECTION_NAME,
        )
        DEMO_DOCUMENTS = getattr(vectorize_documents, "DEMO_DOCUMENTS", {"sample_course_guide.pdf", "sample_doc_two.pdf"})
        SENSITIVE_PATTERNS = getattr(
            vectorize_documents,
            "SENSITIVE_PATTERNS",
            ["cv", "resume", "certificate", "offer", "vachan", "dhurishett", "codevita"]
        )

        def is_sensitive_document(filename: str) -> bool:
            if hasattr(vectorize_documents, "is_sensitive_document"):
                return vectorize_documents.is_sensitive_document(filename)
            name_lower = Path(filename).name.lower()
            return any(p in name_lower for p in SENSITIVE_PATTERNS)

        def get_indexed_pdfs(data_dir: Path = DATA_DIR) -> list:
            if hasattr(vectorize_documents, "get_indexed_pdfs"):
                return vectorize_documents.get_indexed_pdfs(data_dir)
            resolved_data_dir = data_dir.resolve()
            resolved_data_dir.mkdir(parents=True, exist_ok=True)
            valid_pdfs = []
            for pdf_path in resolved_data_dir.glob("*.pdf"):
                if pdf_path.resolve().parent != resolved_data_dir:
                    continue
                if is_sensitive_document(pdf_path.name):
                    continue
                valid_pdfs.append(pdf_path)
            return sorted(valid_pdfs, key=lambda p: p.name.lower())

        def reset_to_demo_documents(data_dir: Path = DATA_DIR) -> int:
            if hasattr(vectorize_documents, "reset_to_demo_documents"):
                return vectorize_documents.reset_to_demo_documents(data_dir)
            resolved_data_dir = data_dir.resolve()
            if not resolved_data_dir.exists():
                return 0
            removed = 0
            for item in resolved_data_dir.iterdir():
                if item.is_file() and item.name not in DEMO_DOCUMENTS:
                    try:
                        item.unlink(missing_ok=True)
                        removed += 1
                    except Exception:
                        pass
            return removed

BASE_DIR = Path(__file__).resolve().parent
# Load .env file directly from this project directory with override enabled
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# 1. Secure API Key Resolution: check .env / os.environ, then safely check Streamlit secrets
def get_groq_api_key():
    # Priority 1: Check environment variable (including .env loaded above)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key

    # Priority 2: Check Streamlit secrets without crashing when secrets.toml does not exist
    try:
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            val = secrets.get("GROQ_API_KEY", None)
            if val:
                return str(val).strip()
    except Exception:
        # FileNotFoundError, KeyError, or other secrets configuration errors
        pass

    return ""

groq_api_key = get_groq_api_key()

# Custom Dark Theme Styling for a sleek, modern look
st.markdown("""
<style>
    /* Metric pill header */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-online {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-neutral {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    /* Compact header container */
    .header-container {
        padding-bottom: 12px;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .file-chip {
        padding: 6px 10px;
        margin: 4px 0px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Grounded QA Prompt to ensure responses come strictly from context in Document Chat mode
QA_PROMPT_TEMPLATE = """You are a helpful and precise assistant answering questions about documents.
Use ONLY the following pieces of retrieved context to answer the question at the end.
If the context does not contain enough information to answer the question accurately, respond with:
"I do not have enough information in the provided documents to answer that question."
Do not try to make up an answer or rely on outside knowledge.

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def format_source_page(doc):
    page_val = doc.metadata.get("page")
    if page_val is not None:
        try:
            return f" · page {int(page_val) + 1}"
        except (ValueError, TypeError):
            return f" · page {page_val}"
    page_num = doc.metadata.get("page_number")
    if page_num is not None:
        return f" · page {page_num}"
    return ""

# Check directory stats strictly within DATA_DIR
current_pdfs = get_indexed_pdfs(DATA_DIR)
pdf_count = len(current_pdfs)
active_db_dir = get_db_dir()
index_ready = active_db_dir.exists() and any(active_db_dir.iterdir())
has_api_key = bool(groq_api_key)

# Sidebar: Mode Selector & Documents Panel
with st.sidebar:
    st.subheader("💬 Chat Mode")
    chat_mode = st.radio(
        "Select Mode:",
        ["General AI Chat", "Document Chat (RAG)"],
        index=1,
        help="General AI Chat responds directly using Groq without PDFs. Document Chat grounds answers in your indexed PDFs."
    )

    st.divider()

    if chat_mode == "Document Chat (RAG)":
        st.subheader("📑 Documents Panel")
        st.caption("Manage your knowledge base and vector store.")

        # Status indicators
        st.write(f"**Knowledge Base:** `{pdf_count}` PDF(s)")
        if index_ready:
            st.markdown('<span class="status-badge badge-online">● Vector Index Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-neutral">○ Index Not Built</span>', unsafe_allow_html=True)

        st.write("")
        
        # Initialize uploader key in session state
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0

        # One-time upload notification (consumed on display to prevent repeating across reruns)
        if "upload_notice" in st.session_state:
            notice_type, notice_text = st.session_state.pop("upload_notice")
            if notice_type == "warning":
                st.warning(notice_text)
            elif notice_type == "success":
                st.success(notice_text)
            elif notice_type == "error":
                st.error(notice_text)

        # PDF Upload Controls with dynamic key rotation for clean state management
        uploader_widget_key = f"pdf_uploader_{st.session_state['uploader_key']}"
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more text-based PDFs to add to the knowledge base.",
            label_visibility="collapsed",
            key=uploader_widget_key,
        )
        if uploaded_files:
            saved_files = []
            has_sensitive = False
            has_invalid_type = False
            save_error = None

            for file in uploaded_files:
                safe_name = os.path.basename(file.name).strip()
                if not safe_name.lower().endswith(".pdf"):
                    has_invalid_type = True
                    continue
                if is_sensitive_document(safe_name):
                    has_sensitive = True
                    continue
                try:
                    target_path = DATA_DIR / safe_name
                    with open(target_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_files.append(safe_name)
                except Exception as e:
                    save_error = str(e)

            # Construct one-time notification without exposing personal filenames
            messages = []
            notice_type = "info"

            if saved_files:
                messages.append(f"✅ Added {len(saved_files)} valid file(s). Click Rebuild Index below.")
                notice_type = "success"

            if has_sensitive:
                messages.append("⚠️ Personal documents (CVs, resumes, certificates) cannot be added to the public knowledge base and were skipped.")
                if not saved_files:
                    notice_type = "warning"

            if has_invalid_type:
                messages.append("⚠️ Non-PDF file(s) were skipped (only PDFs are supported).")
                if not saved_files and not has_sensitive:
                    notice_type = "warning"

            if save_error:
                messages.append(f"❌ Upload error: {save_error}")
                notice_type = "error"

            if messages:
                st.session_state["upload_notice"] = (notice_type, "\n\n".join(messages))

            # Rotate key and rerun to clear upload widget state safely
            st.session_state["uploader_key"] += 1
            st.rerun()

        # Rebuild Index Button
        if st.button("🔨 Build / Rebuild Index", type="primary", use_container_width=True):
            try:
                with st.status("Building vector store...", expanded=True) as status_box:
                    def on_status(msg):
                        st.write(f"• {msg}")
                    vs, count_indexed, chunk_count, scanned_files, failed_files = build_vectorstore(status_callback=on_status)
                    status_label = f"✅ Indexed {count_indexed} PDF(s) ({chunk_count} chunks)"
                    if scanned_files:
                        status_label += f" | {len(scanned_files)} scanned file(s) skipped"
                    status_box.update(
                        label=status_label,
                        state="complete",
                        expanded=False
                    )
                st.session_state["indexed"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Indexing error: {exc}")

        # Reset to Demo Documents Button
        if st.button("🔄 Reset to Demo Documents", use_container_width=True, help="Removes any non-demo uploaded documents and reloads default sample PDFs."):
            removed = reset_to_demo_documents(DATA_DIR)
            try:
                with st.spinner("Restoring demo documents and rebuilding index..."):
                    build_vectorstore()
                st.success(f"Reset complete. Restored demo documents ({len(DEMO_DOCUMENTS)} files).")
                st.rerun()
            except Exception as exc:
                st.error(f"Reset error: {exc}")

        # Document list preview
        if current_pdfs:
            with st.expander(f"📁 Document Files ({pdf_count})", expanded=False):
                for p in current_pdfs:
                    st.markdown(f'<div class="file-chip">📄 {p.name}</div>', unsafe_allow_html=True)

        st.divider()
    else:
        st.subheader("💡 General AI Mode")
        st.caption("Ask questions directly to `openai/gpt-oss-20b` without referencing document context.")
        st.divider()

    # Clear Chat
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        st.rerun()

    # Settings Expander (Safe system status, never exposes keys)
    with st.expander("⚙️ System Status", expanded=False):
        st.markdown("**Active Mode:** `" + chat_mode + "`")
        st.markdown("**LLM Engine:** `openai/gpt-oss-20b` (Groq)")
        if chat_mode == "Document Chat (RAG)":
            st.markdown("**Embeddings:** `all-MiniLM-L6-v2` (Local)")
            st.markdown(f"**Storage:** Chroma DB (`{active_db_dir.name}/`)")
            st.markdown(f"**Collection:** `{COLLECTION_NAME}`")
        if has_api_key:
            st.markdown("**API Key:** Configured (`.env` / secrets)")
        else:
            st.markdown("**API Key:** Missing")

# Main Interface: Compact Header
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("### 📚 Multi-Document RAG Chatbot")
    mode_desc = (
        "Document Chat: answers grounded in your indexed local PDFs."
        if chat_mode == "Document Chat (RAG)"
        else "General AI Chat: fast direct responses from Groq without document retrieval."
    )
    st.caption(mode_desc)
with header_col2:
    status_html = (
        '<div style="text-align: right; padding-top: 8px;">'
        + ('<span class="status-badge badge-online">● Groq Connected</span>' if has_api_key else '<span class="status-badge badge-offline">● Key Missing</span>')
    )
    if chat_mode == "Document Chat (RAG)":
        status_html += f'&nbsp;<span class="status-badge badge-neutral">📑 {pdf_count} PDF(s)</span>'
    status_html += '</div>'
    st.markdown(status_html, unsafe_allow_html=True)

st.markdown('<div class="header-container"></div>', unsafe_allow_html=True)

# Missing API Key Warning with Clear Instructions
if not has_api_key:
    st.warning("⚠️ **Groq API Key Required**\n\n"
               "To enable the AI assistant, please provide your Groq API key:\n"
               "- **Local development:** Create a `.env` file in `project1_groq_multi_document_rag/` containing `GROQ_API_KEY=gsk_...`\n"
               "- **Hosted deployment:** Add `GROQ_API_KEY` to your Streamlit secrets (`.streamlit/secrets.toml`).\n\n"
               "*Your key is kept safe and is never committed to Git or displayed on screen.*")

# Document Mode Notice: Request PDFs or building index if not ready
if chat_mode == "Document Chat (RAG)" and has_api_key:
    if pdf_count == 0:
        st.info("ℹ️ **No PDFs detected in `data/` folder.** Please upload text-based PDFs using the sidebar and build the index before querying documents.")
    elif not index_ready:
        st.info("ℹ️ **Vector index not built.** Click **🔨 Build / Rebuild Index** in the sidebar to enable Document Chat.")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 Sources used"):
                for i, doc in enumerate(msg["sources"], 1):
                    source = doc.metadata.get("source", "PDF")
                    st.markdown(f"**{i}. {os.path.basename(str(source))}**{format_source_page(doc)}")
                    st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

# Input placeholder text tailored to mode
input_placeholder = (
    "Ask a question about your documents..."
    if chat_mode == "Document Chat (RAG)"
    else "Ask anything (General AI mode)..."
)

# Chat Input
question = st.chat_input(input_placeholder, disabled=not has_api_key)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            if not has_api_key:
                raise RuntimeError("Groq API key is missing. Please configure GROQ_API_KEY in your .env file or Streamlit secrets.")

            llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=groq_api_key)

            if chat_mode == "General AI Chat":
                # Mode 1: Direct General AI Chat (no retrieval, no PDF dependency)
                with st.spinner("Generating answer from Groq..."):
                    # Build conversational history from memory
                    history_messages = st.session_state.memory.chat_memory.messages
                    call_messages = list(history_messages) + [HumanMessage(content=question)]
                    response = llm.invoke(call_messages)
                    answer = response.content if hasattr(response, "content") else str(response)
                    
                    # Update memory for general chat
                    st.session_state.memory.chat_memory.add_user_message(question)
                    st.session_state.memory.chat_memory.add_ai_message(answer)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                # Mode 2: Document Chat (RAG)
                if pdf_count == 0:
                    raise RuntimeError("No PDFs available. Please upload at least one PDF in the sidebar before asking document questions.")
                if not index_ready:
                    raise RuntimeError("Vector database not found. Please click '🔨 Build / Rebuild Index' in the sidebar first.")

                vectorstore = load_vectorstore()
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                    memory=st.session_state.memory,
                    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                    return_source_documents=True,
                    verbose=False,
                )

                with st.spinner("Searching documents & generating grounded response..."):
                    result = chain.invoke({"question": question})

                answer = result.get("answer", "No answer returned.")
                st.markdown(answer)
                sources = result.get("source_documents", [])
                if sources:
                    with st.expander("📖 Sources used"):
                        for i, doc in enumerate(sources, 1):
                            source = doc.metadata.get("source", "PDF")
                            st.markdown(f"**{i}. {os.path.basename(str(source))}**{format_source_page(doc)}")
                            st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as exc:
            st.error(f"Could not answer: {exc}")
