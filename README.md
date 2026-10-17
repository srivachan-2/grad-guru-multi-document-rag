# Project 1: Multi-Document RAG Chatbot (Groq)

**Grad Guru Course-Aligned Project (Lesson 20)**  
A full-stack, retrieval-augmented generation (RAG) assistant designed to index multiple PDF documents locally and deliver accurate, strictly grounded answers with page-level citations using Groq's high-speed inference engine.

---

## 🏛️ Architecture & Tech Stack

- **User Interface**: [Streamlit](https://streamlit.io/) with a clean dark-themed interface, compact header status indicators, dedicated Documents sidebar panel, and dual-mode chat support (General AI Chat & Document Chat).
- **LLM Orchestration**: [LangChain](https://www.langchain.com/) (`ConversationalRetrievalChain` with `ConversationBufferMemory`).
- **Inference Provider**: [Groq Cloud](https://groq.com/) using the `openai/gpt-oss-20b` open-weight model for rapid inference.
- **Embeddings**: Local Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Vector Database**: [Chroma](https://www.trychroma.com/) with persistent local storage and automated clean rebuilding.
- **Document Ingestion**: Pure-Python `pypdf` loader supporting multi-PDF directories and dynamic page metadata resolution without external Poppler dependencies.

---

## 🌟 Key Features

1. **Dual Chat Modes**:
   - **General AI Chat**: Direct high-speed conversational responses from Groq without requiring PDF uploads or vector indexing.
   - **Document Chat (RAG)**: Strictly grounded retrieval augmented responses citing exact file and page sources.
2. **Dedicated Documents Panel**: Clean sidebar focusing on document management: uploaded PDF count, active vector store badge, multi-file upload control, and one-click index rebuilding.
3. **Accurate Page Citations**: Resolves document metadata to display exact file and page references in collapsible source cards.
4. **Clean Rebuild Mechanism**: Re-indexing cleans previous collections and database directories, preventing chunk duplication.
5. **Secure Cloud-Ready Configuration**: Automatically loads `GROQ_API_KEY` from Streamlit secrets (`secrets.toml`) on Streamlit Community Cloud or a local `.env` file during local development. API keys are never exposed in the UI or committed to git.
6. **Chat Management**: Provides a one-click **Clear Chat History** button to reset conversational memory and messages.

---

## 🚀 Setup & Local Installation

### 1. Prerequisites
- **Python 3.11** (recommended for full pre-compiled wheel compatibility).
- A free Groq API key from [Groq Console](https://console.groq.com/keys).

### 2. Create an Isolated Virtual Environment
In your terminal, navigate to this project folder:
```bash
cd project1_groq_multi_document_rag
py -3.11 -m venv .venv
```

Activate the environment:
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### 1. Local Development (`.env`)
1. Copy the provided `.env.example` file to create `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```
   *(Note: `.env` is listed in `.gitignore` to prevent committing or leaking secrets).*

### 2. Streamlit Community Cloud Deployment
When deploying to [share.streamlit.io](https://share.streamlit.io/):
1. In your App Settings, open the **Secrets** section.
2. Paste your API key in TOML format:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
3. Save the secret. The app will automatically detect it upon startup.

---

## ☁️ Streamlit Cloud Deployment Guide

1. Push this project repository to GitHub.
2. Go to **share.streamlit.io** and select **New app**.
3. Configure deployment:
   - **Repository**: Your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `project1_groq_multi_document_rag/app.py` (or `app.py` if deployed from the subfolder root)
4. Add your `GROQ_API_KEY` in **Advanced Settings > Secrets**.
5. Click **Deploy**. On first load, the app starts with the included non-sensitive demo PDFs (`sample_course_guide.pdf` and `sample_doc_two.pdf`).

---

## 🧪 Verification & Sample Test

This project includes verified non-sensitive demo documents in `data/`:

- **Test Question (Document Mode)**:
  > *"What is the secret project verification code?"*
- **Expected Answer**:
  > Identifies code **`GG-RAG-2026-ALPHA`**.
- **Expected Source**:
  > `sample_course_guide.pdf · page 2`
- **Out-of-Scope Question**:
  > *"Who won the 1994 World Cup?"*
- **Expected Answer**:
  > *"I do not have enough information in the provided documents to answer that question."*

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Key Missing` status banner | `GROQ_API_KEY` not found | Add `GROQ_API_KEY=...` to `.env` locally or in Streamlit Cloud Secrets. |
| `No readable text extracted` | Scanned image PDF or empty file | Use selectable text PDFs or upload text documents via the sidebar. |
| `403 AxiosError on upload` | XSRF cookie conflict | Configured in `.streamlit/config.toml` (`enableXsrfProtection = false`). |
