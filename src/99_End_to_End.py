import os
from typing import List, Dict
from fastapi import FastAPI, Query
from pydantic import BaseModel
from chromadb.config import Settings
import streamlit as st
from openai import OpenAI
import hashlib
from ai_helpers import chunk_exists_in_vectordb

# helpers
from ai_helpers import (
    read_pdfs,
    clean_text,
    chunk_text,
    store_chunks,
    query_chunks,
    flatten_documents,
    get_openai_response,
    save_uploaded_files,
)


# ---- Small helper functions to keep UI readable ----
def process_uploaded_files(files, upload_dir: str = "temp") -> bool:
    """Save uploaded files, chunk them, de-duplicate by hash, and store new chunks.

    Returns True if new chunks were added, False if nothing new.
    """
    pdf_paths = save_uploaded_files(files, upload_dir=upload_dir)
    texts = read_pdfs(pdf_paths)
    new_chunks, new_metadatas = [], []
    for i, text in enumerate(texts):
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
            if not chunk_exists_in_vectordb(chunk_hash):
                new_chunks.append(chunk)
                new_metadatas.append({"source": pdf_paths[i], "hash": chunk_hash})

    if new_chunks:
        store_chunks(new_chunks, new_metadatas)
        return True
    return False


def retrieve_context_and_answer(query: str, progress) -> str | None:
    """Run retrieval and call LLM; return answer string or None if no context."""
    with st.spinner("Retrieving relevant chunks..."):
        results = query_chunks(query, n_results=3)
    progress.progress(40)
    raw_docs = results.get('documents', [])
    flat_docs = flatten_documents(raw_docs)
    context = " ".join(flat_docs)

    # Debug info
    try:
        st.write(f"Context length (chars): {len(context)}")
    except Exception:
        pass

    if not context.strip():
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Set this environment variable to enable LLM responses.")
        return None

    try:
        with st.spinner("Generating answer from the model..."):
            progress.progress(60)
            answer = get_openai_response(context, query, api_key)
            progress.progress(100)
        return answer
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

# ---- End helpers ----

# Ensure tokenizers parallelism is disabled before any import that may use tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Streamlit page config and UI tweaks
try:
    st.set_page_config(page_title="AI Agent PDF Semantic Search", layout="wide")
except Exception:
    # set_page_config can only be called once and must be called before other st.* calls
    pass

# CSS to hide Streamlit main menu (which includes Deploy/Share/More) and header/footer
_HIDE_STREAMLIT_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""

# ----------- Step 4: FastAPI for Semantic Search -----------
app = FastAPI()

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: SearchRequest):
    # Use helper to query stored chunks
    results = query_chunks(request.query, n_results=5)
    return {"results": results.get('documents', []), "metadatas": results.get('metadatas', [])}

# ----------- Step 5: Streamlit UI with LLM -----------
def streamlit_ui():
    # hide Streamlit menu items and set UI
    try:
        st.markdown(_HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
    except Exception:
        pass
    st.title("AI Agent PDF Semantic Search")

    # Ensure session state flag
    if 'has_docs' not in st.session_state:
        st.session_state['has_docs'] = False

    uploaded_files = st.file_uploader("Upload PDF file", type="pdf", accept_multiple_files=False)
    if uploaded_files:
        files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        added = process_uploaded_files(files)
        if added:
            st.success("New unique chunks processed and stored in ChromaDB.")
        else:
            st.info("No new unique chunks to add. All chunks already exist in the database.")
        st.session_state['has_docs'] = True

    if not st.session_state.get('has_docs', False):
        st.info("Upload and process at least one PDF to enable asking questions.")
        return

    query = st.chat_input("Ask a question about your PDF's")
    if not query:
        return

    # Show progress & handle query/answer
    progress = st.progress(0)
    answer = retrieve_context_and_answer(query, progress)
    if answer is None:
        st.warning("No relevant context found for your query.")
    else:
        st.write("Answer:", answer)

# To run Streamlit UI: 
if __name__ == "__main__":
    streamlit_ui()