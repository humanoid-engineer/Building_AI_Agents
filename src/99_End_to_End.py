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
    # Track whether documents have been uploaded/processed in this session
    if 'has_docs' not in st.session_state:
        st.session_state['has_docs'] = False
    uploaded_files = st.file_uploader("Upload PDF file", type="pdf", accept_multiple_files=False)
    if uploaded_files:
        # Normalize to a list whether file_uploader returned one file or multiple
        files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

        # Save uploaded files using helper which handles various upload shapes
        upload_dir = "temp"
        pdf_paths = save_uploaded_files(files, upload_dir=upload_dir)
        texts = read_pdfs(pdf_paths)
        all_chunks, all_metadata = [], []
        # Hashing function for chunk deduplication

        # Assume you have a helper to check if a chunk hash exists in the vectordb

        for i, text in enumerate(texts):
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)
            for chunk in chunks:
                chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
                if not chunk_exists_in_vectordb(chunk_hash):
                    all_chunks.append(chunk)
                    all_metadata.append({"source": pdf_paths[i], "hash": chunk_hash})
        if all_chunks:
            store_chunks(all_chunks, all_metadata)
            st.success("New unique chunks processed and stored in ChromaDB.")
        else:
            st.info("No new unique chunks to add. All chunks already exist in the database.")
        # mark that we have documents available for querying in this session
        st.session_state['has_docs'] = True
    # Only enable the question input after PDFs have been uploaded/processed in this session
    if st.session_state.get('has_docs', False):
        query = st.chat_input("Ask a question about your PDF's")
        if query:
            pass  # continue to query handling below
    else:
        st.info("Upload and process at least one PDF to enable asking questions.")
        return

    if query:
        # Show a progress bar and spinner to indicate work is happening
        progress = st.progress(0)
        with st.spinner("Retrieving relevant chunks..."):
            results = query_chunks(query, n_results=3)
        progress.progress(40)
        raw_docs = results.get('documents', [])
        flat_docs = flatten_documents(raw_docs)

        context = " ".join(flat_docs)

        # Debug: show top retrieved chunks so you can verify the context
        try:
            st.write(f"Context length (chars): {len(context)}")
        except Exception:
            pass

        if not context.strip():
            st.warning("No relevant context found for your query.")
        else:
            # Use OpenAI LLM (set OPENAI_API_KEY env variable)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY is not set. Set this environment variable to enable LLM responses.")
            else:
                try:
                    # Show spinner while LLM generates the answer; update progress accordingly
                    with st.spinner("Generating answer from the model..."):
                        progress.progress(60)
                        answer = get_openai_response(context, query, api_key)
                        progress.progress(100)
                    st.write("Answer:", answer)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

# To run Streamlit UI: 
if __name__ == "__main__":
    streamlit_ui()