import os
from typing import List, Dict
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
import streamlit as st
from openai import OpenAI

# ----------- Step 1: Read PDF files -----------
def read_pdfs(pdf_paths: List[str]) -> List[str]:
    texts = []
    for path in pdf_paths:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
    return texts

# ----------- Step 2: Clean and Chunk -----------
def clean_text(text: str) -> str:
    return " ".join(text.split())

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    sentences = text.split('. ')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# ----------- Step 3: Store chunks in ChromaDB -----------
def store_chunks(chunks: List[str], metadata: List[Dict], collection_name: str = "pdf_chunks"):
    client = chromadb.Client(Settings())
    collection = client.get_or_create_collection(collection_name)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[metadata[i]],
            ids=[f"chunk_{i}"]
        )

# ----------- Step 4: FastAPI for Semantic Search -----------
app = FastAPI()

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: SearchRequest):
    client = chromadb.Client(Settings())
    collection = client.get_collection("pdf_chunks")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode([request.query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=5
    )
    return {"results": results['documents'], "metadatas": results['metadatas']}

# ----------- Step 5: Streamlit UI with LLM -----------
def streamlit_ui():
    st.title("AI Agent PDF Semantic Search")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        pdf_paths = []
        upload_dir = "temp"
        # Ensure upload directory exists
        os.makedirs(upload_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            # sanitize filename to avoid path traversal
            safe_name = os.path.basename(uploaded_file.name)
            path = os.path.join(upload_dir, safe_name)
            try:
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            except Exception as e:
                st.error(f"Failed to save uploaded file {safe_name}: {e}")
                continue
            pdf_paths.append(path)
        texts = read_pdfs(pdf_paths)
        all_chunks, all_metadata = [], []
        for i, text in enumerate(texts):
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks)
            all_metadata.extend([{"source": pdf_paths[i]} for _ in chunks])
        store_chunks(all_chunks, all_metadata)
        st.success("PDFs processed and stored in ChromaDB.")

    query = st.text_input("Ask a question about your PDFs:")
    if query:
        client = chromadb.Client(Settings())
        collection = client.get_collection("pdf_chunks")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = model.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )
        # results['documents'] may be a nested list (one list per query). Flatten and ensure strings.
        raw_docs = results.get('documents', [])
        flat_docs = []
        for item in raw_docs:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        flat_docs.append(sub)
                    else:
                        flat_docs.append(str(sub))
            elif isinstance(item, str):
                flat_docs.append(item)
            else:
                flat_docs.append(str(item))

        context = " ".join(flat_docs)
        if not context.strip():
            st.warning("No relevant context found for your query.")
        else:
            # Use OpenAI LLM (set OPENAI_API_KEY env variable)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY is not set. Set this environment variable to enable LLM responses.")
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for answering questions about PDF documents."},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                        ]
                    )
                    # Response shape is similar; guard access
                    try:
                        answer = response.choices[0].message.content
                    except Exception:
                        # Fallback to stringifying the response
                        answer = str(response)
                    st.write("Answer:", answer)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

# To run Streamlit UI: 
if __name__ == "__main__":
    streamlit_ui()