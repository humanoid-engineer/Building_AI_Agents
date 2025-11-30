import os
from typing import List, Dict, Iterable
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from openai import OpenAI


def read_pdfs(pdf_paths: Iterable[str]) -> List[str]:
    """Read list of PDF file paths and return list of extracted text per file."""
    texts = []
    for path in pdf_paths:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        texts.append(text)
    return texts


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


def get_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    return SentenceTransformer(model_name)


def store_chunks(chunks: List[str], metadata: List[Dict], collection_name: str = "pdf_chunks") -> None:
    client = chromadb.Client(Settings())
    collection = client.get_or_create_collection(collection_name)
    model = get_sentence_transformer()
    embeddings = model.encode(chunks)
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[metadata[i]],
            ids=[f"chunk_{i}"]
        )


def query_chunks(query: str, n_results: int = 3, collection_name: str = "pdf_chunks") -> Dict:
    client = chromadb.Client(Settings())
    collection = client.get_collection(collection_name)
    model = get_sentence_transformer()
    query_emb = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )
    return results


def flatten_documents(raw_docs) -> List[str]:
    flat_docs = []
    for item in raw_docs:
        if isinstance(item, list):
            for sub in item:
                flat_docs.append(sub if isinstance(sub, str) else str(sub))
        elif isinstance(item, str):
            flat_docs.append(item)
        else:
            flat_docs.append(str(item))
    return flat_docs


def get_openai_response(context: str, question: str, api_key: str, model: str = "gpt-3.5-turbo") -> str:
    """Call OpenAI v1 client and return a safe string answer."""
    # Try to trim context if it's too large (use tiktoken if available)
    max_context_tokens = 3000
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        ctx_tokens = len(enc.encode(context))
        if ctx_tokens > max_context_tokens:
            # keep the last tokens (assume later chunks are more relevant)
            tokens = enc.encode(context)
            tokens = tokens[-max_context_tokens:]
            context = enc.decode(tokens)
    except Exception:
        # fallback: trim by characters
        if len(context) > 15000:
            context = context[-15000:]

    client = OpenAI(api_key=api_key)
    # Stronger system instruction to prefer context and admit when information is missing.
    system_message = (
        "You are a helpful assistant specialized in answering questions about PDF documents. "
        "Only use the provided CONTEXT to answer the question. If the answer is not present in the CONTEXT, "
        "respond with 'I don't know' or say you don't have enough information. Be concise and cite the source when possible."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)


def save_uploaded_files(uploaded_files, upload_dir: str = "temp") -> List[str]:
    """Save Streamlit uploaded files to `upload_dir` and return file paths."""
    os.makedirs(upload_dir, exist_ok=True)
    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    paths = []
    for uploaded_file in files:
        filename = getattr(uploaded_file, "name", None) or "uploaded.pdf"
        safe_name = os.path.basename(filename)
        path = os.path.join(upload_dir, safe_name)
        if hasattr(uploaded_file, "getbuffer"):
            data = uploaded_file.getbuffer()
        elif hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
        else:
            data = uploaded_file
        with open(path, "wb") as f:
            f.write(data)
        paths.append(path)
    return paths


def chunk_exists_in_vectordb(chunk_hash: str, collection_name: str = "pdf_chunks") -> bool:
    """Check whether a chunk with the given hash already exists in the ChromaDB collection.

    Attempts a metadata-filtered query first (if supported). Falls back to scanning stored metadatas.
    """
    client = chromadb.Client(Settings())
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        # collection may not exist yet
        return False

    # Try a metadata-filtered query (may work depending on chromadb version)
    try:
        results = collection.query(query_embeddings=[], where={"hash": chunk_hash}, n_results=1, include=["metadatas"])
        metadatas = results.get("metadatas", [])
        for item in metadatas:
            if isinstance(item, list):
                for m in item:
                    if isinstance(m, dict) and m.get("hash") == chunk_hash:
                        return True
            elif isinstance(item, dict) and item.get("hash") == chunk_hash:
                return True
    except Exception:
        pass

    # Fallback: retrieve metadatas and scan
    try:
        res = collection.get(include=["metadatas"])
        metadatas = res.get("metadatas", [])
        for m in metadatas:
            if isinstance(m, dict) and m.get("hash") == chunk_hash:
                return True
    except Exception:
        try:
            res = collection.get()
            metadatas = res.get("metadatas", []) if isinstance(res, dict) else []
            for m in metadatas:
                if isinstance(m, dict) and m.get("hash") == chunk_hash:
                    return True
        except Exception:
            pass

    return False
