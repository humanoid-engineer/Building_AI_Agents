import logging
from sentence_transformers import SentenceTransformer, util
import re
from pypdf import PdfReader
import chromadb

# Set logging level to ERROR to suppress warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Initialize model and ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
reader = PdfReader("/Users/mathivanan/Downloads/Building AI Agents - A Practical Beginners Guide.pdf")

# Read PDF and extract text
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# Chunk by paragraphs FIRST (before cleaning)
def chunk_by_paragraphs(text):
    # Split by single newlines and group into meaningful chunks
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Add line to current chunk
        current_chunk += line + " "
        
        # Create new chunk when we reach good size (300-600 chars)
        if len(current_chunk) >= 300:
            # Look for sentence ending
            if line.endswith(('.', '!', '?')):
                chunks.append(current_chunk.strip())
                current_chunk = ""
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

chunks = chunk_by_paragraphs(text)

# Clean each chunk
def clean_text(text):
    # Remove unwanted characters: extra whitespaces, special chars
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
    text = re.sub(r'[,:;\-•]', '', text)  # Remove colons, commas, semicolons, hyphens, bullets
    return text.strip()

chunks = [clean_text(chunk) for chunk in chunks]
print(f"Total chunks created: {len(chunks)}")

# Query the vector database
query = "Why we need RAG?"

# Add chunks to ChromaDB with embeddings and search
def embeddingAndSearch_with_chromadb(chunks):
    chroma = chromadb.Client()  
    db = chroma.create_collection("pdf_knowledge")
    # Persist chunks in ChromaDB
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    db.add(ids=chunk_ids, documents=chunks)

    # Query ChromaDB
    results = db.query(query_texts=[query], n_results=1)

    # Print the best match
    best_match = results["documents"][0][0]
    print("\n" + "="*80)
    print("QUERY:", query)
    print("="*80)
    print("\nBEST MATCHED PARAGRAPH:")
    print("-"*80)
    print(best_match)
    print("-"*80)
    
# Manual embedding and similarity search without ChromaDB
def manual_embedding_search(chunks):
    # Embed all the chunks
    embeddings = model.encode(chunks)

    # Embed the query
    query_embedding = model.encode([query])

    # Compute cosine similarities
    scores = util.cos_sim(query_embedding, embeddings)  

    # Find best match
    best_idx = scores.argmax()
    print("Best match:", chunks[best_idx])
    print("Score:", scores[0][best_idx].item())
    
# benchmarking manual embedding 

    print(scores[0][:10])

# Either choose manual search or ChromaDB
# manual_embedding_search(chunks, query)
manual_embedding_search(chunks)

# Embedding and similarity search with ChromaDB
#embeddingAndSearch_with_chromadb(chunks)