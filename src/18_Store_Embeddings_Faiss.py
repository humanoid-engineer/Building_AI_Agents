import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documents and IDs
documents = [
    "Deep learning uses neural networks to learn patterns.",
    "Sachin is a good cricketer",
    "Dhoni has good friends network"
]

# Embed documents
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query
query = "How do networks learn?"
query_embedding = model.encode([query]).astype('float32')
D, I = index.search(query_embedding, k=1)

# Print most similar document
print(D[0][0])
print(documents[I[0][0]])