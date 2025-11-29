import chromadb

# Setup
chroma = chromadb.Client()
db = chroma.create_collection("knowledge_base")

# Add knowledge
docs = [
"The iPhone 15 launched in 2023.",
"The iPhone 12 launched in 2020."
]

db.add(ids=["1","2"], documents=docs)

# Query
query = "When was the latest iPhone released?"
hits = db.query(query_texts=[query], n_results=1)
context = hits["documents"][0][0]

prompt = f"Answer based ONLY on this:\n{context} \n\nQuestion: {query}"
print(prompt)