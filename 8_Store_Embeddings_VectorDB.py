import chromadb

chroma = chromadb.Client()
collection = chroma.create_collection("notes")

collection.add( ids=["1"],
documents=["Deep learning uses neural networks to learn patterns."]
)

collection.add( ids=["2"],
documents=["Sachin is a good cricketer"]
)

collection.add( ids=["3"],
documents=["Dhoni has good friends network"]
)

results = collection.query(
query_texts=["How do networks learn?"],
#query_texts=["How to create network in cricket?"],
n_results=1
)
print(results.get("documents")[0][0])