from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = ["Machine Learning is a subset of AI", "I visited Chennai last week."]
embeddings = model.encode(sentences)

query = model.encode("Tell me about Machine Learning")
scores = util.cos_sim(query, embeddings)
print(scores)