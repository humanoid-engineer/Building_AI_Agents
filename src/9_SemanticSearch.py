from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

notes = [
"Python is great for AI applications.",
"Chennai is a coastal city in Tamil Nadu.",
"Transformers power modern AI models."
]
vecs = model.encode(notes)

query = input("Ask something: ") # I am an Indian Citizen
query_vec = model.encode(query)
scores = util.cos_sim(query_vec, vecs)
best = scores.argmax()
print("Match:", notes[best])