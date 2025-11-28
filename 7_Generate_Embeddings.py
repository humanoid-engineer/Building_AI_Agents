from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence_1 = [
"I love playing football.",
"Soccer is my favorite sport."
]

emb = model.encode(sentence_1)
similarity = util.cos_sim(emb[0], emb[1])
print("Sentence similarity:", similarity.item()) ## High similarity expected

print(20 * "--")

sentence_2 = ["I love football",
"The sky is brown"
]

emb = model.encode(sentence_2)
similarity = util.cos_sim(emb[0], emb[1])
print("Sentence similarity:", similarity.item()) ## Low similarity expecte
