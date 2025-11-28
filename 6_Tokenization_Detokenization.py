import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4.1-mini")
text = "Chennai is a coastal city in India."

tokens = encoder.encode(text)
print(tokens)
print(len(tokens), "tokens")

decoded = [encoder.decode([t]) for t in tokens]
print(decoded)

