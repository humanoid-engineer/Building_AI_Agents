import tiktoken
encoder = tiktoken.encoding_for_model("gpt-4.1-mini")

tokens = encoder.encode("Chennai is amazing!")
print(tokens)
