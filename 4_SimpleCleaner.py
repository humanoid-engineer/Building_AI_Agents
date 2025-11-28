import re

def simple_cleaner(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return " ".join(text.split())

def chunk(text, n=300):
    words = text.split()
    for i in range(0, len(words), n):
        yield " ".join(words[i:i+n])

print(list(chunk(simple_cleaner("Hello, World! This is a test."))))