import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

prompt = """
You are a helpful assistant.
Task: Extract details from the text and return in
JSON.
Fields: name, city, topic
Text: "Hi, I'm Asha from Mumbai, and I need help
understanding insurance claims."
"""
resp = client.chat.completions.create(
	model="gpt-4.1-mini",
	messages=[{"role": "user", "content": prompt}],
)
try:
	print(resp.choices[0].message.content)
except Exception:
	# Fallback: print the raw response
	print(resp)