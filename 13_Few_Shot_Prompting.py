import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

prompt = """
You are an AI that classifies customer feedback as Positive, Neutral, or Negative.

Examples:

Text: "I love this product" → positive
Text: "Worst experience ever" → negative

Now classify the sentiment of the following text:
Text: "The movie was okay, nothing special" →

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