import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

prompt = """
You are an AI that classifies customer feedback as Positive, Neutral, or Negative.

Examples:

Text: "The movie was okay, nothing special" → neutral
Text: "Worst experience watching this movie" → negative

Now classify the sentiment of the following text:
Text: "I loved watching Enthiran (Robo) movie" → 

Output:
{
    "Movie": "Name of the movie",
    "sentiment": ""
}

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