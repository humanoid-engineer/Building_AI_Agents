import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

prompt = """
You are an agent. Use only the tool: search(query).
Goal: Find EV sales growth in India.
FORMAT:

Thought:
Action: search["..."]
"""
resp = client.chat.completions.create(
model="gpt-4.1-mini",
messages=[{"role": "user", "content": prompt}]
)
print(resp.choices[0].message.content)