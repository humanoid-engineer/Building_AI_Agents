import os
import json
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

user_input = "Hi, I'm Sachin from Mumbai, and I need help practicing bowling."

with open("./prompts/prompts_template.json") as f:
    prompts = json.load(f)
prompt = prompts["extract_user"]["template"].format(text=user_input)

resp = client.chat.completions.create(
model="gpt-4.1-mini",
messages=[{"role": "user", "content": prompt}]
)
print(resp.choices[0].message.content)