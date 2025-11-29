import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

goal = input("Enter the goal for the agent: ")

prompt = """
You are an agent. You have to find which tool to use to achive the goal. 
Tools are search(query), calculator(query).

FORMAT:
Thought:
Action: search["..."] or calculator["..."]

If you unable to find the tool, respond "Tool Not Found".

"""
prompt += f"Goal: {goal}\n"

resp = client.chat.completions.create(
model="gpt-4.1-mini",
messages=[{"role": "user", "content": prompt}]
)
print(resp.choices[0].message.content)