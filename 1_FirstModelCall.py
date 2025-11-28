from openai import OpenAI
client = OpenAI(api_key='') #Get API Key from https://platform.openai.com/api-keys

response = client.chat.completions.create(
model="gpt-4.1-mini",
messages=[{"role": "user", "content": "Explain AI in one sentence"}]
)
print(response.choices[0].message.content)