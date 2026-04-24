from dotenv import load_dotenv
from openai import OpenAI

import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b:freeze-2024-06-01",
    messages=[{"role": "user", "content": "¿Cuál es la capital de Francia?"}],
)

print(response.choices[0].message.content)


print(f"Mi clave es: {api_key}")  # <--- AÑADE ESTO PARA TESTEAR
