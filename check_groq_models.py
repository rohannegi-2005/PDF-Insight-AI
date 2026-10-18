from dotenv import load_dotenv
import os
load_dotenv()

from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
print("Key loaded (first 10 chars):", api_key[:10] if api_key else "NOT FOUND")

client = Groq(api_key=api_key)
models = client.models.list()
print("\nAvailable models for this key:")
for m in models.data:
    print(" -", m.id)