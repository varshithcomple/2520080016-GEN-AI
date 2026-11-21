
import os
from dotenv import load_dotenv

from llm_client import (
    GeminiProvider,
    OllamaProvider,
    LLMClient
)

load_dotenv(override=True)

provider_name = input(
    "Choose provider (gemini/ollama): "
).strip().lower()

if provider_name == "gemini":

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Add a valid GEMINI_API_KEY in the .env file.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    provider = GeminiProvider(api_key, model)

elif provider_name == "ollama":

    provider = OllamaProvider(model="llama3.2")

else:
    raise ValueError("Invalid provider. Enter gemini or ollama.")

client = LLMClient(provider)

prompt = input("Enter your question: ").strip()

if not prompt:
    raise ValueError("Prompt cannot be empty.")

try:
    answer = client.generate(prompt)

    print("\nLLM Response:")
    print(answer)

except Exception as error:
    print(f"\nError: {error}")
