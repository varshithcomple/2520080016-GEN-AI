
from abc import ABC, abstractmethod

from google import genai
import requests


# Common interface
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate and return a text response."""
        raise NotImplementedError


# Gemini cloud provider
class GeminiProvider(LLMProvider):

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text


# Ollama local provider
class OllamaProvider(LLMProvider):

    def __init__(
        self,
        model: str = "llama3.2",
        url: str = "http://localhost:11434/api/generate"
    ):
        self.model = model
        self.url = url

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Cannot connect to Ollama. Check that Ollama is installed "
                "and running, then try again."
            ) from exc

        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Ollama request timed out.") from exc

        data = response.json()
        return data.get("response", "")


# Common client
class LLMClient:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, prompt: str) -> str:
        return self.provider.generate(prompt)
