import requests

class LocalLLM:

    def __init__(self, model="llama3:latest"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:

        payload = {
        "model": self.model,
        "prompt": prompt,
        "stream": False
        }

        response = requests.post(self.url, json=payload)
        return response.json()["response"]