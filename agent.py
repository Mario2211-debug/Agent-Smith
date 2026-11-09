import os
from typing import Any
from openai import OpenAI


class Agent:
    def __init__(self, model_name: str, provider_url: str):
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise SystemExit("Set the GROQ_API_KEY environment variable before running.")

        self.client = OpenAI(api_key=self.api_key, base_url=provider_url, timeout=30.0, max_retries=0)

    def run(self, prompt: str) -> Any:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0,
        )
        return response
