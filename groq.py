import os
from openai import OpenAI

from structures import ModelResponse


class GroqAPI:
    def __init__(self, model_name: str, provider_url: str):
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise SystemExit("Set the GROQ_API_KEY environment variable before running.")

        self.client = OpenAI(api_key=self.api_key, base_url=provider_url, timeout=30.0, max_retries=0)

    def run(self, prompt: str) -> ModelResponse:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0,
        )
        if response.usage is None:
            raise ValueError("The provider did not return token usage.")
        return ModelResponse(
            text=response.output_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
