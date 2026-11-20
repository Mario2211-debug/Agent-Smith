# Load .env with: uv run --env-file .env python ...
import os
from openai import OpenAI

from structures import ModelResponse


class OpenRouterAPI:
    def __init__(self, model_name: str, provider_url: str = "https://openrouter.ai/api/v1"):
        self.model_name = model_name
        self.api_key1 = os.environ.get("OPENROUTER_API_KEY1")
        self.api_key2 = os.environ.get("OPENROUTER_API_KEY2")
        initial_key = self.api_key1 or self.api_key2
        if not initial_key:
            raise SystemExit("Set OPENROUTER_API_KEY1 or OPENROUTER_API_KEY2 before running.")

        self.client = OpenAI(api_key=initial_key, base_url=provider_url, timeout=30.0, max_retries=0)

    def use_first_key(self) -> bool:
        if not self.api_key1 or self.client.api_key == self.api_key1:
            return False
        self.client.api_key = self.api_key1
        return True

    def use_second_key(self) -> bool:
        if not self.api_key2 or self.client.api_key == self.api_key2:
            return False
        self.client.api_key = self.api_key2
        return True

    def run(
        self, prompt: str, *, system_prompt: str | None = None,
        max_output_tokens: int = 2000, timeout: float = 30.0,
    ) -> ModelResponse:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            extra_body={"reasoning": {"enabled": False}},
            max_completion_tokens=max_output_tokens,
            timeout=timeout,
        )
        if response.usage is None:
            raise ValueError("The provider did not return token usage.")
        return ModelResponse(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
