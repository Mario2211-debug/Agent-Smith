"""Provider for any OpenAI-compatible chat API (OpenRouter, Groq, ...).

Errors are classified by HTTP status / exception name, so this module does
not depend on the exception classes of a specific SDK version:

* 429                 -> put the key on cooldown, rotate, retry
* 401 / 403           -> disable the key, rotate, retry
* timeout / 5xx / net -> back off and retry with the same key
* anything else       -> fail immediately
"""

import time
from typing import Any, Callable, List, Optional

from src.llm.keys import KeyPool, NoKeyAvailable
from src.llm.provider import Completion, LLMProvider, Message, ProviderError

MAX_REQUESTS = 6
REQUEST_TIMEOUT_SECONDS = 60.0
RATE_LIMIT_COOLDOWN_SECONDS = 30.0
MIN_USEFUL_SECONDS = 2.0


def _default_client_factory(api_key: str, base_url: str) -> Any:
    """Create an ``openai.OpenAI`` client without its own retries."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


class OpenAICompatibleProvider(LLMProvider):
    """Chat completions with key rotation, retries and usage tracking."""

    def __init__(self, model_name: str, api_url: str, keys: KeyPool,
                 client_factory: Callable[[str, str], Any] = None,
                 extra_body: Optional[dict] = None):
        """Configure the endpoint; ``client_factory`` is injectable for tests."""
        self.model_name = model_name
        self.api_url = api_url
        self.keys = keys
        self.extra_body = extra_body or {}
        self._factory = client_factory or _default_client_factory
        self._clients: dict = {}

    def _client(self, key: str) -> Any:
        """Return (and cache) the client bound to ``key``."""
        if key not in self._clients:
            self._clients[key] = self._factory(key, self.api_url)
        return self._clients[key]

    def complete(self, messages: List[Message], max_tokens: int,
                 stop: Optional[List[str]] = None,
                 timeout: float = 60.0) -> Completion:
        """Request a completion, rotating keys and retrying transient errors."""
        deadline = time.monotonic() + timeout
        requests, latency_ms, last_error = 0, 0.0, "no request made"
        while requests < MAX_REQUESTS:
            remaining = deadline - time.monotonic()
            if remaining < MIN_USEFUL_SECONDS:
                break
            try:
                key = self.keys.current()
            except NoKeyAvailable as error:
                last_error = str(error)
                wait = self.keys.seconds_until_available()
                if not wait or wait > remaining - MIN_USEFUL_SECONDS:
                    break
                time.sleep(wait)
                continue
            requests += 1
            started = time.monotonic()
            try:
                response = self._client(key).chat.completions.create(
                    model=self.model_name, messages=messages,
                    max_tokens=max_tokens, stop=stop or None, temperature=0,
                    timeout=min(REQUEST_TIMEOUT_SECONDS, remaining),
                    **({"extra_body": self.extra_body}
                       if self.extra_body else {}))
                latency_ms += (time.monotonic() - started) * 1000
                return self._to_completion(response, latency_ms, requests)
            except Exception as error:  # noqa: BLE001 - classified below
                latency_ms += (time.monotonic() - started) * 1000
                last_error = f"{type(error).__name__}: {error}"
                if not self._handle(error, key, requests, deadline):
                    break
        raise ProviderError(f"{self.model_name} at {self.api_url}: "
                            f"{last_error}", requests, latency_ms)

    def _handle(self, error: Exception, key: str, attempt: int,
                deadline: float) -> bool:
        """React to an API error; return True if a retry makes sense."""
        status = getattr(error, "status_code", None)
        name = type(error).__name__
        if status == 429 or name == "RateLimitError":
            self.keys.rate_limited(key, RATE_LIMIT_COOLDOWN_SECONDS)
            return True
        if status in (401, 403) or name in ("AuthenticationError",
                                            "PermissionDeniedError"):
            self.keys.invalid(key)
            return True
        transient = (status is not None and status >= 500) or any(
            word in name for word in ("Timeout", "Connection"))
        if not transient:
            return False
        pause = min(2 ** attempt, deadline - time.monotonic()
                    - MIN_USEFUL_SECONDS)
        if pause > 0:
            time.sleep(pause)
        return True

    def _to_completion(self, response: Any, latency_ms: float,
                       requests: int) -> Completion:
        """Convert an SDK response into a :class:`Completion`."""
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ProviderError("The provider did not return token usage.",
                                requests, latency_ms)
        choices = getattr(response, "choices", None) or []
        text = (choices[0].message.content or "") if choices else ""
        return Completion(text=text, input_tokens=usage.prompt_tokens or 0,
                          output_tokens=usage.completion_tokens or 0,
                          latency_ms=latency_ms, requests=requests,
                          api_url=self.api_url, model_name=self.model_name)
