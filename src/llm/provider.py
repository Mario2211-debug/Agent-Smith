"""Provider-independent LLM interface used by the agent loop.

The loop only knows :class:`LLMProvider`; switching provider (or chaining
several with :class:`FallbackProvider`) needs no change in the loop.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

Message = Dict[str, str]


@dataclass
class Completion:
    """One successful LLM answer plus the metrics the subject requires.

    Attributes:
        text: raw generated text.
        input_tokens / output_tokens: usage reported by the API (reasoning
            tokens are included in output_tokens by OpenAI-compatible APIs).
        latency_ms: time spent in API calls, waits between retries excluded.
        requests: number of HTTP requests made (1 + retries).
        api_url / model_name: endpoint that produced the answer.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    requests: int
    api_url: str
    model_name: str

    @property
    def retries(self) -> int:
        """Requests made after the first one."""
        return self.requests - 1


class ProviderError(RuntimeError):
    """Raised when no answer could be obtained (after retries/rotation).

    ``requests`` and ``latency_ms`` keep the cost of the failed attempts so
    the agent can still report accurate usage.
    """

    def __init__(self, message: str, requests: int = 0,
                 latency_ms: float = 0.0):
        """Store the message and the cost of the failed attempts."""
        super().__init__(message)
        self.requests = requests
        self.latency_ms = latency_ms


class LLMProvider(ABC):
    """Chat-completion provider."""

    api_url: str = ""
    model_name: str = ""

    @abstractmethod
    def complete(self, messages: List[Message], max_tokens: int,
                 stop: Optional[List[str]] = None,
                 timeout: float = 60.0) -> Completion:
        """Return a completion for ``messages`` within ``timeout`` seconds.

        Raises:
            ProviderError: if no answer could be obtained.
        """


class FallbackProvider(LLMProvider):
    """Try several providers in order until one answers."""

    def __init__(self, providers: List[LLMProvider]):
        """Keep the providers; the first one is the preferred one."""
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider.")
        self.providers = providers
        self.api_url = providers[0].api_url
        self.model_name = providers[0].model_name

    def complete(self, messages: List[Message], max_tokens: int,
                 stop: Optional[List[str]] = None,
                 timeout: float = 60.0) -> Completion:
        """Return the first successful completion, summing failed costs."""
        requests, latency, errors = 0, 0.0, []
        deadline = time.monotonic() + timeout
        for provider in self.providers:
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                errors.append("time budget exhausted")
                break
            try:
                result = provider.complete(messages, max_tokens, stop,
                                           remaining)
            except ProviderError as error:
                requests += error.requests
                latency += error.latency_ms
                errors.append(f"{provider.model_name}: {error}")
                continue
            result.requests += requests
            result.latency_ms += latency
            return result
        raise ProviderError("All providers failed: " + "; ".join(errors),
                            requests, latency)
