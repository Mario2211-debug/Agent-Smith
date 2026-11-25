"""Build a provider from a model name and an API URL.

Each known endpoint maps to the environment-variable prefix of its keys
(see :mod:`src.llm.keys`). Unknown OpenAI-compatible endpoints use
``LLM_API_KEY``. Adding a provider is a one-line change in ``KEY_PREFIXES``.
"""

from typing import List, Optional, Tuple

from src.llm.keys import KeyPool, load_keys
from src.llm.openai_compat import OpenAICompatibleProvider
from src.llm.provider import FallbackProvider, LLMProvider

KEY_PREFIXES = {
    "openrouter.ai": "OPENROUTER_API_KEY",
    "api.groq.com": "GROQ_API_KEY",
    "api.together.xyz": "TOGETHER_API_KEY",
    "generativelanguage.googleapis.com": "GEMINI_API_KEY",
    "api.mistral.ai": "MISTRAL_API_KEY",
}
DEFAULT_PREFIX = "LLM_API_KEY"
# Disable "thinking" where the API supports it: reasoning tokens count
# against the token limits.
EXTRA_BODY = {"openrouter.ai": {"reasoning": {"enabled": False}}}


def key_prefix(api_url: str) -> str:
    """Return the environment-variable prefix holding keys for ``api_url``."""
    for host, prefix in KEY_PREFIXES.items():
        if host in api_url:
            return prefix
    return DEFAULT_PREFIX


def make_provider(model_name: str, api_url: str) -> LLMProvider:
    """Create an OpenAI-compatible provider with every configured key.

    Raises:
        SystemExit: with a readable message if no key is configured.
    """
    prefix = key_prefix(api_url)
    keys = load_keys(prefix)
    if not keys:
        raise SystemExit(f"No API key for {api_url}: set {prefix} "
                         f"(or {prefix}1, {prefix}2, ...).")
    extra = next((body for host, body in EXTRA_BODY.items()
                  if host in api_url), None)
    return OpenAICompatibleProvider(model_name, api_url.rstrip("/"),
                                    KeyPool(keys), extra_body=extra)


def make_fallback(primary: Tuple[str, str],
                  fallbacks: Optional[List[Tuple[str, str]]] = None
                  ) -> LLMProvider:
    """Chain ``(model, url)`` pairs; providers without keys are skipped."""
    providers = [make_provider(*primary)]
    for model_name, api_url in fallbacks or []:
        try:
            providers.append(make_provider(model_name, api_url))
        except SystemExit:
            continue
    return providers[0] if len(providers) == 1 else \
        FallbackProvider(providers)
