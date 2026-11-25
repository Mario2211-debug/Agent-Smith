"""API key pool with rotation, loaded from environment variables.

For a prefix such as ``OPENROUTER_API_KEY`` every one of these is accepted:
``OPENROUTER_API_KEY``, ``OPENROUTER_API_KEY1`` .. ``OPENROUTER_API_KEY99``,
``OPENROUTER_API_KEY_1`` .. and a comma-separated ``OPENROUTER_API_KEYS``.
Keys are never hard-coded.
"""

import os
import time
from typing import Dict, List, Mapping, Optional


class NoKeyAvailable(RuntimeError):
    """Raised when every key of the pool is cooling down or invalid."""


def load_keys(prefix: str,
              environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Return the distinct non-empty keys found for ``prefix``, in order."""
    env = os.environ if environ is None else environ
    names = [prefix] + [f"{prefix}{sep}{index}" for index in range(1, 100)
                        for sep in ("", "_")]
    keys = [env.get(name, "").strip() for name in names]
    keys += [key.strip() for key in env.get(f"{prefix}S", "").split(",")]
    return list(dict.fromkeys(key for key in keys if key))


class KeyPool:
    """Round-robin key rotation with per-key cooldown.

    A rate-limited key is put on cooldown; an invalid key is disabled for
    the rest of the run. ``current()`` always returns a usable key.
    """

    def __init__(self, keys: List[str]):
        """Create the pool; at least one key is required.

        Raises:
            ValueError: if ``keys`` is empty.
        """
        if not keys:
            raise ValueError("No API key configured.")
        self.keys = list(keys)
        self._index = 0
        self._cooldown_until: Dict[str, float] = {}
        self._disabled: set = set()

    def current(self) -> str:
        """Return the active key, skipping cooling or disabled keys.

        Raises:
            NoKeyAvailable: with the wait time if no key is usable now.
        """
        now = time.monotonic()
        for offset in range(len(self.keys)):
            index = (self._index + offset) % len(self.keys)
            key = self.keys[index]
            if key not in self._disabled and \
                    self._cooldown_until.get(key, 0) <= now:
                self._index = index
                return key
        usable = [until for key, until in self._cooldown_until.items()
                  if key not in self._disabled]
        if not usable:
            raise NoKeyAvailable("All API keys were rejected as invalid.")
        raise NoKeyAvailable(f"All API keys are rate limited; next one is "
                             f"available in {min(usable) - now:.1f}s.")

    def seconds_until_available(self) -> float:
        """Return how long until some non-disabled key leaves cooldown."""
        now = time.monotonic()
        waits = [max(until - now, 0) for key, until in
                 self._cooldown_until.items() if key not in self._disabled]
        return min(waits) if waits else 0.0

    def rate_limited(self, key: str, cooldown: float = 30.0) -> None:
        """Put ``key`` on cooldown and move to the next key."""
        self._cooldown_until[key] = time.monotonic() + cooldown
        self._index = (self.keys.index(key) + 1) % len(self.keys)

    def invalid(self, key: str) -> None:
        """Disable ``key`` permanently (authentication failure)."""
        self._disabled.add(key)
        self._index = (self.keys.index(key) + 1) % len(self.keys)
