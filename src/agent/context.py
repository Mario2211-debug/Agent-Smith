"""Agent context: limits, cumulative usage and the conversation history.

Token limits are cumulative over the whole task and every request resends
the history, so the conversation can be compacted to fit what is left.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CHARS_PER_TOKEN = 3  # conservative: real tokenizers average ~4 chars/token
MESSAGE_OVERHEAD_TOKENS = 4
COMPACT_OBSERVATION_CHARS = 300


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Return a conservative token estimate for a list of chat messages."""
    return sum(len(m["content"]) // CHARS_PER_TOKEN + MESSAGE_OVERHEAD_TOKENS
               for m in messages)


@dataclass
class Budget:
    """Hard limits of one task (see the subject's section VI.1)."""

    max_iterations: int
    max_input_tokens: int
    max_output_tokens: int
    timeout_seconds: float
    max_output_tokens_per_step: int = 2000
    safety_margin_seconds: float = 3.0

    @classmethod
    def mbpp(cls) -> "Budget":
        """Limits of the MBPP benchmark."""
        return cls(10, 6_000, 1_500, 120.0, max_output_tokens_per_step=600)

    @classmethod
    def swebench(cls) -> "Budget":
        """Limits of the SWE-bench benchmark."""
        return cls(30, 300_000, 10_000, 900.0, max_output_tokens_per_step=1500,
                   safety_margin_seconds=20.0)


@dataclass
class Usage:
    """Cumulative usage of one task, checked against a :class:`Budget`."""

    budget: Budget
    iterations: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        """Seconds since the task started."""
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self) -> float:
        """Seconds left before the timeout, minus the safety margin."""
        return (self.budget.timeout_seconds - self.budget.safety_margin_seconds
                - self.elapsed)

    @property
    def remaining_input(self) -> int:
        """Input tokens left."""
        return self.budget.max_input_tokens - self.input_tokens

    @property
    def remaining_output(self) -> int:
        """Output tokens left."""
        return self.budget.max_output_tokens - self.output_tokens

    def exhausted(self) -> Optional[str]:
        """Return why no further step is allowed, or None."""
        if self.iterations >= self.budget.max_iterations:
            return f"iteration limit ({self.budget.max_iterations}) reached"
        if self.remaining_output <= 0:
            return "output token budget exhausted"
        if self.remaining_input <= 0:
            return "input token budget exhausted"
        if self.remaining_seconds <= 0:
            return "time budget exhausted"
        return None


class Conversation:
    """System prompt, task and the (assistant, observation) steps."""

    def __init__(self, system_prompt: str, task: str):
        """Start a conversation for one task."""
        self.system_prompt = system_prompt
        self.task = task
        self.steps: List[Dict[str, str]] = []

    def add_step(self, assistant: str, observation: str) -> None:
        """Record the model's answer and the resulting observation."""
        self.steps.append({"assistant": assistant,
                           "observation": observation})

    def messages(self, token_limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Return chat messages, compacted to fit ``token_limit`` if given.

        Compaction first shortens old observations (the last one is always
        kept whole), then drops the oldest steps.
        """
        steps = [dict(step) for step in self.steps]
        rendered = self._render(steps, dropped=0)
        if token_limit is None or estimate_tokens(rendered) <= token_limit:
            return rendered
        for step in steps[:-1]:
            if len(step["observation"]) > COMPACT_OBSERVATION_CHARS:
                step["observation"] = (step["observation"]
                                       [:COMPACT_OBSERVATION_CHARS]
                                       + "\n... [shortened to save tokens]")
        dropped = 0
        rendered = self._render(steps, dropped)
        while len(steps) > 1 and estimate_tokens(rendered) > token_limit:
            steps.pop(0)
            dropped += 1
            rendered = self._render(steps, dropped)
        return rendered

    def _render(self, steps: List[Dict[str, str]],
                dropped: int) -> List[Dict[str, str]]:
        """Build the message list from (possibly compacted) steps."""
        task = self.task
        if dropped:
            task += (f"\n\n[{dropped} earlier step(s) omitted to save tokens;"
                     " variables defined in them still exist.]")
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task}]
        for step in steps:
            messages.append({"role": "assistant",
                             "content": step["assistant"]})
            messages.append({"role": "user", "content":
                             f"Observation:\n{step['observation']}"})
        return messages
