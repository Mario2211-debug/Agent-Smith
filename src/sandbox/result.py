"""Execution result of one sandbox run and its text form for the LLM."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExecutionResult:
    """Everything the agent loop needs to know about one code execution.

    Attributes:
        output: captured stdout/stderr (possibly truncated).
        error: formatted error (traceback, security violation...) or None.
        final_answer: argument passed to ``final_answer()``, if called.
        timed_out: True when the execution hit the configured timeout.
        truncated: True when ``output`` was cut to the size limit.
        notes: explicit explanations for the LLM (truncation, resets...).
        duration_seconds: wall-clock time of sandboxed code only.
    """

    output: str = ""
    error: Optional[str] = None
    final_answer: Optional[str] = None
    timed_out: bool = False
    truncated: bool = False
    notes: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def is_final(self) -> bool:
        """True when the code called ``final_answer()``."""
        return self.final_answer is not None

    @property
    def success(self) -> bool:
        """True when the code ran to completion without an error."""
        return self.error is None and not self.timed_out

    def to_observation(self) -> str:
        """Render the result as the observation text fed back to the LLM.

        Every abnormal situation is stated explicitly so the model never
        has to guess what happened.
        """
        sections = []
        if self.output:
            sections.append(f"Output:\n{self.output.rstrip()}")
        if self.error:
            sections.append(f"Error:\n{self.error.rstrip()}")
        if self.is_final:
            sections.append("final_answer() was called: the task is "
                            "submitted.")
        for note in self.notes:
            sections.append(f"[sandbox note] {note}")
        if not sections:
            sections.append("Code executed successfully but printed nothing. "
                            "Use print() to see values.")
        return "\n\n".join(sections)
