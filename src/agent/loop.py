"""The agent loop: Thought -> Code -> Observation, until final_answer().

Each iteration:
1. sends the (compacted) conversation to the LLM, with stop sequences so
   the model cannot invent observations;
2. extracts Python code from the answer (several formats supported);
3. executes it in the sandbox and feeds the observation back.

All limits (iterations, cumulative tokens, wall-clock time) are enforced
before each request, and every step is recorded as ``StepMetrics``.
"""

from datetime import datetime
from typing import Any, Optional

from src.agent.context import Budget, Conversation, Usage, estimate_tokens
from src.agent.extraction import END_CODE, extract_code
from src.llm.provider import LLMProvider, ProviderError
from structures import SolutionOutput, StepMetrics

STOP_SEQUENCES = [END_CODE, "\nObservation:"]


class AgentLoop:
    """Runs one task with a provider and a sandbox; returns SolutionOutput."""

    def __init__(self, provider: LLMProvider, sandbox: Any, budget: Budget,
                 system_prompt: str, benchmark: str, task_id: str,
                 log: Any = print):
        """Store collaborators; ``log`` receives progress lines (or None)."""
        self.provider = provider
        self.sandbox = sandbox
        self.budget = budget
        self.system_prompt = system_prompt
        self.benchmark = benchmark
        self.task_id = task_id
        self.log = log or (lambda *_: None)
        self.usage = Usage(budget)
        self.steps: list = []
        self.answer: Optional[str] = None
        self.error: Optional[str] = None

    def run(self, task_prompt: str) -> SolutionOutput:
        """Iterate until final_answer(), a limit, or a fatal error."""
        conversation = Conversation(self.system_prompt, task_prompt)
        try:
            while self.answer is None:
                reason = self.usage.exhausted()
                if reason:
                    self.error = f"Stopped: {reason}."
                    break
                if not self._step(conversation):
                    break
        except (KeyboardInterrupt, SystemExit) as control:
            self.error = f"Interrupted by {type(control).__name__}."
        return self._output()

    def _step(self, conversation: Conversation) -> bool:
        """Run one iteration; return False when the loop must stop."""
        step = self.usage.iterations + 1
        messages = conversation.messages(self.usage.remaining_input)
        if estimate_tokens(messages) > self.usage.remaining_input:
            self.error = ("Stopped: the next request would exceed the input "
                          "token budget.")
            return False
        max_tokens = min(self.budget.max_output_tokens_per_step,
                         self.usage.remaining_output)
        self.log(f"[step {step}] asking {self.provider.model_name}...")
        try:
            completion = self.provider.complete(
                messages, max_tokens, STOP_SEQUENCES,
                timeout=self.usage.remaining_seconds)
        except ProviderError as error:
            self.usage.requests += error.requests
            self.error = f"LLM request failed: {error}"
            return False
        self.usage.iterations = step
        self.usage.requests += completion.requests
        self.usage.input_tokens += completion.input_tokens
        self.usage.output_tokens += completion.output_tokens
        observation, code = self._observe(completion.text)
        conversation.add_step(completion.text, observation)
        self.steps.append(StepMetrics(
            step=step, input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            request_time_ms=completion.latency_ms,
            timestamp=datetime.now().isoformat(),
            api_url=completion.api_url, model_name=completion.model_name,
            llm_output=completion.text, sandbox_input=code,
            sandbox_output=observation, retries=completion.retries))
        self.log(f"[step {step}] observation:\n{observation[:500]}")
        return True

    def _observe(self, text: str) -> tuple:
        """Extract and execute code; return (observation, executed code)."""
        extraction = extract_code(text)
        if not extraction.code:
            return extraction.note, ""
        result = self.sandbox.execute(extraction.code,
                                      timeout=self.usage.remaining_seconds)
        observation = result.to_observation()
        if extraction.note:
            observation = f"[extraction note] {extraction.note}\n\n" \
                          f"{observation}"
        if result.is_final:
            self.answer = result.final_answer
        return observation, extraction.code

    def _output(self) -> SolutionOutput:
        """Build the SolutionOutput expected by the moulinette."""
        return SolutionOutput(
            task_id=self.task_id, benchmark=self.benchmark,
            success=self.answer is not None, solution=self.answer or "",
            iterations=self.usage.iterations,
            total_requests=self.usage.requests,
            total_input_tokens=self.usage.input_tokens,
            total_output_tokens=self.usage.output_tokens,
            total_time_seconds=self.usage.elapsed, steps=self.steps,
            system_prompt=self.system_prompt,
            error=None if self.answer is not None else
            (self.error or "No final answer."))
