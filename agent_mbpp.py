r"""Run these commands from the project root.

Run cached task 407 with GPT-OSS 120B on Groq. The agent opens its own MCP server.
    uv run --env-file .env python -m agent_mbpp \
        --task-file cache/mbpp_task_407.json --output cache/mbpp_solution_mcp_120b.json \
        --model-name openai/gpt-oss-120b --provider-url https://api.groq.com/openai/v1

Run the same task with OpenRouter.
    uv run --env-file .env python -m agent_mbpp \
        --task-file cache/mbpp_task_407.json --output cache/mbpp_solution_mcp_hy4.json \
        --model-name tencent/hy4-flash --provider-url https://openrouter.ai/api/v1

Validate the 120B solution and its metrics with the local Python sandbox.
    uv run --project moulinette moulinette_eval validate mbpp \
        cache/mbpp_task_407.json cache/mbpp_solution_mcp_120b.json --mbpp-backend sandbox

Run the five-task MBPP exam with GPT-OSS 120B and local sandbox validation.
    bash exams/exam_mbpp.sh --student-path . --moulinette-path moulinette \
        --env-file .env --model-name openai/gpt-oss-120b \
        --provider-url https://api.groq.com/openai/v1
"""
import argparse
import ast
from src.sandbox.sandbox import Sandbox
import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import tiktoken
from openai import APITimeoutError, RateLimitError
from pydantic import ValidationError

from src.APIs.groq import GroqAPI
from src.APIs.openrouter import OpenRouterAPI
from src.mcp.client import MCPClient
from src.sandbox.result import ExecutionResult
from structures import AttemptResult, MBPPTaskInput, ModelResponse
from structures import SolutionOutput, StepMetrics

MAX_ATTEMPTS = 10
MAX_RATE_LIMIT_RETRIES = 2
RATE_LIMIT_DELAY_SECONDS = 30
INPUT_TOKEN_LIMIT = 6000
OUTPUT_TOKEN_LIMIT = 1500
TASK_TIME_LIMIT_SECONDS = 120.0
RESULT_WRITE_MARGIN_SECONDS = 1.0
API_TIMEOUT_SECONDS = 30.0
ERROR_COLOR = "\033[31m"
WARNING_COLOR = "\033[33m"
RESET_COLOR = "\033[0m"
SYSTEM_PROMPT = (
    "Implement the requested function correctly, including edge cases. "
    "Prefer a simple, efficient solution. "
    "Return one fenced Python block as your text response. The runner executes it "
    "locally in a persistent sandbox. Write run_tests and final_answer as normal "
    "Python calls inside that block. Do not emit API tool calls.\n\n"
    "The sandbox already defines run_tests and final_answer. Never redefine them "
    "or write your own tests. Put the requested function only inside the string code. "
    "Keep the solution short, with no comments or docstrings.\n\n"
    "Example testing step when public tests are available:\n"
    "```python\ncode = 'def add(a, b): return a + b'\n"
    "print(run_tests(code))\n```\n\n"
    "Wait for the observation. If the tests fail, update code and retest. "
    "If they pass, your entire next response is:\n"
    "```python\nfinal_answer(code)\n```\n"
    "Pass the source string to final_answer, never the function object. "
    "If there are no public tests, define code and call final_answer(code) in the same block."
)
MODELS = {
    "https://api.groq.com/openai/v1": {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    },
    "https://openrouter.ai/api/v1": {
        "nvidia/nemotron-3.5-lightning:free",
        "tencent/hy4-flash",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    },
}
START = perf_counter()


def format_diagnostic(message: str, color: str = ERROR_COLOR) -> str:
    if sys.stderr.isatty():
        return f"{color}{message}{RESET_COLOR}"
    return message


def print_diagnostic(message: str, color: str = ERROR_COLOR) -> None:
    print(format_diagnostic(message, color), file=sys.stderr)


class MBPPArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(format_diagnostic(message))


class TokenBudgetError(RuntimeError):
    pass


@dataclass
class RequestMetrics:
    request_time_ms: float = 0.0
    retries: int = 0


@dataclass
class SandboxExecution:
    code: str = ""
    output: str = ""


class MBPPRunner:
    def __init__(self, agent: GroqAPI | OpenRouterAPI,
                 task: MBPPTaskInput,
                 client: MCPClient,
                 max_iterations: int = MAX_ATTEMPTS,):
        self.agent = agent
        self.task = task
        self.max_iterations = max_iterations
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.steps: list[StepMetrics] = []
        self.mcp = client

    @staticmethod
    def extract_code(text: str) -> tuple[str, str]:
        code = text.strip()
        feedback = ""
        opening = re.search(r"^[ \t]*```(?:python|py)?[ \t]*\r?$", code, re.MULTILINE | re.IGNORECASE)
        if opening:
            block = code[opening.end():]
            closing = re.search(r"^[ \t]*```[ \t]*\r?$", block, re.MULTILINE)
            if closing:
                code = block[:closing.start()].strip()
            else:
                code = block.strip()
                feedback = (
                    "The Python block had no closing Markdown fence. "
                    "I treated the rest of the response as code."
                )

        try:
            tree = ast.parse(code)
            compile(tree, "<model response>", "exec")
        except SyntaxError as error:
            return "", f"No valid Python code found ({error.msg}). Return executable Python code."

        if not tree.body:
            return "", "No executable Python code found in the response."
        return code, feedback

    @staticmethod
    def load_task(task_file: str) -> MBPPTaskInput:
        try:
            with open(task_file, encoding="utf-8") as file:
                return MBPPTaskInput.model_validate(json.load(file))
        except FileNotFoundError as error:
            raise SystemExit(f"Task file '{task_file}' not found.") from error
        except json.JSONDecodeError as error:
            raise SystemExit(f"Task file '{task_file}' is not a valid JSON file.") from error
        except ValidationError as error:
            raise SystemExit(f"Task file '{task_file}' does not conform to the expected structure: {error}") from error
        except OSError as error:
            raise SystemExit(f"Could not read task file '{task_file}': {error}") from error

    @staticmethod
    def save_solution(output_file: str, solution: SolutionOutput) -> None:
        try:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(solution.model_dump_json(indent=4),
                            encoding="utf-8")
        except OSError as error:
            raise SystemExit(
                f"Could not write solution to '{output_file}': {error}"
            ) from error

    def build_prompt(self) -> str:
        return (
            f"{self.task.task_definition}\n"
            f"{self.task.function_definition}\n"
            f"Public tests: {json.dumps(self.task.test_list)}"
        )

    def request_solution(self, prompt: str, metrics: RequestMetrics | None = None) -> ModelResponse:
        if metrics is None:
            metrics = RequestMetrics()

        remaining_output_tokens = OUTPUT_TOKEN_LIMIT - self.total_output_tokens
        if remaining_output_tokens <= 0:
            raise TokenBudgetError("MBPP output token budget exhausted.")

        remaining_input_tokens = INPUT_TOKEN_LIMIT - self.total_input_tokens
        if remaining_input_tokens <= 0:
            raise TokenBudgetError("MBPP input token budget exhausted.")

        tokenizer = tiktoken.get_encoding("o200k_harmony")
        system_tokens = len(tokenizer.encode_ordinary(SYSTEM_PROMPT))
        prompt_tokens = system_tokens + len(tokenizer.encode_ordinary(prompt))
        print(f"Estimated input tokens: {prompt_tokens}. Remaining budget: {remaining_input_tokens}.")
        if prompt_tokens > remaining_input_tokens:
            raise TokenBudgetError("Estimated prompt size exceeds the remaining MBPP input token budget.")

        deadline = START + TASK_TIME_LIMIT_SECONDS - RESULT_WRITE_MARGIN_SECONDS
        switched = False
        while True:
            start_time = perf_counter()
            remaining = deadline - start_time
            if remaining <= 0:
                raise TimeoutError("MBPP time budget exhausted.")

            try:
                if switched:
                    metrics.retries += 1
                self.total_requests += 1
                try:
                    response = self.agent.run(
                        prompt,
                        system_prompt=SYSTEM_PROMPT,
                        max_output_tokens=remaining_output_tokens,
                        timeout=min(API_TIMEOUT_SECONDS, remaining),
                    )
                finally:
                    metrics.request_time_ms += (perf_counter() - start_time) * 1000
                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens
                return response
            except APITimeoutError as error:
                if remaining <= API_TIMEOUT_SECONDS or perf_counter() >= deadline:
                    raise TimeoutError("MBPP time budget exhausted.") from error
                raise
            except RateLimitError as e:
                if metrics.retries >= MAX_RATE_LIMIT_RETRIES:
                    raise
                if switched:
                    remaining = deadline - perf_counter()
                    time_to_wait = min(RATE_LIMIT_DELAY_SECONDS, remaining / 3)
                    if time_to_wait < 3:
                        raise TimeoutError("Not enough time to retry after a rate limit.") from e
                    print(f"Retrying in {time_to_wait:.1f} seconds...")
                    sleep(time_to_wait)
                if not self.agent.use_first_key():
                    self.agent.use_second_key()
                    print("Switched to second API key due to rate limit.")
                else:
                    print("Switched to first API key due to rate limit.")
                switched = True

    def test_solution(self, code: str, execution: SandboxExecution, sandbox: Sandbox) -> ExecutionResult:
        deadline = START + TASK_TIME_LIMIT_SECONDS - RESULT_WRITE_MARGIN_SECONDS
        remaining = deadline - perf_counter()

        if remaining <= 0:
            raise TimeoutError("MBPP time budget exhausted before tests.")

        execution.code = code
        result = sandbox.execute(code, timeout=min(30.0, remaining))
        if result.final_answer is not None:
            source, feedback = self.extract_code(result.final_answer)
            result.final_answer = source or None
            if not source:
                result.error = (
                    "Invalid submission. Pass the complete Python source as a string "
                    "to final_answer(code). Do not pass the function object. " + feedback
                )
        execution.output = result.to_observation()

        if perf_counter() >= deadline:
            raise TimeoutError("MBPP time budget exhausted during tests.")

        return result

    def run_attempt(self, attempt: int, prompt: str, sandbox: Sandbox) -> AttemptResult:
        answer = solution = output = parsing_feedback = ""
        input_tokens = output_tokens = 0
        request_metrics = RequestMetrics()
        execution = SandboxExecution()
        timestamp = datetime.now().isoformat()
        passed = stop = completed = False

        try:
            print(f"Attempt {attempt}: Sending prompt to agent...")
            response = self.request_solution(prompt, request_metrics)
            answer = response.text
            timestamp = datetime.now().isoformat()
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            code, parsing_feedback = self.extract_code(answer)
            if not code:
                output = parsing_feedback
                parsing_feedback = ""
                print_diagnostic(output)
            else:
                result = self.test_solution(code, execution, sandbox)
                output = execution.output
                if result.is_final:
                    solution = result.final_answer or ""
                    passed = True
            completed = True
        except Exception as error:
            print_diagnostic(f"Attempt {attempt} failed: {error}")
            output = str(error)
            stop = isinstance(error, (RateLimitError, TimeoutError, TokenBudgetError))

        self.steps.append(StepMetrics(
            step=attempt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_time_ms=request_metrics.request_time_ms,
            timestamp=timestamp,
            api_url=str(self.agent.client.base_url),
            model_name=self.agent.model_name,
            llm_output=answer,
            sandbox_input=execution.code,
            sandbox_output=execution.output,
            retries=request_metrics.retries,
        ))
        return AttemptResult(passed, answer, solution, output, stop, completed, parsing_feedback)

    def build_result(
        self, attempt: int, passed: bool, solution: str, output: str,
    ) -> SolutionOutput:
        return SolutionOutput(
            task_id=str(self.task.task_id),
            benchmark="mbpp",
            success=passed,
            solution=solution,
            iterations=attempt,
            total_requests=self.total_requests,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_time_seconds=perf_counter() - START,
            steps=self.steps,
            system_prompt=SYSTEM_PROMPT if self.total_requests else "",
            error=None if passed else output,
        )

    def agent_cycle(self) -> SolutionOutput:
        prompt = self.build_prompt()
        solution = output = ""
        passed = False
        with Sandbox(mcp=self.mcp) as sandbox:
            for attempt in range(1, self.max_iterations + 1):
                result = self.run_attempt(attempt, prompt, sandbox)
                if result.solution:
                    solution = result.solution
                passed, output = result.passed, result.output
                if result.completed and not passed:
                    if result.parsing_feedback:
                        prompt += f"\n{result.parsing_feedback}"
                    prompt += f"\nObservation:\n{output}"
                if result.stop:
                    break
                if passed:
                    print("Solution submitted.")
                    print(solution)
                    break
                if attempt == self.max_iterations:
                    print_diagnostic("Max attempts reached. Exiting.")

        return self.build_result(attempt, passed, solution, output)


def main():
    parser = MBPPArgumentParser()
    parser.add_argument("--task-file", required=True, help="Path to the MBPP task JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    parser.add_argument("--max-iterations", "--max_iterations", type=int, default=MAX_ATTEMPTS, help="Maximum number of attempts (1-10)")
    parser.add_argument("--model-name", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--provider-url", default="https://api.groq.com/openai/v1", help="Provider API URL")
    args = parser.parse_args()
    if args.provider_url not in MODELS:
        parser.error(
            f"Provider URL '{args.provider_url}' is not supported. "
            f"Supported URLs: {', '.join(MODELS)}"
        )

    if args.model_name not in MODELS[args.provider_url]:
        parser.error(
            f"Model '{args.model_name}' is not supported for "
            f"provider '{args.provider_url}'. "
            f"Supported models: {', '.join(sorted(MODELS[args.provider_url]))}"
        )
    if not 1 <= args.max_iterations <= MAX_ATTEMPTS:
        parser.error(f"max_iterations must be between 1 and {MAX_ATTEMPTS}.")

    task = MBPPRunner.load_task(args.task_file)
    if args.provider_url == "https://api.groq.com/openai/v1":
        smith = GroqAPI(args.model_name, args.provider_url)
    else:
        smith = OpenRouterAPI(args.model_name, args.provider_url)

    server = Path(__file__).resolve().with_name("mcp_tools_mbpp.py")
    # Build the MCP server command with the current Python
    # and quote paths with spaces.
    command = shlex.join([sys.executable, str(server),
                          "--task-file", args.task_file])
    # MCP reads the command from stdin and starts the server,
    # then the client connects to it.
    with MCPClient.stdio(command) as client:
        output = MBPPRunner(smith, task, client,
                            args.max_iterations,).agent_cycle()
    MBPPRunner.save_solution(args.output, output)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as error:
        if isinstance(error.code, str):
            print_diagnostic(error.code)
            raise SystemExit(1) from None
        raise
