# Groq: uv run --env-file .env python -m agent_mbpp --task-file cache/mbpp_task_original.json --output cache/mbpp_solution_120b.json --model-name openai/gpt-oss-120b --provider-url https://api.groq.com/openai/v1
# OpenRouter: uv run --env-file .env python -m agent_mbpp --task-file cache/mbpp_task_original.json --output cache/mbpp_solution_north.json --model-name nvidia/nemotron-3-ultra-550b-a55b:free --provider-url https://openrouter.ai/api/v1
# Moulinette Groq: set -a; source .env; set +a; ./moulinette/quickstart.sh mbpp --student-path . --model-name openai/gpt-oss-120b --provider-url https://api.groq.com/openai/v1
# Moulinette OpenRouter: set -a; source .env; set +a; ./moulinette/quickstart.sh mbpp --student-path . --model-name nvidia/nemotron-3-ultra-550b-a55b:free --provider-url https://openrouter.ai/api/v1
"""Example agent for MBPP tasks using Groq or OpenRouter APIs.

Copy and paste this line into your terminal to run the MBPP exam on macOS:
cd "$HOME/Agent-Smith" && DOCKER_HOST="unix://$HOME/.docker/run/docker.sock" PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH" ./exams/exam_mbpp.sh --student-path . --moulinette-path moulinette --env-file .env --model-name nvidia/nemotron-3-ultra-550b-a55b:free --provider-url https://openrouter.ai/api/v1
"""
import argparse
import ast
import importlib.util
import json
import re
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
    "Return only executable Python code with the required function. "
    "No tests, examples, comments, explanations, or Markdown."
)
MODELS = {
    "https://api.groq.com/openai/v1": {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    },
    "https://openrouter.ai/api/v1": {
        "cohere/north-mini-code:free",
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


class MBPPRunner:
    def __init__(self, agent: GroqAPI | OpenRouterAPI,
                 task: MBPPTaskInput, max_iterations: int = MAX_ATTEMPTS):
        self.agent = agent
        self.task = task
        self.max_iterations = max_iterations
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.steps: list[StepMetrics] = []

    "TEMPORARY: later we will use our own sandbox"
    @staticmethod
    def load_docker_helper():
        """Load the sandbox implementation supplied with the local moulinette."""
        helper_path = Path(__file__).resolve().parent / "moulinette/moulinette/mbpp/interact.py"
        spec = importlib.util.spec_from_file_location("mbpp_docker_helper", helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load sandbox helper from {helper_path}")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        return helper.run_code_in_docker

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
            return "", f"No valid Python code found ({error.msg}). Return the complete Python function."

        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
            return "", "No Python function found in the response. Return the complete Python function."
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

    def test_solution(self, solution, run_code_in_docker):
        test_code = "\n".join([*self.task.test_imports, solution, *self.task.test_list])
        deadline = START + TASK_TIME_LIMIT_SECONDS - RESULT_WRITE_MARGIN_SECONDS
        remaining = deadline - perf_counter()

        if remaining <= 0:
            raise TimeoutError("MBPP time budget exhausted before tests.")

        passed, output = run_code_in_docker(test_code, timeout=min(30.0, remaining))

        if perf_counter() >= deadline:
            raise TimeoutError("MBPP time budget exhausted during tests.")

        return passed, output, test_code

    def run_attempt(self, attempt: int, prompt: str, run_code_in_docker) -> AttemptResult:
        answer = solution = test_code = output = parsing_feedback = ""
        input_tokens = output_tokens = 0
        request_metrics = RequestMetrics()
        timestamp = datetime.now().isoformat()
        passed = stop = completed = False

        try:
            print(f"Attempt {attempt}: Sending prompt to agent...")
            response = self.request_solution(prompt, request_metrics)
            answer = response.text
            timestamp = datetime.now().isoformat()
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            solution, parsing_feedback = self.extract_code(answer)
            if not solution:
                output = parsing_feedback
                parsing_feedback = ""
                print_diagnostic(output)
            else:
                if parsing_feedback:
                    print_diagnostic(parsing_feedback, WARNING_COLOR)
                if self.task.test_list:
                    passed, output, test_code = self.test_solution(solution, run_code_in_docker)
                    if not passed:
                        print_diagnostic(f"Public tests failed: {output}")
                else:
                    print("No public tests available; skipping execution.")
                    output = "No public tests available."
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
            sandbox_input=test_code,
            sandbox_output=output,
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
        try:
            run_code_in_docker = self.load_docker_helper()
        except Exception as error:
            raise SystemExit(f"Could not load Docker helper: {error}") from error

        solution = output = ""
        passed = False
        for attempt in range(1, self.max_iterations + 1):
            result = self.run_attempt(attempt, prompt, run_code_in_docker)
            if result.solution:
                solution = result.solution
            passed, output = result.passed, result.output
            if result.completed and not passed and (self.task.test_list or not result.solution):
                if result.parsing_feedback:
                    prompt += f"\n{result.parsing_feedback}"
                prompt += f"\nError: {output}\nFix this."
            if result.stop:
                break
            if passed or (not self.task.test_list and result.solution):
                print("Public tests passed." if passed else "Solution generated without public tests.")
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

    output = MBPPRunner(smith, task, args.max_iterations).agent_cycle()
    MBPPRunner.save_solution(args.output, output)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as error:
        if isinstance(error.code, str):
            print_diagnostic(error.code)
            raise SystemExit(1) from None
        raise
