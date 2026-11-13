# Run: uv run --env-file .env python -m agent_mbpp --task-file cache/mbpp_task_original.json --output cache/mbpp_solution_120b.json --model-name openai/gpt-oss-120b
# Moulinette: set -a; source .env; set +a; ./moulinette/quickstart.sh mbpp --student-path . --model-name openai/gpt-oss-120b --provider-url https://api.groq.com/openai/v1
import argparse
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

from openai import RateLimitError
from pydantic import ValidationError

from groq import GroqAPI
from structures import AttemptResult, MBPPTaskInput, SolutionOutput, StepMetrics

MAX_ATTEMPTS = 10
MAX_RATE_LIMIT_RETRIES = 2
RATE_LIMIT_DELAY_SECONDS = 30


class MBPPRunner:
    def __init__(self, agent: GroqAPI, task: MBPPTaskInput):
        self.agent = agent
        self.task = task
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.steps: list[StepMetrics] = []

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
    def extract_code(text: str) -> str:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

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
            path.write_text(solution.model_dump_json(indent=4), encoding="utf-8")
        except OSError as error:
            raise SystemExit(f"Could not write solution to '{output_file}': {error}") from error

    def build_prompt(self) -> str:
        return (
            f"{self.task.task_definition}\n"
            f"{self.task.function_definition}\n"
            f"Public tests: {json.dumps(self.task.test_list)}\n"
            "Implement the requested function correctly, including edge cases. "
            "Prefer a simple, efficient solution. "
            "Return only executable Python code with the required function. "
            "No tests, examples, comments, explanations, or Markdown."
        )

    def request_solution(self, prompt: str):
        retries = 0
        while True:
            try:
                self.total_requests += 1
                start_time = perf_counter()
                response = self.agent.run(prompt)
                request_time_ms = (perf_counter() - start_time) * 1000
                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens
                return response, request_time_ms, retries
            except RateLimitError as e:
                if retries >= MAX_RATE_LIMIT_RETRIES:
                    raise
                retries += 1
                print(f"Rate limit error: {e}. Retrying in {RATE_LIMIT_DELAY_SECONDS} seconds...")
                sleep(RATE_LIMIT_DELAY_SECONDS)

    def test_solution(self, solution, run_code_in_docker):
        test_code = "\n".join([*self.task.test_imports, solution, *self.task.test_list])
        passed, output = run_code_in_docker(test_code)
        return passed, output, test_code

    def run_attempt(self, attempt: int, prompt: str, run_code_in_docker) -> AttemptResult:
        answer = solution = test_code = output = ""
        input_tokens = output_tokens = retries = 0
        request_time_ms = 0.0
        timestamp = datetime.now().isoformat()
        passed = stop = completed = False

        try:
            print(f"Attempt {attempt}: Sending prompt to agent...")
            response, request_time_ms, retries = self.request_solution(prompt)
            answer = response.text
            timestamp = datetime.now().isoformat()
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            solution = self.extract_code(answer)
            if not solution:
                output = "The model returned no code. Please generate the complete Python function."
                print(output)
            elif self.task.test_list:
                passed, output, test_code = self.test_solution(solution, run_code_in_docker)
            else:
                print("No public tests available; skipping execution.")
                output = "No public tests available."
            completed = True
        except Exception as error:
            print(f"Attempt {attempt} failed: {error}")
            output = str(error)
            stop = isinstance(error, RateLimitError)
            if stop:
                retries = MAX_RATE_LIMIT_RETRIES

        self.steps.append(StepMetrics(
            step=attempt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_time_ms=request_time_ms,
            timestamp=timestamp,
            api_url=str(self.agent.client.base_url),
            model_name=self.agent.model_name,
            llm_output=answer,
            sandbox_input=test_code,
            sandbox_output=output,
            retries=retries,
        ))
        return AttemptResult(passed, answer, solution, output, stop, completed)

    def build_result(
        self, attempt: int, passed: bool, solution: str, output: str,
        prompt: str, start: float,
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
            total_time_seconds=perf_counter() - start,
            steps=self.steps,
            system_prompt=prompt,
            error=None if passed else output,
        )

    def agent_cycle(self) -> SolutionOutput:
        prompt = self.build_prompt()
        try:
            run_code_in_docker = self.load_docker_helper()
        except Exception as error:
            raise SystemExit(f"Could not load Docker helper: {error}") from error

        start = perf_counter()
        solution = output = ""
        passed = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result = self.run_attempt(attempt, prompt, run_code_in_docker)
            if result.solution:
                solution = result.solution
            passed, output = result.passed, result.output
            if result.completed and not passed and (self.task.test_list or not result.solution):
                prompt += f"Error: {output} fix this"
            if result.stop:
                break
            if passed or (not self.task.test_list and result.solution):
                print("Public tests passed." if passed else "Solution generated without public tests.")
                print(solution)
                break
            if attempt == MAX_ATTEMPTS:
                print("Max attempts reached. Exiting.")

        return self.build_result(attempt, passed, solution, output, prompt, start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True, help="Path to the MBPP task JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    parser.add_argument("--model-name", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--provider-url", default="https://api.groq.com/openai/v1", help="Provider API URL")
    args = parser.parse_args()
    task = MBPPRunner.load_task(args.task_file)
    smith = GroqAPI(args.model_name, args.provider_url)

    output = MBPPRunner(smith, task).agent_cycle()
    MBPPRunner.save_solution(args.output, output)


if __name__ == "__main__":
    main()
