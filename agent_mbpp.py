# Run: uv run --env-file .env python -m agent_mbpp --task-file cache/mbpp_task_original.json --output cache/mbpp_solution_20b.json --model-name openai/gpt-oss-20b
# Moulinette: set -a; source .env; set +a; ./moulinette/quickstart.sh mbpp --student-path . --model-name openai/gpt-oss-120b --provider-url https://api.groq.com/openai/v1
import argparse
import json
from datetime import datetime
import importlib.util
from pathlib import Path
from time import perf_counter, sleep
from openai import RateLimitError
from pydantic import ValidationError
from structures import MBPPTaskInput, SolutionOutput, StepMetrics
from agent import Agent
import re


# Temporary development helper from the local moulinette checkout.
def load_docker_helper():
    helper_path = Path(__file__).resolve().parent / "moulinette/moulinette/mbpp/interact.py"
    spec = importlib.util.spec_from_file_location("mbpp_docker_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    run_code_in_docker = helper.run_code_in_docker
    return run_code_in_docker


def extract_code(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def load_task(task_file: str) -> MBPPTaskInput:
    try:
        with open(task_file, "r") as f:
            task_data = json.load(f)
        task_data = MBPPTaskInput(**task_data)
    except FileNotFoundError:
        raise SystemExit(f"Task file '{task_file}' not found.")
    except json.JSONDecodeError:
        raise SystemExit(f"Task file '{task_file}' is not a valid JSON file.")
    except ValidationError as e:
        raise SystemExit(f"Task file '{task_file}' does not conform to the expected structure: {e}")
    except Exception as e:
        raise SystemExit(f"An error occurred while loading the task file: {e}")
    return task_data


def save_solution(output_file: str, solution: SolutionOutput):
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(solution.model_dump_json(indent=4))
    except FileNotFoundError:
        raise SystemExit(f"Output directory for '{output_file}' does not exist and could not be created.")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to write solution to '{output_file}': {e}")
    except Exception as e:
        raise SystemExit(f"An error occurred while saving the solution: {e}")


def agent_cycle(agent: Agent, task: MBPPTaskInput) -> SolutionOutput:
    prompt = (
        f"{task.task_definition}\n"
        f"{task.function_definition}\n"
        f"Public tests: {json.dumps(task.test_list)}\n"
        "Do not inlude tests. Respond only what asked with executable Python code, without explanations or Markdown code fences or comments."
    )
    try:
        run_code_in_docker = load_docker_helper()
    except Exception as error:
        raise SystemExit(f"Could not load Docker helper: {error}")
    total_requests = 0
    input_tokens = 0
    output_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    output_tokens = 0
    start = perf_counter()
    steps = []
    for attempt in range(10):
        retries = 0
        try:
            print(f"Attempt {attempt + 1}: Sending prompt to agent...")
            while True:
                try:
                    total_requests += 1
                    start_time = perf_counter()
                    response = agent.run(prompt)
                    answer = response.output_text
                    request_time_ms = perf_counter() - start_time * 1000
                    timestamp = datetime.now().isoformat()
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    break
                except RateLimitError as e:
                    retries += 1
                    print(f"Rate limit error encountered: {e}. Retrying after 30 seconds...")
                    sleep(30)
            cleaned_answer = extract_code(answer)
            if not task.test_list:
                print("No public tests available; skipping execution.")
                return cleaned_answer
            test_code = "\n".join([*task.test_imports, cleaned_answer, *task.test_list])
            passed, output = run_code_in_docker(test_code)
            print(answer)
            prompt += f"Error: {output} fix this"
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        if attempt == 9 and not passed:
            print("Max attempts reached. Exiting.")
            passed = False
            output = str(e)
        steps.append(StepMetrics(
            step=attempt + 1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_time_ms=request_time_ms,
            timestamp=timestamp,
            api_url=str(agent.client.base_url),
            model_name=agent.model_name,
            llm_output=answer,
            sandbox_input=test_code,
            sandbox_output=output,
            retries=retries
        ))
        if passed:
            print("Public tests passed.")
            print(cleaned_answer)
            break


    output = SolutionOutput(
        task_id=str(task.task_id),
        benchmark="mbpp",
        success=passed,
        solution=cleaned_answer,
        iterations=attempt + 1,
        total_requests=total_requests,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_time_seconds=perf_counter() - start,
        steps=steps,
        system_prompt=prompt,
        error=None if passed else output,
    )
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True, help="Path to the MBPP task JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    parser.add_argument("--model-name", default="openai/GPT-OSS-120B", help="Model name")
    parser.add_argument("--provider-url", default="https://api.groq.com/openai/v1", help="Provider API URL")
    args = parser.parse_args()
    task = load_task(args.task_file)
    smith = Agent(args.model_name, args.provider_url)

    start = perf_counter()

    output = agent_cycle(smith, task)
    elapsed = perf_counter() - start
    save_solution(args.output, output)


if __name__ == "__main__":
    main()
