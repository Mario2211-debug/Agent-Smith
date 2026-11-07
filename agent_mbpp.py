# Run: uv run --env-file .env python -m agent_mbpp --task-file cache/mbpp_task.json --output cache/mbpp_solution.json
# Moulinette: set -a; source .env; set +a; ./moulinette/quickstart.sh mbpp --student-path . --model-name openai/gpt-oss-120b --provider-url https://api.groq.com/openai/v1
import argparse
import json
import importlib.util
from pathlib import Path
from time import perf_counter
from pydantic import ValidationError
from structures import MBPPTaskInput
from agent import Agent

# Temporary development helper from the local moulinette checkout.
def load_docker_helper():
    helper_path = Path(__file__).resolve().parent / "moulinette/moulinette/mbpp/interact.py"
    spec = importlib.util.spec_from_file_location("mbpp_docker_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    run_code_in_docker = helper.run_code_in_docker
    return run_code_in_docker


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


def save_solution(output_file: str, solution: str):
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"solution": solution}, f, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to write solution to '{output_file}': {e}")
    except Exception as e:
        raise SystemExit(f"An error occurred while saving the solution: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True, help="Path to the MBPP task JSON file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    parser.add_argument("--model-name", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--provider-url", default="https://api.groq.com/openai/v1", help="Provider API URL")
    args = parser.parse_args()
    task = load_task(args.task_file)
    smith = Agent(args.model_name, args.provider_url)

    start = perf_counter()

    prompt = (
        f"{task.task_definition}\n"
        f"{task.function_definition}\n"
        f"Public tests: {json.dumps(task.test_list)}\n"
        "Respond only what asked with executable Python code, without explanations or Markdown code fences or comments."
    )
    #  print(f"Prompt sent to LLM:\n{prompt}\n")
    answer = smith.run(prompt)
    elapsed = perf_counter() - start
    save_solution(args.output, answer)
    test_code = "\n".join([*task.test_imports, answer, *task.test_list])
    if not task.test_list:
        print("No public tests available; skipping execution.")
        return
    try:
        run_code_in_docker = load_docker_helper()
        passed, output = run_code_in_docker(test_code)
    except Exception as error:
        print(f"Could not run tests in Docker: {error}")
        return
    print("Public tests passed." if passed else "Public tests failed or could not run.")
    if output:
        print(output)


if __name__ == "__main__":
    main()
