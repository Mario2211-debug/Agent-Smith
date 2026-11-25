"""MCP server with the MBPP tools.

Usage:
    python mcp_tools_mbpp.py [--task-file task.json]          # stdio
    python mcp_tools_mbpp.py --task-file task.json --http     # HTTP :8000/mcp

The task file can also be given with the ``MBPP_TASK_FILE`` variable.
Exposes the ``run_tests`` tool, the ``mbpp://task`` resource and the
``solve_mbpp`` prompt.
"""

import json
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from src.mcp.serve import base_parser, load_task, serve
from src.tools.mbpp import run_mbpp_tests

server = FastMCP("agent-smith-mbpp", log_level="WARNING")
TASK: dict = {}


@server.tool()
def run_tests(code: str, tests: Optional[List[str]] = None) -> str:
    """Run Python solution code against the task's public tests.

    Pass the complete solution (imports and function definitions) as
    `code`. Optionally pass your own assert statements as `tests`; by
    default the task's public tests are used. Reports each assertion as
    PASS/FAIL and, for `assert f(x) == y`, the value f(x) returned.
    """
    return run_mbpp_tests(code, tests or TASK.get("test_list", []),
                          TASK.get("test_imports", []))


@server.resource("mbpp://task", mime_type="application/json")
def task_resource() -> str:
    """The current MBPP task: description, signature and public tests."""
    return json.dumps(TASK, indent=2) if TASK else "No task loaded."


@server.prompt()
def solve_mbpp() -> str:
    """Step-by-step methodology for solving the current MBPP task."""
    return ("1. Read the task and the public tests carefully.\n"
            "2. Write the function, then call run_tests(code) and print it.\n"
            "3. Fix any failing test using the reported values.\n"
            "4. When all tests pass, call final_answer(code).\n\n"
            f"Task:\n{json.dumps(TASK, indent=2) if TASK else 'not loaded'}")


def main() -> None:
    """Parse the command line, load the task and start the server."""
    parser = base_parser("MBPP MCP tools server")
    parser.add_argument("--task-file", help="MBPP task JSON file")
    args = parser.parse_args()
    TASK.update(load_task(args.task_file, "MBPP_TASK_FILE"))
    serve(server, args)


if __name__ == "__main__":
    main()
