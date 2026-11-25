"""MCP server with the mandatory SWE-bench tools.

Usage:
    python mcp_tools_swebench.py [options]           # stdio
    python mcp_tools_swebench.py --http [options]    # HTTP :8000/mcp

Options (or environment variables):
    --task-file  / SWEBENCH_TASK_FILE  task JSON (eval_script, docker_image)
    --container  / SWEBENCH_CONTAINER  existing container to work in
    --root       / SWEBENCH_ROOT       repository root (default /testbed)
    --local                            work on the host instead of Docker

With a task but no container, a container is started from the task's
``docker_image`` and removed when the server exits. Without a task or a
container the tools run on the host, in ``--root`` (or the current dir).
"""

import atexit
import os
import signal
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.mcp.serve import base_parser, load_task, serve
from src.tools.execution import ExecutionTools
from src.tools.executor import DockerExecutor, Executor, LocalExecutor
from src.tools.filesystem import FileSystemTools
from src.tools.search import SearchTools

server = FastMCP("agent-smith-swebench", log_level="WARNING")
FILES: FileSystemTools
SEARCH: SearchTools
EXEC: ExecutionTools


@server.tool()
def read_file(filepath: str, start_line: int = 1,
              end_line: Optional[int] = None) -> str:
    """Read a file with line numbers, formatted as '<line_number>: <line>'.
    Use start_line/end_line (1-based, inclusive) to read only a range."""
    return FILES.read_file(filepath, start_line, end_line)


@server.tool()
def edit_file(filepath: str, old_str: str, new_str: str) -> str:
    """Replace the exact, unique text old_str with new_str in a file.
    Python edits that would introduce a syntax error are rejected."""
    return FILES.edit_file(filepath, old_str, new_str)


@server.tool()
def list_files(directory: Optional[str] = None, pattern: str = "*") -> str:
    """List files under a directory (default: repository root) whose name
    matches a glob pattern such as '*.py'."""
    return FILES.list_files(directory, pattern)


@server.tool()
def search_code(pattern: str, file_pattern: str = "*") -> str:
    """grep -E search in the repository. Output lines look like
    '/abs/path.py:<line_number> <line_content>'. file_pattern filters files
    by glob, e.g. '*.py'."""
    return SEARCH.search_code(pattern, file_pattern)


@server.tool()
def search_function_or_class_definition_in_code(name: str) -> str:
    """Find where a function or class is defined ('def name' / 'class
    name'). Output uses the search_code format."""
    return SEARCH.search_function_or_class_definition_in_code(name)


@server.tool()
def find_references(name: Optional[str] = None, filepath: Optional[str] = None,
                    line: Optional[int] = None) -> str:
    """Find all usages of a symbol in Python files (whole-word match). Give
    the name, or a filepath and line containing it."""
    return SEARCH.find_references(name, filepath, line)


@server.tool()
def run_tests() -> str:
    """Run the task's evaluation script and return the end of its output."""
    return EXEC.run_tests()


@server.tool()
def get_patch() -> str:
    """Return the unified git diff of all changes made to the repository."""
    return EXEC.get_patch()


@server.tool()
def run_command(command: str, workdir: Optional[str] = None) -> str:
    """Run a shell command in workdir (default: repository root). Returns
    its stdout, stderr and exit code."""
    return EXEC.run_command(command, workdir)


def build_executor(args, task: dict) -> Executor:
    """Choose a local or Docker executor from the CLI options and task."""
    container = args.container or os.environ.get("SWEBENCH_CONTAINER")
    root = args.root or os.environ.get("SWEBENCH_ROOT")
    if args.local or not (container or task.get("docker_image")):
        return LocalExecutor(os.path.abspath(root or os.getcwd()))
    if container:
        return DockerExecutor(container, root or "/testbed")
    executor = DockerExecutor.from_image(task["docker_image"],
                                         root or "/testbed")
    atexit.register(executor.stop)
    # Turn SIGTERM into a normal exit so the atexit cleanup still runs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    return executor


def main() -> None:
    """Parse options, create the tools and start the server."""
    global FILES, SEARCH, EXEC
    parser = base_parser("SWE-bench MCP tools server")
    parser.add_argument("--task-file", help="SWE-bench task JSON file")
    parser.add_argument("--container", help="existing Docker container")
    parser.add_argument("--root", help="repository root")
    parser.add_argument("--local", action="store_true",
                        help="run tools on the host instead of Docker")
    args = parser.parse_args()
    task = load_task(args.task_file, "SWEBENCH_TASK_FILE")
    executor = build_executor(args, task)
    FILES, SEARCH = FileSystemTools(executor), SearchTools(executor)
    EXEC = ExecutionTools(executor, task.get("eval_script", ""))
    serve(server, args)


if __name__ == "__main__":
    main()
