"""Execution tools: run_tests, get_patch and run_command."""

import shlex
from typing import Optional

from src.tools.executor import Executor

TEST_TIMEOUT_SECONDS = 600
COMMAND_TIMEOUT_SECONDS = 120
MAX_TEST_OUTPUT_CHARS = 15_000
EVAL_SCRIPT_PATH = "/tmp/agent_smith_eval.sh"


class ExecutionTools:
    """Run the evaluation script, shell commands and collect the patch."""

    def __init__(self, executor: Executor, eval_script: str = ""):
        """Bind to an executor and the task's evaluation script (bash)."""
        self.executor = executor
        self.eval_script = eval_script

    def run_tests(self) -> str:
        """Run the evaluation script and return the end of its output.

        The tail is kept because test summaries are printed last.
        """
        if not self.eval_script.strip():
            return "Error: no evaluation script is configured for this task."
        written = self.executor.run(f"cat > {EVAL_SCRIPT_PATH}",
                                    stdin=self.eval_script)
        if written.exit_code != 0:
            return f"Error: cannot write eval script: {written.stderr}"
        result = self.executor.run(f"bash {EVAL_SCRIPT_PATH} 2>&1",
                                   timeout=TEST_TIMEOUT_SECONDS)
        output = result.stdout
        if len(output) > MAX_TEST_OUTPUT_CHARS:
            omitted = len(output) - MAX_TEST_OUTPUT_CHARS
            output = (f"... [first {omitted} chars omitted, showing the end]"
                      f"\n{output[-MAX_TEST_OUTPUT_CHARS:]}")
        status = ("timed out" if result.timed_out
                  else f"exit code {result.exit_code}")
        return f"Tests finished ({status}).\n{output}"

    def get_patch(self) -> str:
        """Return ``git -c core.fileMode=false diff`` of the repository."""
        result = self.executor.run("git -c core.fileMode=false diff")
        if result.exit_code != 0:
            return f"Error: git diff failed: {result.stderr.strip()}"
        return result.stdout

    def run_command(self, command: str, workdir: Optional[str] = None) -> str:
        """Run a shell command and return stdout, stderr and exit code."""
        workdir = workdir or self.executor.root
        check = self.executor.run(f"test -d {shlex.quote(workdir)}")
        if check.exit_code != 0:
            return f"Error: working directory {workdir} does not exist."
        return self.executor.run(command, workdir,
                                 timeout=COMMAND_TIMEOUT_SECONDS).format()
