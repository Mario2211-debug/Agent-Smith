"""Host side of the sandbox: runs agent code in an isolated worker process.

The sandbox wraps the MCP client (not the other way around): MCP tools are
discovered from the connected server and exposed as Python functions inside
the worker. When agent code calls a tool, the worker asks the host, which
performs the MCP call outside the sandbox and sends the result back. The
time spent in MCP calls is not counted against the execution timeout.
"""

import os
import queue
import tempfile
import time
from typing import Any, List, Optional

from src.mcp.schema import ToolSpec
from src.sandbox.bridge import ToolBridge
from src.sandbox.config import SandboxSettings
from src.sandbox.manual import build_manual
from src.sandbox.process import WorkerProcess
from src.sandbox.result import ExecutionResult

RESET_NOTE = ("The worker was restarted, so all variables from previous "
              "steps were lost.")


class Sandbox:
    """Secure, configurable Python executor with optional MCP tools.

    Example:
        with Sandbox(SandboxSettings(), mcp_client) as sandbox:
            result = sandbox.execute("print(read_file('/testbed/a.py'))")
            print(result.to_observation())
    """

    def __init__(self, config: Optional[SandboxSettings] = None,
                 mcp: Any = None):
        """Store the configuration and an optional connected MCP client."""
        self.config = config or SandboxSettings()
        self.mcp = mcp
        self.bridge = ToolBridge(mcp, self.config.max_tool_output_chars)
        self.tools: List[ToolSpec] = self.bridge.tools
        self._worker: Optional[WorkerProcess] = None

    def __enter__(self) -> "Sandbox":
        """Start the worker when entering a ``with`` block."""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the worker when leaving a ``with`` block."""
        self.close()

    def manual(self) -> str:
        """Return the LLM-facing manual generated from the MCP tool schemas."""
        server = getattr(self.mcp, "server_name", None)
        return build_manual(self.tools, self.config, server)

    def start(self) -> None:
        """Spawn a fresh worker interpreter with the current configuration."""
        self.close()
        init = {
            "authorized_imports": self.config.authorized_imports,
            "allowed_directories": self.config.allowed_directories,
            "max_memory_mb": self.config.max_memory_mb,
            "max_output_chars": self.config.max_output_chars,
            "tools": self.bridge.worker_specs(),
        }
        self._worker = WorkerProcess(init, self._working_directory())

    def close(self) -> None:
        """Terminate the worker process if it is running."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _working_directory(self) -> str:
        """Return the first existing allowed directory, else a temp dir."""
        for directory in self.config.allowed_directories:
            if os.path.isdir(directory):
                return os.path.realpath(directory)
        return tempfile.gettempdir()

    # ------------------------------------------------------------ execution
    def execute(self, code: str,
                timeout: Optional[float] = None) -> ExecutionResult:
        """Run ``code`` in the worker and return an :class:`ExecutionResult`.

        ``timeout`` may only lower the configured limit (e.g. when the task
        itself is about to run out of time).

        Raises:
            KeyboardInterrupt / SystemExit: propagated from agent code (or
                a Ctrl-C while waiting) so the agent loop can shut down.
        """
        if self._worker is None or not self._worker.alive:
            self.start()
        self.bridge.notes, output = [], []
        limit = self.config.max_execution_time_seconds
        timeout = limit if timeout is None else max(min(timeout, limit), 0.1)
        started = time.monotonic()
        deadline = started + timeout
        self._worker.send({"op": "exec", "code": code})
        try:
            while True:
                try:
                    message = self._worker.receive(deadline - time.monotonic())
                except queue.Empty:
                    return self._failure(output, started, timeout)
                if message is None:
                    return self._failure(output, started, None)
                if message.get("op") == "out":
                    output.append(message.get("data", ""))
                elif message.get("op") in ("tool", "mcp"):
                    paused = time.monotonic()
                    self._worker.send(self.bridge.answer(message))
                    deadline += time.monotonic() - paused
                elif message.get("op") == "done":
                    return self._done_result(message, output, started)
        except KeyboardInterrupt:
            self.close()
            raise

    # -------------------------------------------------------------- results
    def _build(self, output: List[str], started: float,
               total_chars: Optional[int] = None) -> ExecutionResult:
        """Create a result with output, truncation note and duration."""
        text = "".join(output)
        result = ExecutionResult(output=text, notes=list(self.bridge.notes),
                                 duration_seconds=time.monotonic() - started)
        total = total_chars if total_chars is not None else len(text)
        if total > self.config.max_output_chars:
            result.truncated = True
            result.notes.append(
                f"Printed output was truncated: showing the first "
                f"{self.config.max_output_chars} of {total} characters.")
        return result

    def _done_result(self, message: dict, output: List[str],
                     started: float) -> ExecutionResult:
        """Convert a worker ``done`` message, re-raising control flow."""
        if message.get("control") == "KeyboardInterrupt":
            raise KeyboardInterrupt
        if message.get("control") == "SystemExit":
            raise SystemExit(message.get("exit_code"))
        result = self._build(output, started, message.get("output_chars"))
        result.error = message.get("error")
        result.final_answer = message.get("final_answer")
        return result

    def _failure(self, output: List[str], started: float,
                 timeout: Optional[float]) -> ExecutionResult:
        """Report a timeout (``timeout`` set) or a dead worker (None)."""
        worker, self._worker = self._worker, None
        timed_out = timeout is not None
        if timed_out:
            worker.kill()
            error = f"TimeoutError: execution exceeded {timeout:g} seconds."
            notes = ["Execution hit the timeout: the output above is "
                     "partial.", RESET_NOTE]
        else:
            code, details = worker.exit_code(), worker.stderr_tail()
            worker.kill()
            error = (f"The sandbox process died (exit code {code}), most "
                     f"likely because the memory limit of "
                     f"{self.config.max_memory_mb} MB was exceeded."
                     + (f"\n{details}" if details else ""))
            notes = [RESET_NOTE]
        result = self._build(output, started)
        result.timed_out, result.error = timed_out, error
        result.notes.extend(notes)
        return result
