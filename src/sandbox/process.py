"""Lifecycle of the sandbox worker process (spawn, messaging, kill)."""

import os
import queue
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, Optional

from src.sandbox import protocol

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "worker.py")
STARTUP_TIMEOUT_SECONDS = 15.0


class SandboxError(RuntimeError):
    """Raised when the worker process cannot be started."""


class WorkerProcess:
    """A ``python -I worker.py`` subprocess speaking the JSON-lines protocol.

    The worker runs in its own session (so a Ctrl-C in the terminal reaches
    only the host), with a minimal environment (no API keys leak into it)
    and without site-packages.
    """

    def __init__(self, init: Dict[str, Any], cwd: str):
        """Spawn the worker, send ``init`` and wait for ``ready``.

        Raises:
            SandboxError: if the worker does not report ready in time.
        """
        self._messages: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._stderr = tempfile.TemporaryFile(mode="w+")
        self._process = subprocess.Popen(
            [sys.executable, "-I", WORKER_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, text=True, encoding="utf-8",
            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
            cwd=cwd, start_new_session=True)
        threading.Thread(target=self._read_messages, daemon=True).start()
        self.send(dict(init, op="init", cwd=cwd))
        ready = self.receive(STARTUP_TIMEOUT_SECONDS)
        if not ready or ready.get("op") != "ready":
            self.kill()
            raise SandboxError(f"Sandbox worker failed to start: "
                               f"{self.stderr_tail()}")

    @property
    def alive(self) -> bool:
        """True while the worker process is running."""
        return self._process.poll() is None

    def send(self, message: Dict[str, Any]) -> None:
        """Send one message to the worker."""
        protocol.send(self._process.stdin, message)

    def receive(self, timeout: float) -> Optional[dict]:
        """Return the next message, ``None`` on EOF.

        Raises:
            queue.Empty: if nothing arrives within ``timeout`` seconds.
        """
        return self._messages.get(timeout=max(timeout, 0))

    def _read_messages(self) -> None:
        """Reader thread: push decoded worker messages; None means EOF."""
        with self._process.stdout:  # closed here, once EOF is reached
            for line in self._process.stdout:
                self._messages.put(protocol.decode(line))
        self._messages.put(None)

    def exit_code(self) -> Optional[int]:
        """Wait briefly for the process and return its exit code."""
        try:
            return self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return None

    def stderr_tail(self, limit: int = 2000) -> str:
        """Return the end of the worker's stderr (for crash reports)."""
        try:
            self._stderr.seek(0)
            return self._stderr.read()[-limit:].strip()
        except (OSError, ValueError):
            return ""

    def close(self) -> None:
        """Ask the worker to stop, then make sure it is gone."""
        try:
            if self.alive:
                self.send({"op": "shutdown"})
                self._process.wait(timeout=1)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        self.kill()

    def kill(self) -> None:
        """Forcefully stop the worker (used on timeout and interrupts)."""
        if self.alive:
            self._process.kill()
        self._process.wait()
        for stream in (self._process.stdin, self._stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
