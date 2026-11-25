"""Command executors: run shell commands locally or inside a Docker container.

Every tool is written against :class:`Executor`, so the same code works on
the host (tests, local repositories) and inside a SWE-bench container.
"""

import shlex
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_TIMEOUT_SECONDS = 120


@dataclass
class CommandResult:
    """Outcome of a shell command."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    def format(self) -> str:
        """Render stdout, stderr and exit code for an LLM."""
        status = "timed out" if self.timed_out else str(self.exit_code)
        return (f"exit code: {status}\n--- stdout ---\n{self.stdout.rstrip()}"
                f"\n--- stderr ---\n{self.stderr.rstrip()}")


class Executor(ABC):
    """Runs ``bash -c`` commands relative to a repository root."""

    def __init__(self, root: str):
        """Store the repository root used as default working directory."""
        self.root = root

    @abstractmethod
    def argv(self, command: str, workdir: str) -> List[str]:
        """Return the argv that runs ``command`` in ``workdir``."""

    def run(self, command: str, workdir: Optional[str] = None,
            timeout: float = DEFAULT_TIMEOUT_SECONDS,
            stdin: Optional[str] = None) -> CommandResult:
        """Run a command and capture its output; never raises on failure."""
        try:
            done = subprocess.run(
                self.argv(command, workdir or self.root), input=stdin,
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace")
            return CommandResult(done.stdout, done.stderr, done.returncode)
        except subprocess.TimeoutExpired as error:
            return CommandResult(_text(error.stdout), _text(error.stderr),
                                 -1, timed_out=True)
        except OSError as error:
            return CommandResult("", str(error), -1)

    def read_text(self, path: str) -> str:
        """Return a file's content.

        Raises:
            FileNotFoundError: if the file cannot be read.
        """
        result = self.run(f"cat -- {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr.strip() or path)
        return result.stdout

    def write_text(self, path: str, content: str) -> None:
        """Overwrite a file with ``content``.

        Raises:
            OSError: if the file cannot be written.
        """
        result = self.run(f"cat > {shlex.quote(path)}", stdin=content)
        if result.exit_code != 0:
            raise OSError(result.stderr.strip() or f"cannot write {path}")


class LocalExecutor(Executor):
    """Runs commands on the host machine."""

    def argv(self, command: str, workdir: str) -> List[str]:
        """Run through bash in ``workdir``."""
        return ["bash", "-c", f"cd {shlex.quote(workdir)} && {command}"]


class DockerExecutor(Executor):
    """Runs commands inside a Docker container via ``docker exec``."""

    def __init__(self, container: str, root: str = "/testbed",
                 owned: bool = False):
        """Attach to ``container``; ``owned`` containers are removed on stop."""
        super().__init__(root)
        self.container = container
        self.owned = owned

    @classmethod
    def from_image(cls, image: str, root: str = "/testbed") -> "DockerExecutor":
        """Start a detached container from ``image`` that we own.

        Raises:
            RuntimeError: if Docker cannot start the container.
        """
        name = f"agent-smith-{uuid.uuid4().hex[:12]}"
        done = subprocess.run(
            ["docker", "run", "-d", "--name", name, "-w", root, image,
             "sleep", "infinity"], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"docker run failed: {done.stderr.strip()}")
        return cls(name, root, owned=True)

    def argv(self, command: str, workdir: str) -> List[str]:
        """Run through bash inside the container."""
        return ["docker", "exec", "-i", "-w", workdir, self.container,
                "bash", "-c", command]

    def stop(self) -> None:
        """Remove the container if this executor started it."""
        if self.owned:
            subprocess.run(["docker", "rm", "-f", self.container],
                           capture_output=True)
            self.owned = False


def _text(data: object) -> str:
    """Decode partial output captured by a TimeoutExpired."""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return data or ""
