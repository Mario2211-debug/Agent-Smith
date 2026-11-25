"""Objects injected into, or used around, agent code inside the worker.

Standard library only: this module is loaded by the isolated worker.
"""

import traceback

import protocol

FILENAME = "<agent>"
CHUNK_SIZE = 4096


class FinalAnswer(BaseException):
    """Raised by ``final_answer()``.

    It derives from BaseException so that ``except Exception`` in agent
    code cannot swallow it.
    """

    def __init__(self, value: str):
        """Keep the submitted answer."""
        super().__init__(value)
        self.value = value


class ToolError(RuntimeError):
    """Raised inside agent code when an MCP tool call fails."""


def final_answer(answer) -> None:
    """Submit the final answer and stop the current execution."""
    raise FinalAnswer(answer if isinstance(answer, str) else repr(answer))


class OutputCapture:
    """File-like object that streams printed text to the host.

    Text is forwarded line by line so a timeout still yields partial output.
    Anything beyond ``limit`` characters is counted but not sent.
    """

    def __init__(self, channel, limit: int):
        """Bind the capture to the host channel with a character budget."""
        self.channel, self.limit = channel, limit
        self.buffer, self.sent, self.total = [], 0, 0

    def write(self, text) -> int:
        """Buffer ``text`` and forward it on newlines or large chunks."""
        text = str(text)
        self.total += len(text)
        if self.sent < self.limit:
            chunk = text[:self.limit - self.sent]
            self.sent += len(chunk)
            self.buffer.append(chunk)
            if "\n" in chunk or sum(map(len, self.buffer)) > CHUNK_SIZE:
                self.flush()
        return len(text)

    def flush(self) -> None:
        """Send buffered text to the host."""
        if self.buffer:
            protocol.send(self.channel, {"op": "out",
                                         "data": "".join(self.buffer)})
            self.buffer.clear()

    def reset(self) -> None:
        """Start a new execution with a fresh budget."""
        self.flush()
        self.sent = self.total = 0

    @staticmethod
    def isatty() -> bool:
        """Captured output is never a terminal."""
        return False


def format_error(error: BaseException) -> str:
    """Format a traceback keeping only frames from the agent's code."""
    frames = [f for f in traceback.extract_tb(error.__traceback__)
              if f.filename == FILENAME]
    lines = ["Traceback (most recent call last):"] if frames else []
    for frame in frames:
        lines.append(f"  line {frame.lineno}, in {frame.name}\n"
                     f"    {frame.line}")
    lines.extend(traceback.format_exception_only(type(error), error))
    return "\n".join(line.rstrip("\n") for line in lines)
