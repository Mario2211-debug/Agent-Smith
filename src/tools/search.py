"""Code search tools built on grep.

Output format (one match per line)::

    /absolute/path_to_file.py:<line_number> <line_content>
"""

import re
import shlex
from typing import Optional

from src.tools.executor import Executor

MAX_MATCHES = 200
_GREP_LINE = re.compile(r"^(.*?):(\d+):(.*)$")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SearchTools:
    """grep-like searches over the repository behind an executor."""

    def __init__(self, executor: Executor):
        """Bind the tools to an executor (local or Docker)."""
        self.executor = executor

    def search_code(self, pattern: str, file_pattern: str = "*") -> str:
        """Search ``pattern`` (extended regex) in files matching the glob."""
        return self._grep(["-E", "-e", pattern],
                          file_pattern or "*", f"pattern '{pattern}'")

    def search_function_or_class_definition_in_code(self, name: str) -> str:
        """Find ``def name`` / ``class name`` definitions in Python files."""
        name = name.split(".")[-1].strip()
        if not _IDENTIFIER.fullmatch(name):
            return f"Error: '{name}' is not a valid Python identifier."
        regex = (rf"^[[:space:]]*(async[[:space:]]+)?(def|class)[[:space:]]+"
                 rf"{name}([^[:alnum:]_]|$)")
        return self._grep(["-E", "-e", regex], "*.py",
                          f"definition of '{name}'")

    def find_references(self, name: Optional[str] = None,
                        filepath: Optional[str] = None,
                        line: Optional[int] = None) -> str:
        """Find every whole-word usage of a symbol in Python files.

        If ``name`` is omitted, the first identifier on ``filepath:line``
        is used as the symbol.
        """
        if not name and filepath and line:
            name = self._identifier_at(filepath, int(line))
        name = (name or "").split(".")[-1].strip()
        if not _IDENTIFIER.fullmatch(name):
            return "Error: give a valid symbol name (or filepath and line)."
        return self._grep(["-w", "-F", "-e", name], "*.py",
                          f"references to '{name}'")

    def _identifier_at(self, filepath: str, line: int) -> str:
        """Return the first non-keyword identifier on a given line."""
        try:
            lines = self.executor.read_text(filepath).splitlines()
            words = _IDENTIFIER.findall(lines[line - 1])
        except (FileNotFoundError, IndexError):
            return ""
        skip = {"def", "class", "async", "return", "self", "import", "from"}
        return next((word for word in words if word not in skip), "")

    def _grep(self, args: list, file_pattern: str, what: str) -> str:
        """Run grep recursively and format matches as ``path:line content``."""
        command = " ".join(
            ["grep", "-rnI", "--exclude-dir=.git",
             f"--include={shlex.quote(file_pattern)}",
             *[shlex.quote(arg) for arg in args],
             shlex.quote(self.executor.root),
             f"| head -n {MAX_MATCHES + 1}"])
        result = self.executor.run(command)
        if not result.stdout:
            if "grep:" in result.stderr:
                return f"Error: {result.stderr.strip()}"
            return f"No matches for {what}."
        matches = []
        for raw in result.stdout.splitlines()[:MAX_MATCHES]:
            found = _GREP_LINE.match(raw)
            if found:
                path, number, text = found.groups()
                matches.append(f"{path}:{number} {text.strip()}")
        if len(result.stdout.splitlines()) > MAX_MATCHES:
            matches.append(f"... (more than {MAX_MATCHES} matches, refine "
                           "the pattern or file_pattern)")
        return "\n".join(matches)
