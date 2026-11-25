"""File system tools: read_file, edit_file and list_files."""

import shlex
from typing import Optional

from src.tools.executor import Executor

MAX_LISTED_FILES = 500
SYNTAX_CHECK = ("python3 -c 'import ast, sys; "
                "ast.parse(sys.stdin.read(), sys.argv[1])' ")


class FileSystemTools:
    """Read, edit and list files through an :class:`Executor`."""

    def __init__(self, executor: Executor):
        """Bind the tools to an executor (local or Docker)."""
        self.executor = executor

    def read_file(self, filepath: str, start_line: int = 1,
                  end_line: Optional[int] = None) -> str:
        """Return lines ``start_line..end_line`` as ``<n>: <content>``."""
        try:
            lines = self.executor.read_text(filepath).splitlines()
        except FileNotFoundError as error:
            return f"Error: cannot read {filepath}: {error}"
        if not lines:
            return f"{filepath} is empty."
        start = max(int(start_line or 1), 1)
        end = min(int(end_line), len(lines)) if end_line else len(lines)
        if start > len(lines):
            return (f"Error: start_line {start} is past the end of "
                    f"{filepath} ({len(lines)} lines).")
        return "\n".join(f"{number}: {lines[number - 1]}"
                         for number in range(start, end + 1))

    def edit_file(self, filepath: str, old_str: str, new_str: str) -> str:
        """Replace the unique occurrence of ``old_str`` by ``new_str``.

        Python files are syntax-checked first; an edit that would introduce
        a syntax error is rejected and the file is left unchanged.
        """
        try:
            content = self.executor.read_text(filepath)
        except FileNotFoundError as error:
            return f"Error: cannot read {filepath}: {error}"
        count = content.count(old_str) if old_str else 0
        if count == 0:
            return (f"Error: old_str not found in {filepath}. Read the file "
                    "again and copy the exact text, including indentation.")
        if count > 1:
            return (f"Error: old_str appears {count} times in {filepath}. "
                    "Include more surrounding lines to make it unique.")
        updated = content.replace(old_str, new_str, 1)
        if filepath.endswith(".py"):
            error = self._syntax_error(filepath, updated)
            if error:
                return (f"Error: edit rejected, it would introduce a syntax "
                        f"error in {filepath}; the file is unchanged.\n"
                        f"{error}")
        try:
            self.executor.write_text(filepath, updated)
        except OSError as error:
            return f"Error: cannot write {filepath}: {error}"
        first = content[:content.index(old_str)].count("\n") + 1
        last = first + max(new_str.count("\n"), 0)
        return (f"Edited {filepath}. New content of lines {first}-{last}:\n"
                + self.read_file(filepath, max(first - 2, 1), last + 2))

    def list_files(self, directory: Optional[str] = None,
                   pattern: str = "*") -> str:
        """List files under ``directory`` whose name matches ``pattern``."""
        directory = directory or self.executor.root
        test = "-path" if "/" in pattern else "-name"
        command = (f"find {shlex.quote(directory)} -type f {test} "
                   f"{shlex.quote(pattern or '*')} -not -path '*/.git/*' "
                   f"| sort | head -n {MAX_LISTED_FILES + 1}")
        result = self.executor.run(command)
        if result.exit_code != 0 and not result.stdout:
            return f"Error: {result.stderr.strip()}"
        files = result.stdout.splitlines()
        if not files:
            return f"No files matching '{pattern}' in {directory}."
        if len(files) > MAX_LISTED_FILES:
            files = files[:MAX_LISTED_FILES]
            files.append(f"... (more than {MAX_LISTED_FILES} files, use a "
                         "narrower directory or pattern)")
        return "\n".join(files)

    def _syntax_error(self, filepath: str, content: str) -> str:
        """Return a syntax error message for ``content``, or ''."""
        # A non-existent name makes Python show the edited line, not the
        # line currently on disk.
        label = shlex.quote(f"{filepath} (after edit)")
        result = self.executor.run(SYNTAX_CHECK + label,
                                   stdin=content, timeout=30)
        if result.exit_code in (0, 127):  # 127: no python3, skip the check
            return ""
        lines = result.stderr.strip().splitlines()
        return "\n".join(lines[-4:]) or "unknown syntax error"
