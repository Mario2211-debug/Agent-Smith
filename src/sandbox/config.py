"""Sandbox configuration: Pydantic model plus JSON file loading."""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import Field, ValidationError

from structures import SandboxConfig

DEFAULT_IMPORTS = [
    "math", "math.*",
    "collections", "collections.*",
    "itertools", "re", "json",
    "typing", "typing.*",
    "functools", "operator",
    "heapq", "bisect", "copy",
    "string", "random",
    "datetime", "datetime.*",
    "array", "cmath",
]
DEFAULT_DIRECTORIES = ["/testbed", "/tmp/agent"]


class SandboxSettings(SandboxConfig):
    """Moulinette ``SandboxConfig`` with project defaults and output limits.

    Allowlist approach: only ``authorized_imports`` may be imported and only
    ``allowed_directories`` may be touched; everything else is blocked.
    """

    authorized_imports: List[str] = Field(
        default_factory=lambda: list(DEFAULT_IMPORTS),
        description="Allowed import names; glob patterns like 'math.*' work.")
    allowed_directories: List[str] = Field(
        default_factory=lambda: list(DEFAULT_DIRECTORIES),
        description="Filesystem paths the sandboxed code may access.")
    max_output_chars: int = Field(
        default=10_000, gt=0,
        description="Maximum printed characters returned per execution.")
    max_tool_output_chars: int = Field(
        default=20_000, gt=0,
        description="Maximum characters returned by a single MCP tool call.")

    @classmethod
    def load(cls, path: Optional[str]) -> "SandboxSettings":
        """Load settings from a JSON file, or return defaults if ``path`` is None.

        Raises:
            SystemExit: with a readable message if the file is missing,
                is not JSON, or does not match the schema.
        """
        if path is None:
            return cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except FileNotFoundError:
            raise SystemExit(f"Sandbox config '{path}' not found.") from None
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"Sandbox config '{path}' is not valid JSON: {error}") from None
        except ValidationError as error:
            raise SystemExit(
                f"Sandbox config '{path}' is invalid:\n{error}") from None
        except OSError as error:
            raise SystemExit(
                f"Could not read sandbox config '{path}': {error}") from None
