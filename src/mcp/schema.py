"""Transport-independent descriptions of MCP tools and their results.

These plain dataclasses decouple the sandbox and the manual generator from
the MCP SDK: anything that produces ``ToolSpec`` objects can drive them.
"""

import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_JSON_TO_PYTHON = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}


@dataclass
class ParamSpec:
    """One parameter of an MCP tool, derived from its JSON schema."""

    name: str
    type_hint: str = "Any"
    required: bool = False
    description: str = ""
    default: Any = None


@dataclass
class ToolSpec:
    """An MCP tool as seen from Python.

    ``name`` is the original MCP name; ``python_name`` is a valid Python
    identifier used inside the sandbox namespace.
    """

    name: str
    description: str = ""
    params: List[ParamSpec] = field(default_factory=list)

    @property
    def python_name(self) -> str:
        """Return the tool name sanitised into a Python identifier."""
        cleaned = re.sub(r"\W", "_", self.name)
        if not cleaned or cleaned[0].isdigit() or keyword.iskeyword(cleaned):
            cleaned = f"tool_{cleaned}"
        return cleaned

    @property
    def signature(self) -> str:
        """Return a Python-style signature, e.g. ``read_file(filepath: str)``."""
        parts = []
        for param in self.params:
            text = f"{param.name}: {param.type_hint}"
            if not param.required:
                text += f" = {param.default!r}"
            parts.append(text)
        return f"{self.python_name}({', '.join(parts)})"

    @classmethod
    def from_schema(cls, name: str, description: Optional[str],
                    schema: Optional[Dict[str, Any]]) -> "ToolSpec":
        """Build a spec from an MCP tool name, description and inputSchema."""
        schema = schema or {}
        required = set(schema.get("required", []))
        params = []
        for pname, pschema in (schema.get("properties") or {}).items():
            params.append(ParamSpec(
                name=pname,
                type_hint=json_type_to_hint(pschema),
                required=pname in required,
                description=pschema.get("description", ""),
                default=pschema.get("default"),
            ))
        # Required parameters first so positional calls stay intuitive.
        params.sort(key=lambda p: not p.required)
        return cls(name=name, description=(description or "").strip(),
                   params=params)


@dataclass
class ToolResult:
    """Text result of an MCP tool call."""

    text: str
    is_error: bool = False


def json_type_to_hint(schema: Dict[str, Any]) -> str:
    """Convert a JSON-schema fragment into a readable Python type hint."""
    if "anyOf" in schema:
        hints = [json_type_to_hint(option) for option in schema["anyOf"]]
        return " | ".join(dict.fromkeys(hints))
    json_type = schema.get("type")
    if isinstance(json_type, list):
        return " | ".join(_JSON_TO_PYTHON.get(t, "Any") for t in json_type)
    if json_type == "array" and isinstance(schema.get("items"), dict):
        return f"list[{json_type_to_hint(schema['items'])}]"
    return _JSON_TO_PYTHON.get(json_type, "Any")


def content_to_text(blocks: List[Any]) -> str:
    """Join MCP content blocks, describing non-text blocks briefly."""
    parts = []
    for block in blocks:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
        elif getattr(block, "type", "") == "resource":
            parts.append(getattr(block.resource, "text", "[binary resource]"))
        else:
            parts.append(f"[{getattr(block, 'type', 'unknown')} content "
                         "omitted]")
    return "\n".join(parts)
