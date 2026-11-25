"""Extract executable Python code from an LLM response.

Supported formats, tried in this order:

(a) Python code blocks  ```python ... ```  (optionally ended by <end_code>)
(b) XML tool calls      <invoke name="t"><parameter name="p">v</parameter></invoke>
(c) JSON/Hermes calls   <tool_call>{"name": "t", "arguments": {...}}</tool_call>
(d) ReAct               Action: t / Action Input: {...}

Non-Python formats are converted into equivalent Python calls, e.g.
``result = read_file(filepath='/testbed/a.py')``, so the sandbox stays
format-agnostic. Every recovery is explained in ``Extraction.note``.
"""

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional

END_CODE = "<end_code>"
_FENCE = re.compile(
    r"```[ \t]*(python|py|python3)?[ \t]*\n(.*?)(?:```|<end_code>)",
    re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*)$",
                         re.DOTALL | re.IGNORECASE)
_INVOKE = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL)
_PARAM = re.compile(r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>',
                    re.DOTALL)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)",
                        re.DOTALL)
_REACT = re.compile(r"Action:\s*([\w.]+)\s*\n\s*Action Input:\s*(.*?)"
                    r"(?=\n\s*(?:Observation|Thought|Action):|\Z)", re.DOTALL)

NO_CODE_NOTE = ("No valid code block was found in your response, so nothing "
                "was executed. Answer with a Thought, then a ```python code "
                "block``` ending with <end_code>.")


@dataclass
class Extraction:
    """Result of extraction: ``code`` is empty when nothing was found."""

    code: str
    note: str = ""


def extract_code(text: str) -> Extraction:
    """Return the code to execute and an explanatory note for the LLM."""
    text = text or ""
    for extractor in (_from_fences, _from_xml, _from_hermes, _from_react,
                      _from_open_fence):
        found = extractor(text)
        if found is not None and found.code.strip():
            return found
    return Extraction("", NO_CODE_NOTE)


def _from_fences(text: str) -> Optional[Extraction]:
    """(a) Closed Markdown code blocks; untagged blocks must parse."""
    blocks = []
    for language, body in _FENCE.findall(text):
        if language or _parses(body):
            blocks.append(body.strip())
    if not blocks:
        return None
    note = ""
    if len(blocks) > 1:
        note = (f"Your response had {len(blocks)} code blocks; they were "
                "executed together, in order. Prefer one block per step.")
    return Extraction("\n\n".join(blocks), note)


def _from_open_fence(text: str) -> Optional[Extraction]:
    """Unclosed code block: treat the rest of the response as code."""
    match = _OPEN_FENCE.search(text)
    if not match or text.count("```") % 2 == 0:  # every block is closed
        return None
    return Extraction(match.group(1).replace(END_CODE, "").strip(),
                      "Your code block was not closed; everything after the "
                      "opening ``` was interpreted as code. Close blocks with "
                      "``` and <end_code>.")


def _from_xml(text: str) -> Optional[Extraction]:
    """(b) Anthropic-style XML ``<invoke>`` tool calls."""
    calls = [(name, {key: _coerce(value.strip())
                     for key, value in _PARAM.findall(body)})
             for name, body in _INVOKE.findall(text)]
    return _calls_to_code(calls, "XML <invoke>")


def _from_hermes(text: str) -> Optional[Extraction]:
    """(c) JSON/Hermes ``<tool_call>`` blocks."""
    calls = []
    for raw in _TOOL_CALL.findall(text):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        arguments = data.get("arguments") or data.get("parameters") or {}
        if isinstance(arguments, str):
            arguments = _coerce(arguments)
        if isinstance(data.get("name"), str) and isinstance(arguments, dict):
            calls.append((data["name"], arguments))
    return _calls_to_code(calls, "JSON <tool_call>")


def _from_react(text: str) -> Optional[Extraction]:
    """(d) ReAct ``Action`` / ``Action Input`` pairs."""
    calls = []
    for name, raw in _REACT.findall(text):
        arguments = _coerce(raw.strip())
        if not isinstance(arguments, dict):
            arguments = {} if arguments in ("", None) else {"input": arguments}
        calls.append((name, arguments))
    return _calls_to_code(calls, "ReAct Action")


def _calls_to_code(calls: List[tuple], label: str) -> Optional[Extraction]:
    """Turn ``[(name, {arg: value})]`` into Python statements."""
    calls = [(n, a) for n, a in calls if n.isidentifier()]
    if not calls:
        return None
    lines = []
    for name, arguments in calls:
        args = ", ".join(f"{key}={value!r}" for key, value in arguments.items()
                         if key.isidentifier())
        if name == "final_answer":
            lines.append(f"final_answer({args})")
        else:
            lines.append(f"result = {name}({args})\nprint(result)")
    code = "\n".join(lines)
    return Extraction(code, f"Your {label} tool call(s) were converted to "
                            f"this Python code before execution:\n{code}")


def _coerce(value: str) -> Any:
    """Parse a JSON value when possible, otherwise keep the raw string."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _parses(code: str) -> bool:
    """True if ``code`` is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

