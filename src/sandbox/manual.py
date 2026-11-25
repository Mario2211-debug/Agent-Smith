"""Generate the sandbox manual fed to the LLM in its system prompt.

The manual is built dynamically from the connected MCP server's tool
schemas, so connecting a different server automatically changes it.
"""

from typing import Any, List, Optional

from src.mcp.schema import ToolSpec

BUILTINS_SECTION = """\
## Built-in functions (always available, not MCP tools)
- final_answer(answer: str) -> None
    Submit your final answer and stop. For MBPP pass the solution code,
    for SWE-bench pass get_patch().
- list_resources() -> str / read_resource(uri: str) -> str
    List / read the MCP server's resources.
- list_prompts() -> str / get_prompt(name: str, arguments: dict = None) -> str
    List / render the MCP server's prompts."""


def describe_tool(tool: ToolSpec) -> str:
    """Return the manual entry of one tool: signature, description, params."""
    lines = [f"- {tool.signature} -> str"]
    for text_line in (tool.description or "No description.").splitlines():
        if text_line.strip():
            lines.append(f"    {text_line.strip()}")
    for param in tool.params:
        if param.description:
            lines.append(f"    * {param.name}: {param.description}")
    return "\n".join(lines)


def build_manual(tools: List[ToolSpec], config: Any,
                 server_name: Optional[str] = None) -> str:
    """Build the complete manual for the given tools and sandbox config.

    Args:
        tools: tool specs discovered from the MCP server (may be empty).
        config: a ``SandboxSettings`` (imports, directories, limits).
        server_name: name reported by the MCP server, for context.
    """
    sections = [
        "# Sandbox manual",
        "Your Python code runs in a restricted sandbox. Variables, functions "
        "and imports persist between steps. Use print() to see results: only "
        "printed output (and the value of a final bare expression) is "
        "returned to you.",
        BUILTINS_SECTION,
    ]
    if tools:
        header = f"## MCP tools (server: {server_name or 'unknown'})"
        sections.append(header + "\nCall them as normal Python functions; "
                        "each returns a string.\n"
                        + "\n".join(describe_tool(tool) for tool in tools))
    else:
        sections.append("## MCP tools\nNo MCP server is connected: only the "
                        "built-in functions are available.")
    sections.append(
        "## Restrictions\n"
        f"- Allowed imports: {', '.join(config.authorized_imports) or 'none'}"
        "\n"
        f"- File access only inside: "
        f"{', '.join(config.allowed_directories) or 'nowhere'}\n"
        "- No network access, no subprocesses, no eval/exec/compile/input.\n"
        f"- Each execution is limited to {config.max_execution_time_seconds}"
        f" s and {config.max_memory_mb} MB (MCP tool time is not counted).\n"
        f"- Printed output is truncated after {config.max_output_chars} "
        f"characters, tool results after {config.max_tool_output_chars}.")
    return "\n\n".join(sections)
