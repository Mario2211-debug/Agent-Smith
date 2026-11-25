"""Bridge between worker requests and the MCP client, on the host side.

MCP tool actions happen here, outside the sandbox security domain: the
worker only receives the (size-limited) text result.
"""

from typing import Any, Dict, List

from src.mcp.schema import ToolSpec

MCP_HELPERS = ("list_resources", "read_resource", "list_prompts",
               "get_prompt")


class ToolBridge:
    """Executes ``tool`` / ``mcp`` requests and limits their output size.

    ``notes`` collects explanations (e.g. truncation) for the current
    execution so they can be shown to the LLM.
    """

    def __init__(self, mcp: Any, max_chars: int):
        """Discover the server's tools (if any) and store the size limit."""
        self.mcp = mcp
        self.max_chars = max_chars
        self.tools: List[ToolSpec] = mcp.list_tools() if mcp else []
        self._by_name: Dict[str, ToolSpec] = {
            tool.python_name: tool for tool in self.tools}
        self.notes: List[str] = []

    def worker_specs(self) -> List[dict]:
        """Describe the tools for the worker (names and parameter order)."""
        return [{"name": t.python_name, "doc": t.description,
                 "params": [p.name for p in t.params]} for t in self.tools]

    def answer(self, message: dict) -> dict:
        """Handle one worker request and return the ``reply`` message."""
        try:
            return {"op": "reply", "ok": True, "value": self._dispatch(message)}
        except Exception as error:  # noqa: BLE001 - forwarded to agent code
            return {"op": "reply", "ok": False, "error": str(error)}

    def _dispatch(self, message: dict) -> str:
        """Route a ``tool`` or ``mcp`` request to the MCP client."""
        if self.mcp is None:
            raise RuntimeError("No MCP server is connected to the sandbox.")
        if message.get("op") == "mcp":
            method = message.get("method")
            if method not in MCP_HELPERS:
                raise RuntimeError(f"Unknown MCP helper '{method}'")
            value = getattr(self.mcp, method)(*message.get("args", []))
            return self._limit(value, method)
        tool = self._by_name.get(message.get("name"))
        if tool is None:
            raise RuntimeError(f"Unknown tool '{message.get('name')}'")
        result = self.mcp.call_tool(tool.name, message.get("arguments", {}))
        if result.is_error:
            raise RuntimeError(f"{tool.name} failed: {result.text}")
        return self._limit(result.text, tool.name)

    def _limit(self, text: Any, source: str) -> str:
        """Truncate a result to ``max_chars`` and record a note about it."""
        text = str(text)
        if len(text) <= self.max_chars:
            return text
        self.notes.append(
            f"Output of {source}() was truncated from {len(text)} to "
            f"{self.max_chars} characters. Request a smaller range or a "
            "narrower pattern to see the rest.")
        omitted = len(text) - self.max_chars
        return text[:self.max_chars] + f"\n... [truncated {omitted} chars]"
