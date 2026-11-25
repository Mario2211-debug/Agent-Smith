"""Synchronous MCP client supporting stdio and streamable HTTP transports.

The official MCP SDK is asynchronous, while the sandbox is synchronous. This
module runs the SDK session inside a private event loop on a background
thread and exposes blocking methods on top of it.
"""

import asyncio
import os
import shlex
import threading
from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, Optional

from src.mcp.schema import ToolResult, ToolSpec, content_to_text

CONNECT_TIMEOUT_SECONDS = 30.0
CALL_TIMEOUT_SECONDS = 900.0


class MCPError(RuntimeError):
    """Raised when the MCP server cannot be reached or a request fails."""


class MCPClient:
    """Blocking facade over an ``mcp.ClientSession``.

    Build it with :meth:`stdio` or :meth:`http`, then use it as a context
    manager (or call :meth:`connect` / :meth:`close`).
    """

    def __init__(self, transport_factory: Callable[[], Any], label: str):
        """Store the transport factory; nothing is started until connect()."""
        self._transport_factory = transport_factory
        self.label = label
        self.server_name = label
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Any = None
        self._stop: Optional[asyncio.Event] = None
        self._lifecycle: Any = None
        self._tools: Optional[List[ToolSpec]] = None

    @classmethod
    def stdio(cls, command: str) -> "MCPClient":
        """Create a client that spawns ``command`` and talks over stdio."""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        argv = shlex.split(command)
        if not argv:
            raise MCPError("Empty --mcp-stdio command.")
        params = StdioServerParameters(command=argv[0], args=argv[1:],
                                       env=dict(os.environ))
        return cls(lambda: stdio_client(params), f"stdio:{command}")

    @classmethod
    def http(cls, url: str) -> "MCPClient":
        """Create a client for a streamable HTTP MCP endpoint."""
        from mcp.client import streamable_http

        factory = getattr(streamable_http, "streamablehttp_client", None) \
            or getattr(streamable_http, "streamable_http_client")
        return cls(lambda: factory(url), f"http:{url}")

    # ----------------------------------------------------------- lifecycle
    def connect(self) -> "MCPClient":
        """Start the event loop thread and initialise the MCP session."""
        if self._session is not None:
            return self
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True,
                         name="mcp-client").start()
        ready = threading.Event()
        self._lifecycle = asyncio.run_coroutine_threadsafe(
            self._run_session(ready), self._loop)
        ready.wait(CONNECT_TIMEOUT_SECONDS)
        if self._session is None:
            try:
                self._lifecycle.result(timeout=2)
                error: Any = "the server closed the connection"
            except Exception as exc:  # noqa: BLE001 - reported below
                error = exc or "timed out"
            self.close()
            raise MCPError(f"Could not connect to MCP server "
                           f"({self.label}): {error}")
        return self

    async def _run_session(self, ready: threading.Event) -> None:
        """Own the transport and session contexts for their whole lifetime.

        anyio requires these contexts to be entered and exited by the same
        task, so this coroutine lives until :meth:`close` sets ``_stop``.
        """
        from mcp import ClientSession

        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(
                    self._transport_factory())
                session = await stack.enter_async_context(
                    ClientSession(streams[0], streams[1]))
                info = await session.initialize()
                self.server_name = getattr(info.serverInfo, "name", self.label)
                self._session = session
                ready.set()
                await self._stop.wait()
        finally:
            self._session = None
            ready.set()

    def close(self) -> None:
        """Shut the session down and stop the background loop."""
        if self._loop is None:
            return
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._lifecycle is not None:
            try:
                self._lifecycle.result(timeout=5)
            except Exception:  # noqa: BLE001 - best effort shutdown
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop = None

    def __enter__(self) -> "MCPClient":
        """Connect when used as a context manager."""
        return self.connect()

    def __exit__(self, *exc_info: Any) -> None:
        """Close when leaving the context manager."""
        self.close()

    # ------------------------------------------------------------ requests
    def _request(self, method: str, *args: Any) -> Any:
        """Run ``session.<method>(*args)`` on the loop and wait for it."""
        if self._session is None or self._loop is None:
            raise MCPError("MCP client is not connected.")
        coroutine = getattr(self._session, method)(*args)
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=CALL_TIMEOUT_SECONDS)
        except Exception as error:
            raise MCPError(f"MCP {method} failed: {error}") from error

    def list_tools(self) -> List[ToolSpec]:
        """Return (and cache) the server's tools as :class:`ToolSpec`."""
        if self._tools is None:
            result = self._request("list_tools")
            self._tools = [ToolSpec.from_schema(t.name, t.description,
                                                t.inputSchema)
                           for t in result.tools]
        return self._tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool and flatten its content blocks into text."""
        result = self._request("call_tool", name, arguments)
        text = content_to_text(result.content)
        if not text and getattr(result, "structuredContent", None):
            text = str(result.structuredContent)
        return ToolResult(text=text, is_error=bool(result.isError))

    def list_resources(self) -> str:
        """Return a readable list of the server's resources."""
        resources = self._request("list_resources").resources
        lines = [f"{r.uri} - {r.name}: {r.description or ''}".rstrip(": ")
                 for r in resources]
        return "\n".join(lines) or "No resources available."

    def read_resource(self, uri: str) -> str:
        """Return the text content of a resource."""
        contents = self._request("read_resource", uri).contents
        return "\n".join(getattr(c, "text", f"[binary {c.mimeType}]")
                         for c in contents)

    def list_prompts(self) -> str:
        """Return a readable list of the server's prompts."""
        prompts = self._request("list_prompts").prompts
        lines = []
        for prompt in prompts:
            args = ", ".join(a.name for a in (prompt.arguments or []))
            lines.append(f"{prompt.name}({args}): {prompt.description or ''}")
        return "\n".join(lines) or "No prompts available."

    def get_prompt(self, name: str,
                   arguments: Optional[Dict[str, str]] = None) -> str:
        """Render a prompt and return its messages as text."""
        result = self._request("get_prompt", name, arguments or {})
        return "\n\n".join(f"[{m.role}] {content_to_text([m.content])}"
                           for m in result.messages)

