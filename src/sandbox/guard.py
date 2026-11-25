"""Security guard installed inside the sandbox worker process.

Layers of defence (standard library only):

1. Static check: the AST is rejected if it touches introspection attributes
   commonly used to escape restricted environments (``__subclasses__``...).
2. Restricted builtins: dangerous builtins are removed, ``__import__`` only
   accepts allowlisted modules and ``open`` refuses raw file descriptors.
3. Audit hook (``sys.addaudithook``, cannot be removed): blocks processes,
   sockets, ctypes and file access outside the allowed directories, even if
   the code reaches the real ``os`` module through another object.
4. Resource limits: ``RLIMIT_AS`` caps the worker's memory.
"""

import ast
import builtins
import fnmatch
import os
import resource
import sys
from typing import Any, Callable, Dict, Iterable, List

# Private modules that allowed modules import lazily from C code (through
# the caller's __import__), e.g. datetime.strptime -> _strptime.
INTERNAL_MODULES = frozenset({"_strptime"})
REMOVED_BUILTINS = (
    "eval", "exec", "compile", "input", "breakpoint", "help",
    "exit", "quit", "copyright", "credits", "license",
)
BLOCKED_ATTRIBUTES = frozenset({
    "__subclasses__", "__globals__", "__builtins__", "__code__",
    "__closure__", "__bases__", "__base__", "__mro__", "__loader__",
    "__spec__", "__getattribute__", "__reduce__", "__reduce_ex__",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code",
    "gi_frame", "gi_code", "cr_frame", "ag_frame", "tb_frame",
})
BLOCKED_EVENT_PREFIXES = (
    "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
    "os.fork", "os.forkpty", "os.kill", "os.killpg", "os.putenv",
    "os.unsetenv", "os.startfile", "socket.", "ctypes.", "pty.", "fcntl.",
    "mmap.", "sys._getframe", "sys._current_frames", "sys.setprofile",
    "sys.settrace", "gc.", "code.__new__", "urllib.", "http.", "ftplib.",
    "smtplib.", "telnetlib.", "webbrowser.", "winreg.", "_winapi.",
)
PATH_EVENTS = frozenset({
    "open", "os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.listdir",
    "os.scandir", "os.chdir", "os.chmod", "os.chown", "os.link",
    "os.symlink", "os.truncate", "os.utime", "shutil.copyfile",
    "shutil.rmtree", "shutil.move", "shutil.copytree",
})


class SecurityViolation(PermissionError):
    """Raised when sandboxed code attempts a forbidden operation."""

    __module__ = "sandbox"  # shown as 'sandbox.SecurityViolation'


class Guard:
    """Holds the policy and enforces it while user code is running.

    ``active`` is True only while agent code executes; ``trusted`` is a
    counter raised while the worker itself performs privileged work
    (importing an allowed module, formatting a traceback).
    """

    def __init__(self, imports: Iterable[str], directories: Iterable[str],
                 max_memory_mb: int):
        """Store the allowlists and the memory budget."""
        self.imports: List[str] = list(imports)
        self.directories = [os.path.realpath(d) for d in directories]
        self.max_memory_mb = max_memory_mb
        self.active = False
        self.trusted = 0
        self._in_hook = False
        self._real_import = builtins.__import__

    # ------------------------------------------------------------ installers
    def install(self) -> None:
        """Apply the memory limit and the irreversible audit hook."""
        limit = self.max_memory_mb * 1024 * 1024
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            if hasattr(resource, name):
                try:
                    resource.setrlimit(getattr(resource, name), (limit, limit))
                except (ValueError, OSError):
                    pass
        sys.addaudithook(self._audit)

    def builtins_dict(self) -> Dict[str, Any]:
        """Return the restricted builtins mapping for the user namespace."""
        safe = dict(vars(builtins))
        for name in REMOVED_BUILTINS:
            safe.pop(name, None)
        safe["__import__"] = self.safe_import
        safe["open"] = self.safe_open
        safe["getattr"] = self.safe_getattr
        return safe

    # --------------------------------------------------------------- checks
    def is_allowed_module(self, name: str) -> bool:
        """True if ``name`` matches an entry of the import allowlist."""
        if name in INTERNAL_MODULES:
            return True
        return any(name == pattern or fnmatch.fnmatchcase(name, pattern)
                   for pattern in self.imports)

    def is_allowed_path(self, path: Any) -> bool:
        """True if ``path`` resolves inside one of the allowed directories."""
        if isinstance(path, int):
            return False
        try:
            real = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        return any(real == d or real.startswith(d.rstrip(os.sep) + os.sep)
                   for d in self.directories)

    def check_ast(self, tree: ast.AST) -> None:
        """Reject code that accesses blocked introspection attributes."""
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            if name in BLOCKED_ATTRIBUTES:
                line = getattr(node, "lineno", "?")
                raise SecurityViolation(
                    f"line {line}: access to '{name}' is not allowed")

    # ------------------------------------------------------ safe builtins
    def safe_import(self, name: str, globals_: Any = None, locals_: Any = None,
                    fromlist: Any = (), level: int = 0) -> Any:
        """``__import__`` replacement enforcing the allowlist."""
        if level != 0 or not self.is_allowed_module(name):
            raise ImportError(f"Import of '{name}' is not allowed in the "
                              "sandbox")
        module = self.trusted_call(self._real_import, name, globals_,
                                   locals_, fromlist, level)
        for item in fromlist or ():
            sub = getattr(module, item, None)
            if type(sub) is type(sys) and \
                    not self.is_allowed_module(f"{name}.{item}"):
                raise ImportError(f"Import of '{name}.{item}' is not allowed")
        return module

    def safe_open(self, file: Any, *args: Any, **kwargs: Any) -> Any:
        """``open`` replacement: no raw descriptors, path checked by audit."""
        if isinstance(file, int):
            raise SecurityViolation("Opening file descriptors is not allowed")
        return open(file, *args, **kwargs)

    @staticmethod
    def safe_getattr(obj: Any, name: str, *default: Any) -> Any:
        """``getattr`` replacement refusing blocked attribute names."""
        if name in BLOCKED_ATTRIBUTES:
            raise SecurityViolation(f"access to '{name}' is not allowed")
        return getattr(obj, name, *default)

    def trusted_call(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run ``func`` with the audit restrictions temporarily lifted."""
        self.trusted += 1
        try:
            return func(*args)
        finally:
            self.trusted -= 1

    # ------------------------------------------------------------ audit hook
    def _audit(self, event: str, args: tuple) -> None:
        """Audit hook: raise SecurityViolation for forbidden events."""
        if not self.active or self.trusted or self._in_hook:
            return
        self._in_hook = True
        try:
            self._check_event(event, args)
        finally:
            self._in_hook = False

    def _check_event(self, event: str, args: tuple) -> None:
        """Decide whether a single audit event is allowed."""
        if event.startswith(BLOCKED_EVENT_PREFIXES):
            raise SecurityViolation(f"Operation '{event}' is not allowed in "
                                    "the sandbox")
        if event == "object.__getattr__" and len(args) > 1 \
                and args[1] in BLOCKED_ATTRIBUTES:
            raise SecurityViolation(f"access to '{args[1]}' is not allowed")
        if event in PATH_EVENTS:
            paths = args[:2] if event in ("os.rename", "os.link",
                                          "os.symlink") else args[:1]
            for path in paths:
                path = "." if path is None else path
                if not self.is_allowed_path(path):
                    raise SecurityViolation(
                        f"Access to '{path}' is outside the allowed "
                        f"directories {self.directories}")
