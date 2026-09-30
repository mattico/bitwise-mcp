"""stdio transport that is safe for lazy native imports on Windows.

The MCP SDK reads stdin with a blocking read on a worker thread. On Windows,
while a synchronous ReadFile is pending on the stdin pipe, loading a native
extension whose CRT start-up queries the standard handles (numpy, torch, faiss,
PyMuPDF, ...) blocks until the next byte arrives on stdin. A tool call that
lazily imports one of them therefore hangs until the client happens to send
another message -- a cancel after its timeout, typically minutes later.

On Windows this reader never leaves a read pending: it asks PeekNamedPipe how
many bytes are waiting and only reads those, sleeping briefly otherwise.
Elsewhere, and when stdin is not a pipe, the SDK's own reader is used.
"""

import logging
import sys
from typing import Optional

import anyio

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.005  # seconds between peeks while stdin is idle


class PolledPipeStdin:
    """Async line iterator over a Windows stdin pipe without blocking reads."""

    def __init__(self, handle: int, fd: int):
        import ctypes
        from ctypes import wintypes

        self._fd = fd
        self._handle = handle
        self._buf = b""
        self._avail = wintypes.DWORD(0)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._peek = kernel32.PeekNamedPipe
        self._peek.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
            wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
        ]
        self._peek.restype = wintypes.BOOL
        self._byref = ctypes.byref

    @classmethod
    def for_stdin(cls) -> Optional["PolledPipeStdin"]:
        """A reader for this process's stdin, or None if it is not a pipe."""
        if sys.platform != "win32":
            return None
        try:
            import msvcrt

            fd = sys.stdin.fileno()
            reader = cls(msvcrt.get_osfhandle(fd), fd)
            if not reader._peek(reader._handle, None, 0, None,
                                reader._byref(reader._avail), None):
                return None  # a console or file, not a pipe
            return reader
        except Exception:  # noqa: BLE001 - fall back to the SDK reader
            logger.debug("stdin is not a pollable pipe", exc_info=True)
            return None

    def _available(self) -> int:
        """Bytes waiting in the pipe; -1 once the writer has closed it."""
        if not self._peek(self._handle, None, 0, None, self._byref(self._avail), None):
            return -1
        return self._avail.value

    def __aiter__(self) -> "PolledPipeStdin":
        return self

    async def __anext__(self) -> str:
        import os

        while b"\n" not in self._buf:
            avail = self._available()
            if avail < 0:
                if self._buf:
                    line, self._buf = self._buf, b""
                    return line.decode("utf-8", "replace")
                raise StopAsyncIteration
            if avail:
                # Exactly what is buffered, so this read returns immediately.
                self._buf += os.read(self._fd, avail)
            else:
                await anyio.sleep(_POLL_INTERVAL)
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8", "replace") + "\n"


def run_stdio(mcp) -> None:
    """Run a FastMCP server over stdio with the Windows-safe stdin reader."""
    reader = PolledPipeStdin.for_stdin()
    if reader is None:
        mcp.run(transport="stdio")
        return

    from mcp.server.stdio import stdio_server

    async def _serve() -> None:
        async with stdio_server(stdin=reader) as (read_stream, write_stream):  # type: ignore[arg-type]
            server = mcp._mcp_server
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_serve)
