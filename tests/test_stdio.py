"""The Windows stdin reader: correct line framing, and no deadlock on lazy
native imports (the bug that made first searches hang for minutes)."""

import os
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows pipe reader")

_ECHO = r"""
import anyio, sys
from mcp_embedded_docs.stdio import PolledPipeStdin
async def main():
    reader = PolledPipeStdin.for_stdin()
    assert reader is not None, "stdin should be a pipe"
    async for line in reader:
        sys.stdout.write(repr(line) + "\n"); sys.stdout.flush()
anyio.run(main)
"""

# Starts the polled reader, then imports numpy on another task while no more
# input arrives -- the pattern that deadlocked with the SDK's blocking reader.
_IMPORT_WHILE_IDLE = r"""
import anyio, sys, time
from mcp_embedded_docs.stdio import PolledPipeStdin
async def main():
    reader = PolledPipeStdin.for_stdin()
    first = await reader.__anext__()
    async def read_rest():
        async for _ in reader:
            pass
    async with anyio.create_task_group() as tg:
        tg.start_soon(read_rest)
        await anyio.sleep(0.2)
        t0 = time.perf_counter()
        await anyio.to_thread.run_sync(lambda: __import__("numpy"))
        sys.stdout.write(f"imported {time.perf_counter() - t0:.2f}\n"); sys.stdout.flush()
        tg.cancel_scope.cancel()
anyio.run(main)
"""


def _env():
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def test_reader_frames_lines_across_partial_writes_and_eof():
    p = subprocess.Popen([sys.executable, "-c", _ECHO], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, env=_env())
    p.stdin.write(b'{"a": 1}\n{"b"')
    p.stdin.flush()
    time.sleep(0.2)
    p.stdin.write(b': 2}\n\xc3\xa9 tail-without-newline')
    p.stdin.close()
    out = p.communicate(timeout=30)[0].decode("utf-8").splitlines()
    assert out == ["'{\"a\": 1}\\n'", "'{\"b\": 2}\\n'", "'é tail-without-newline'"]


def test_lazy_native_import_does_not_wait_for_stdin():
    p = subprocess.Popen([sys.executable, "-c", _IMPORT_WHILE_IDLE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, env=_env())
    p.stdin.write(b"hello\n")
    p.stdin.flush()
    lines: list = []
    reader = threading.Thread(target=lambda: lines.append(p.stdout.readline().decode()),
                              daemon=True)
    reader.start()
    try:
        # With the SDK's blocking reader this waits until stdin sees more data.
        reader.join(timeout=60)
        out = lines[0] if lines else "no output: import blocked on stdin"
    finally:
        # keep stdin open until the import finished: closing it would unblock
        # the old reader too and hide the bug
        p.stdin.close()
        p.wait(timeout=30)
    assert out.startswith("imported"), out
