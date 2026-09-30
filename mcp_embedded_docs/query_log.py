"""Append-only JSONL log of MCP tool calls (arguments, timing, outcome).

The MCP host swallows the server's stderr, so without this there is no record
of what agents asked, how long it took, or which queries found nothing.

Location: $BITWISE_MCP_LOG_DIR, default <index dir>/logs/queries.jsonl.
Rotates to queries.jsonl.1 past $BITWISE_MCP_LOG_MAX_MB (default 16).
Set BITWISE_MCP_LOG=0 to disable.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_PREVIEW_CHARS = 600
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("BITWISE_MCP_LOG", "1").lower() not in ("0", "false", "no")


def log_path(index_dir: Path) -> Path:
    d = Path(os.environ.get("BITWISE_MCP_LOG_DIR") or index_dir / "logs")
    return d / "queries.jsonl"


def record(index_dir: Path, tool: str, args: Dict[str, Any], *, ms: float,
           result: Optional[str] = None, error: Optional[str] = None,
           extra: Optional[Dict[str, Any]] = None) -> None:
    """Write one call record. Never raises: logging must not break a tool call."""
    if not enabled():
        return
    try:
        entry: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": tool,
            "args": {k: v for k, v in args.items() if v not in (None, "")},
            "ms": round(ms, 1),
        }
        if error is not None:
            entry["error"] = error
        if result is not None:
            entry["chars"] = len(result)
            entry["preview"] = result[:_PREVIEW_CHARS]
        if extra:
            entry.update(extra)
        path = log_path(index_dir)
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            limit = float(os.environ.get("BITWISE_MCP_LOG_MAX_MB", "16")) * 1e6
            if path.exists() and path.stat().st_size > limit:
                os.replace(path, path.with_suffix(".jsonl.1"))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - best effort
        pass
