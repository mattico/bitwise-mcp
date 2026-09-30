"""Find register tool."""

from typing import Optional
from ..retrieval.hybrid_search import HybridSearch
from ..retrieval.formatter import ResultFormatter
from .read_docs import section_text


def find_register(
    search: HybridSearch,
    name: str,
    peripheral: Optional[str] = None,
    max_chars: int = 12000,
) -> str:
    """Find a specific register by name.

    Args:
        search: Shared HybridSearch instance (caller owns its lifecycle).
        name: Register name to find (case-insensitive; peripheral prefix optional)
        peripheral: Optional peripheral name to filter results
        max_chars: Cap on the returned register description

    Returns:
        Formatted register definition as markdown
    """
    if not name or not name.strip():
        return "Give a register name, e.g. 'RCC_BDCR' or 'GUSBCFG'."
    found = search.find_register_ex(name, peripheral)

    if found["kind"] == "ambiguous":
        cands = ", ".join(f"`{c}`" for c in found["candidates"])
        return (f"'{name}' matches several registers: {cands}\n\n"
                "Call find_register again with the full name.")
    if found["kind"] == "none":
        return (f"Register '{name}' not found in the register index. Try search_docs "
                f"with the register's name or its description.")

    result = found["results"][0]
    header = f"# {found['name']}\n\n"
    if result.structured_data and result.structured_data.get("registers"):
        return header + ResultFormatter.format_register(result)
    # A prose register section: return the whole section, which may span chunks.
    text, truncated = section_text(search.metadata_store, result.chunk_id, max_chars)
    out = header + text + "\n\n" + ResultFormatter.source_line(result)
    if truncated:
        out += f"\n\n(truncated at {max_chars} chars; read_section('{result.chunk_id}', offset={max_chars}) continues)"
    return out
