"""Find register tool."""

from typing import Optional
from ..retrieval.hybrid_search import HybridSearch
from ..retrieval.formatter import ResultFormatter


async def find_register(
    search: HybridSearch,
    name: str,
    peripheral: Optional[str] = None,
) -> str:
    """Find a specific register by name.

    Args:
        search: Shared HybridSearch instance (caller owns its lifecycle).
        name: Register name to find
        peripheral: Optional peripheral name to filter results

    Returns:
        Formatted register definition as markdown
    """
    result = search.find_register(name, peripheral)

    if not result:
        return f"Register '{name}' not found."

    return ResultFormatter.format_register(result)