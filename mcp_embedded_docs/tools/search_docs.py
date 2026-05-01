"""Search documentation tool."""

from typing import Optional
from ..retrieval.hybrid_search import HybridSearch
from ..retrieval.formatter import ResultFormatter


async def search_docs(
    search: HybridSearch,
    query: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
) -> str:
    """Search documentation using hybrid search.

    Args:
        search: Shared HybridSearch instance (caller owns its lifecycle).
        query: Search query
        top_k: Number of results to return (default: 5)
        doc_filter: Optional document ID to filter results

    Returns:
        Formatted search results as markdown
    """
    results = search.search(query, top_k, doc_filter)
    return ResultFormatter.format_results(results, top_k)