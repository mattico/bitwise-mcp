"""Search documentation tool."""

from typing import Optional
from ..retrieval.hybrid_search import HybridSearch
from ..retrieval.formatter import ResultFormatter


def search_docs(
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
        doc_filter: Optional document ID, filename or title fragment

    Returns:
        Formatted search results as markdown
    """
    if not query or not query.strip():
        return "Give a query: keywords or a question about the documentation."
    top_k = max(1, min(top_k, 25))
    doc_id, error = search.resolve_doc(doc_filter)
    if error:
        return error
    response = search.search_ex(query, top_k, doc_id)
    return ResultFormatter.format_response(response)
