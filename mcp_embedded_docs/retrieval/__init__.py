"""Search and retrieval modules."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class SearchResult:
    """A search result."""
    chunk_id: str
    score: float
    text: str
    structured_data: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    doc_id: str
    page_start: int
    page_end: int
    chunk_type: str = "text"
    section: Optional[str] = None
    doc_title: Optional[str] = None
    # Other chunks of the same section that also matched (collapsed into this one).
    more_in_section: int = 0
    # Which channels found it: "keyword", "semantic".
    channels: List[str] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Results of one search plus how it was run."""
    query: str
    results: List[SearchResult]
    terms: List[str] = field(default_factory=list)
    keyword_mode: str = ""
    semantic: str = ""  # "on", or why it was skipped
    notes: List[str] = field(default_factory=list)
