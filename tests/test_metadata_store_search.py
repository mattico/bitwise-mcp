"""Tests for literal-friendly keyword search behavior in MetadataStore."""

from __future__ import annotations

from mcp_embedded_docs.indexing.metadata_store import MetadataStore


def _seed_chunk(store: MetadataStore, *, doc_id: str, chunk_id: str, text: str) -> None:
    store.add_document(doc_id=doc_id, filename=f"{doc_id}.pdf")
    store.add_chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_type="text",
        text=text,
        page_start=1,
        page_end=1,
    )


def test_keyword_search_finds_hex_literal_with_doc_filter(tmp_path):
    db_path = tmp_path / "metadata.db"
    store = MetadataStore(db_path)

    try:
        _seed_chunk(
            store,
            doc_id="an2606",
            chunk_id="c1",
            text="Table 226 row: STM32H74xxx/75xxx system memory start 0x1FF09800.",
        )
        _seed_chunk(
            store,
            doc_id="rm0433",
            chunk_id="c2",
            text="STM32H743 reference manual general memory map.",
        )

        results = store.keyword_search("0x1FF09800", top_k=5, doc_filter="an2606")

        assert results
        assert results[0][0] == "c1"
    finally:
        store.close()


def test_keyword_search_handles_literal_heavy_query_without_fts_failures(tmp_path):
    db_path = tmp_path / "metadata.db"
    store = MetadataStore(db_path)

    try:
        _seed_chunk(
            store,
            doc_id="an2606",
            chunk_id="c1",
            text="STM32H74xxx/75xxx bootloader entry 0x1FF09800 from Table 226.",
        )

        # Includes punctuation-heavy tokens that are commonly brittle in MATCH.
        results = store.keyword_search("STM32H74xxx/75xxx 0x1FF09800 [Table]", top_k=5)

        assert results
        assert results[0][0] == "c1"
    finally:
        store.close()
