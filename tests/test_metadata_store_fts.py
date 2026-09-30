"""FTS schema migration, trigger correctness and register lookup in MetadataStore."""

import sqlite3

from mcp_embedded_docs.indexing.metadata_store import MetadataStore, section_path


def _chunk(chunk_id, text, section="1 Intro", doc_id="d1", structured=None, chunk_type="text"):
    return {
        "chunk_id": chunk_id, "doc_id": doc_id, "chunk_type": chunk_type, "text": text,
        "page_start": 0, "page_end": 0, "structured_data": structured,
        "metadata": {"section_title": section},
    }


def _fts_ids(store, match):
    return {r[0] for r in store.conn.execute(
        "SELECT c.id FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ?", (match,))}


def test_stemming_and_title_column(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        store.add_chunks([
            _chunk("a", "Setup and hold timings for the interface.", section="4.3 Interface"),
            _chunk("b", "Unrelated text about something else entirely.", section="7.2 Timing requirements"),
        ])
        hits = [h[0] for h in store.keyword_search("timing", 5)]
        # porter stemming matches 'timings'; the title column matches 'Timing requirements'
        assert set(hits) == {"a", "b"}
        # title matches are weighted above body matches
        assert hits[0] == "b"
    finally:
        store.close()


def test_replacing_and_deleting_chunks_keeps_fts_consistent(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        store.add_chunks([_chunk("a", "alpha widget"), _chunk("b", "beta widget")])
        # re-adding an existing id must not leave a stale FTS row behind
        store.add_chunks([_chunk("a", "gamma gizmo")])
        assert _fts_ids(store, "alpha") == set()
        assert _fts_ids(store, "gamma") == {"a"}

        store.delete_document("d1")
        assert _fts_ids(store, "widget") == set()
        store.conn.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)")
    finally:
        store.close()


def test_migrates_pre_0_4_fts_schema(tmp_path):
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT NOT NULL, title TEXT,
                                version TEXT, index_date TEXT NOT NULL);
        CREATE TABLE chunks (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, chunk_type TEXT NOT NULL,
            section_hierarchy TEXT, page_start INTEGER, page_end INTEGER, text TEXT NOT NULL,
            structured_data TEXT, metadata TEXT);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, text, content='chunks',
                                                  content_rowid='rowid');
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
        END;
        INSERT INTO documents VALUES ('d1', 'd1.pdf', NULL, NULL, '2026-01-01');
        INSERT INTO chunks VALUES ('c1', 'd1', 'text', 'Clock security', 0, 0,
                                   'The HSE oscillators are monitored.', NULL, NULL);
    """)
    con.commit()
    con.close()

    store = MetadataStore(db)
    try:
        sql = store.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='chunks_fts'").fetchone()[0]
        assert "porter" in sql and "section_hierarchy" in sql
        # rebuilt from existing rows, stemmed: 'oscillator' matches 'oscillators'
        assert [h[0] for h in store.keyword_search("oscillator", 5)] == ["c1"]
        assert store.list_documents()[0]["path"] is None
    finally:
        store.close()


def test_find_register_tolerates_case_and_missing_prefix(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        regs = {"peripheral": "OTG", "registers": [{"name": "OTG_GUSBCFG"}]}
        store.add_chunks([
            _chunk("r1", "GUSBCFG table", structured=regs, chunk_type="bitfield_definition"),
            _chunk("r2", "RCC_CR table", structured={"peripheral": "RCC", "registers": [{"name": "RCC_CR"}]}),
            _chunk("r3", "CRS_CR table", structured={"peripheral": "CRS", "registers": [{"name": "CRS_CR"}]}),
            _chunk("s1", "Address offset 0x70", section="8.7.26 RCC backup domain control register (RCC_BDCR)"),
        ])
        assert store.find_register_matches("otg_gusbcfg")["name"] == "OTG_GUSBCFG"
        m = store.find_register_matches("GUSBCFG")
        assert m["kind"] == "exact" and m["chunks"][0]["id"] == "r1"

        m = store.find_register_matches("CR")
        assert m["kind"] == "ambiguous" and set(m["candidates"]) == {"RCC_CR", "CRS_CR"}

        # prose register sections are found through their '(NAME)' title
        m = store.find_register_matches("bdcr")
        assert m["kind"] == "exact" and m["name"] == "RCC_BDCR" and m["chunks"][0]["id"] == "s1"

        assert store.find_register_matches("NOPE")["kind"] == "none"
        assert store.find_register("GUSBCFG")["id"] == "r1"
    finally:
        store.close()


def test_find_register_merges_sources_and_respects_peripheral_boundaries(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        store.add_chunks([
            _chunk("crc", "t", structured={"peripheral": "CRC", "registers": [{"name": "CRC_IDR"}]}),
            _chunk("gpio", "GPIO input", section="11.4.5 GPIO port input data register (GPIOx_IDR) (x = A to K)"),
            _chunk("qspi", "t", structured={"peripheral": "QUADSPI", "registers": [{"name": "QUADSPI_CR"}]}),
            _chunk("spi", "t", structured={"peripheral": "SPI", "registers": [{"name": "SPI_CR1"}]}),
            _chunk("d2d", "t", structured={"peripheral": "DMA2D", "registers": [{"name": "DMA2D_CR"}]}),
        ])
        # a table suffix match must not hide a section-title match
        m = store.find_register_matches("IDR")
        assert m["kind"] == "ambiguous" and set(m["candidates"]) == {"CRC_IDR", "GPIOx_IDR"}
        # the name comes from the parenthesis that contains it, not the last one
        m = store.find_register_matches("IDR", peripheral="GPIO")
        assert m["kind"] == "exact" and m["name"] == "GPIOx_IDR"
        # an instance resolves to the generic register
        m = store.find_register_matches("GPIOA_IDR")
        assert m["kind"] == "exact" and m["name"] == "GPIOx_IDR" and m["chunks"][0]["id"] == "gpio"
        # peripheral matches at a name boundary only
        assert store.find_register_matches("CR", peripheral="SPI")["kind"] == "none"
        assert store.find_register_matches("CR", peripheral="DMA")["kind"] == "none"
        assert store.find_register_matches("CR1", peripheral="SPI")["name"] == "SPI_CR1"
    finally:
        store.close()


def test_substring_fallback_requires_every_term(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        store.add_chunks([_chunk("a", "Supply 1.8 V typical.")])
        # found by the any-term widening (it does contain 1.8), never by the
        # substring scan, which would need 'zzqqxx' too
        res = store.keyword_search_ex("1.8 zzqqxx", 5)
        assert "substring" not in res.mode and "relaxed-or" in res.mode
        assert store._literal_fallback(["1.8", "zzqqxx"], 5, None) == []
        assert [h[0] for h in store.keyword_search("1.8 supply", 5)] == ["a"]
    finally:
        store.close()


def test_sections_with_the_same_leaf_title_stay_separate(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        store.add_chunks([
            _chunk("a1", "[Spec > Transmit > Signals]\nTX signals", section="Signals"),
            _chunk("a2", "[Spec > Transmit > Signals]\nmore TX", section="Signals"),
            _chunk("b1", "[Spec > Receive > Signals]\nRX signals", section="Signals"),
        ])
        assert [c["id"] for c in store.get_section_chunks("a2")] == ["a1", "a2"]
        assert [c["id"] for c in store.get_section_chunks("b1")] == ["b1"]
    finally:
        store.close()


def test_section_titles_with_brackets_stay_separate(tmp_path):
    assert section_path("[Doc > Bits [31:0] config]\nbody") == "[Doc > Bits [31:0] config]"
    assert section_path("[Doc > Bits [31:0] config]") == "[Doc > Bits [31:0] config]"
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        store.add_chunks([
            _chunk("a", "[Spec > Regs > Bits [31:0]]\nconfig", section="Bits [31:0]"),
            _chunk("b", "[Spec > Other > Bits [31:0]]\nstatus", section="Bits [31:0]"),
        ])
        assert [c["id"] for c in store.get_section_chunks("a")] == ["a"]
        assert [c["id"] for c in store.get_section_chunks("b")] == ["b"]
        assert store.get_section_chunks("missing") == []
    finally:
        store.close()


def test_register_rows_are_not_duplicated_on_re_add(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        regs = {"peripheral": "RCC", "registers": [{"name": "RCC_CR"}]}
        store.add_chunks([_chunk("r", "t", structured=regs)])
        store.add_chunks([_chunk("r", "t", structured=regs)])
        assert store.conn.execute("SELECT COUNT(*) FROM registers").fetchone()[0] == 1
    finally:
        store.close()


def test_section_and_page_chunks(tmp_path):
    store = MetadataStore(tmp_path / "m.db")
    try:
        store.add_document("d1", "d1.pdf")
        a = _chunk("a", "one", section="S")
        b = _chunk("b", "two", section="S")
        c = _chunk("c", "three", section="T")
        c["page_start"] = c["page_end"] = 5
        store.add_chunks([a, b, c])
        assert [x["id"] for x in store.get_section_chunks("b")] == ["a", "b"]
        assert [x["id"] for x in store.get_page_chunks("d1", 4, 6)] == ["c"]
    finally:
        store.close()
