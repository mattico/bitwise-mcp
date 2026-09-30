"""Tests for the shared ingest/remove pipeline and the tools built on it.

These use a tiny generated PDF and a fake embedder, so no model is loaded.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import fitz
import numpy as np
import pytest

from mcp_embedded_docs.config import Config
from mcp_embedded_docs.indexing.vector_store import VectorStore
from mcp_embedded_docs.ingestion.pipeline import doc_id_for, ingest_pdf, remove_document
from mcp_embedded_docs.tools.ingest_docs import ingest_docs
from mcp_embedded_docs.tools.remove_docs import remove_docs


class FakeEmbedder:
    dimension = 16

    def __init__(self):
        self.calls = 0
        self.texts = 0
        self._rng = np.random.default_rng(0)

    def embed_batch(self, texts):
        self.calls += 1
        self.texts += len(texts)
        v = self._rng.standard_normal((len(texts), self.dimension)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_pdf(path: Path, sections: list[tuple[str, str]]) -> Path:
    """Write a PDF with one page and one TOC entry per (title, body)."""
    doc = fitz.open()
    toc = []
    for i, (title, body) in enumerate(sections):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 550, 800), f"{title}\n{body}", fontsize=10
        )
        toc.append([1, title, i + 1])
    doc.set_toc(toc)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


SECTIONS_V1 = [
    ("1 Overview", "The widget controller drives the frobnicator. " * 5),
    ("2 Clocking", "The PLL multiplies the reference clock by N. " * 5),
    ("3 Reset", "Asserting nRESET clears every register to its default. " * 5),
]
SECTIONS_V2 = [
    ("1 Overview", "The widget controller drives the frobnicator. " * 5),
    ("2 Clocking", "Revision B: the PLL now divides the reference clock by M. " * 5),
]


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.index.directory = tmp_path / "index"
    cfg.doc_dirs = [tmp_path / "docs"]
    return cfg


def db_counts(cfg: Config, doc_id: str) -> tuple[int, int]:
    con = sqlite3.connect(cfg.index.directory / cfg.index.metadata_db)
    try:
        chunks = con.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
        docs = con.execute(
            "SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()[0]
    finally:
        con.close()
    return chunks, docs


def load_vectors(cfg: Config) -> VectorStore:
    store = VectorStore()
    store.load(cfg.index.directory / cfg.index.vector_file)
    return store


def test_ingest_writes_chunks_and_vectors(tmp_path, config):
    pdf = make_pdf(tmp_path / "docs" / "widget.pdf", SECTIONS_V1)
    embedder = FakeEmbedder()
    lines: list[str] = []

    report = ingest_pdf(
        pdf, config, title="Widget", embedder=embedder, progress=lines.append
    )

    assert report.doc_id == doc_id_for(pdf)
    assert report.pages == 3
    assert report.chunks > 0
    assert report.vectors == report.chunks
    assert report.replaced_chunks == 0
    assert set(report.timings) >= {"parse", "tables", "chunk", "embed", "store"}
    assert any("Parsing PDF" in line for line in lines)

    chunks, docs = db_counts(config, report.doc_id)
    assert (chunks, docs) == (report.chunks, 1)
    vectors = load_vectors(config)
    assert vectors.size == report.chunks
    assert sorted(vectors.ids) == sorted(_chunk_ids(config))

    con = sqlite3.connect(config.index.directory / config.index.metadata_db)
    path = con.execute("SELECT path FROM documents").fetchone()[0]
    con.close()
    assert Path(path) == pdf.resolve()


def test_reingest_replaces_previous_version(tmp_path, config):
    pdf = make_pdf(tmp_path / "docs" / "widget.pdf", SECTIONS_V1)
    other = make_pdf(
        tmp_path / "docs" / "other.pdf", [("1 Intro", "Unrelated gadget text. " * 5)]
    )
    embedder = FakeEmbedder()

    other_report = ingest_pdf(other, config, embedder=embedder)
    first = ingest_pdf(pdf, config, embedder=embedder)
    again = ingest_pdf(pdf, config, embedder=embedder)

    # Same content: same chunk ids, no duplicates anywhere.
    assert again.replaced_chunks == first.chunks
    assert again.chunks == first.chunks
    assert db_counts(config, first.doc_id) == (first.chunks, 1)
    vectors = load_vectors(config)
    assert vectors.size == first.chunks + other_report.chunks
    assert len(set(vectors.ids)) == vectors.size

    # Changed content: the old chunks and their vectors are gone.
    make_pdf(pdf, SECTIONS_V2)
    v2 = ingest_pdf(pdf, config, embedder=embedder)
    assert v2.replaced_chunks == first.chunks
    assert db_counts(config, v2.doc_id) == (v2.chunks, 1)
    vectors = load_vectors(config)
    assert vectors.size == v2.chunks + other_report.chunks
    assert sorted(vectors.ids) == sorted(_chunk_ids(config))
    texts = " ".join(_chunk_texts(config))
    assert "Asserting nRESET" not in texts
    assert "Revision B" in texts


def test_remove_document_deletes_rows_and_vectors_without_torch(tmp_path, config):
    pdf = make_pdf(tmp_path / "docs" / "widget.pdf", SECTIONS_V1)
    other = make_pdf(
        tmp_path / "docs" / "other.pdf", [("1 Intro", "Unrelated gadget text. " * 5)]
    )
    embedder = FakeEmbedder()
    report = ingest_pdf(pdf, config, embedder=embedder)
    other_report = ingest_pdf(other, config, embedder=embedder)

    for name in [m for m in sys.modules if m.startswith("sentence_transformers")]:
        del sys.modules[name]

    removed = remove_document(report.doc_id, config)

    assert "sentence_transformers" not in sys.modules
    assert removed is not None
    assert removed.filename == "widget.pdf"
    assert removed.chunks == report.chunks
    assert removed.vectors == report.chunks
    assert db_counts(config, report.doc_id) == (0, 0)
    vectors = load_vectors(config)
    assert vectors.size == other_report.chunks
    assert all(cid.startswith(other_report.doc_id) for cid in vectors.ids)

    assert remove_document(report.doc_id, config) is None
    assert remove_document("nope", config) is None


def test_remove_document_without_index_returns_none(config):
    assert remove_document("abc", config) is None
    assert not (config.index.directory / config.index.metadata_db).exists()


def test_embeddings_disabled_writes_no_vectors(tmp_path, config, monkeypatch):
    config.embeddings.enabled = False
    pdf = make_pdf(tmp_path / "docs" / "widget.pdf", SECTIONS_V1)

    class Exploding:
        dimension = 16

        def embed_batch(self, texts):
            raise AssertionError("embedder must not be called")

    monkeypatch.setitem(sys.modules, "mcp_embedded_docs.indexing.embedder", None)

    report = ingest_pdf(pdf, config, embedder=Exploding())
    assert report.vectors == 0
    assert report.chunks > 0
    assert db_counts(config, report.doc_id) == (report.chunks, 1)
    assert not (config.index.directory / config.index.vector_file).exists()

    # Without an embedder passed in, the model is never imported either
    # (importing the blocked module above would raise).
    again = ingest_pdf(pdf, config)
    assert again.replaced_chunks == report.chunks
    assert not (config.index.directory / config.index.vector_file).exists()


def test_ingest_docs_resolves_bare_filename(tmp_path, config):
    make_pdf(tmp_path / "docs" / "sub" / "Widget_RM.pdf", SECTIONS_V1)
    embedder = FakeEmbedder()

    out = ingest_docs(
        "widget_rm.pdf", config=config, embedder=embedder, detect_tables=False
    )

    assert "Successfully indexed Widget_RM.pdf" in out
    assert doc_id_for(Path("Widget_RM.pdf")) in out
    assert embedder.calls > 0

    out = ingest_docs("sub/Widget_RM.pdf", config=config, embedder=embedder)
    assert "Replaced chunks" in out and "previous version removed" in out

    out = ingest_docs("Widget_RM", config=config, embedder=embedder)
    assert "Successfully indexed" in out


def test_ingest_docs_errors(tmp_path, config):
    assert "not found" in ingest_docs(
        "missing.pdf", config=config, embedder=FakeEmbedder()
    )
    txt = tmp_path / "docs" / "notes.txt"
    txt.parent.mkdir(parents=True)
    txt.write_text("hi")
    assert "only PDF files" in ingest_docs(
        str(txt), config=config, embedder=FakeEmbedder()
    )


def test_remove_docs_tool(tmp_path, config):
    pdf = make_pdf(tmp_path / "docs" / "widget.pdf", SECTIONS_V1)
    report = ingest_pdf(pdf, config, embedder=FakeEmbedder())

    assert "not found" in remove_docs("nope", config=config)
    out = remove_docs(report.doc_id, config=config)
    assert "Removed widget.pdf" in out
    assert f"{report.chunks} chunks, {report.chunks} vectors" in out

    ingest_pdf(pdf, config, embedder=FakeEmbedder())
    assert "Removed widget.pdf" in remove_docs("WIDGET.pdf", config=config)


def _chunk_ids(cfg: Config) -> list[str]:
    con = sqlite3.connect(cfg.index.directory / cfg.index.metadata_db)
    try:
        return [r[0] for r in con.execute("SELECT id FROM chunks")]
    finally:
        con.close()


def _chunk_texts(cfg: Config) -> list[str]:
    con = sqlite3.connect(cfg.index.directory / cfg.index.metadata_db)
    try:
        return [r[0] for r in con.execute("SELECT text FROM chunks")]
    finally:
        con.close()
