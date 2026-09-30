"""Query prefix and cross-encoder rerank, with stub models."""

import numpy as np

from mcp_embedded_docs.config import Config
from mcp_embedded_docs.indexing.embedder import QUERY_PREFIXES, LocalEmbedder
from mcp_embedded_docs.retrieval import SearchResponse, SearchResult
from mcp_embedded_docs.retrieval.hybrid_search import HybridSearch


def _embedder(model_name, query_prefix=None):
    # Skip __init__'s model load; only the prefix logic is under test.
    e = LocalEmbedder.__new__(LocalEmbedder)
    e.query_prefix = (QUERY_PREFIXES.get(model_name, "")
                      if query_prefix is None else query_prefix)
    seen = []
    e.embed_single = lambda text: seen.append(text)
    return e, seen


def test_query_prefix_defaults_per_model_and_config_overrides():
    e, seen = _embedder("BAAI/bge-small-en-v1.5")
    e.embed_query("RTC alarm")
    assert seen == ["Represent this sentence for searching relevant passages: RTC alarm"]

    e, seen = _embedder("some/unknown-model")
    e.embed_query("RTC alarm")
    assert seen == ["RTC alarm"]

    e, seen = _embedder("BAAI/bge-small-en-v1.5", query_prefix="")
    e.embed_query("RTC alarm")
    assert seen == ["RTC alarm"]


class _StubReranker:
    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.calls = scores or {}, error, []

    def score(self, query, passages):
        self.calls.append((query, passages))
        if self.error:
            raise self.error
        return [self.scores.get(p, 0.0) for p in passages]


def _result(text):
    return SearchResult(chunk_id=text, score=0.0, text=text, structured_data=None,
                        metadata={}, doc_id="d", page_start=0, page_end=0)


def _search(tmp_path, reranker=None, error=None):
    config = Config()
    config.index.directory = tmp_path
    config.embeddings.enabled = False
    config.search.rerank = True
    search = HybridSearch(config, load_embedder=False)
    search.reranker = reranker
    search._reranker_error = error
    search._reranker_ready.set()
    return search


def test_rerank_orders_by_score_and_keeps_fused_order_on_ties(tmp_path):
    stub = _StubReranker({"b": 2.0, "c": 1.0})
    search = _search(tmp_path, stub)
    response = SearchResponse(query="q", results=[])
    out = search._rerank("q", [_result(t) for t in "abcd"], response)
    assert [r.text for r in out] == ["b", "c", "a", "d"]
    assert out[0].score == 2.0
    assert stub.calls == [("q", ["a", "b", "c", "d"])]
    assert response.notes == []
    search.close()


def test_rerank_falls_back_to_fused_order(tmp_path):
    candidates = [_result(t) for t in "ab"]

    search = _search(tmp_path, error="OSError: no such model")
    response = SearchResponse(query="q", results=[])
    assert search._rerank("q", candidates, response) == candidates
    assert "failed to load" in response.notes[0]
    search.close()

    search = _search(tmp_path, _StubReranker(error=RuntimeError("boom")))
    response = SearchResponse(query="q", results=[])
    assert search._rerank("q", candidates, response) == candidates
    assert "boom" in response.notes[0]
    search.close()


def _index_with_model(tmp_path, model, dimension=3):
    from mcp_embedded_docs.indexing.vector_store import VectorStore

    vs = VectorStore(dimension=dimension, model=model)
    vs.add_vectors(np.eye(dimension, dtype=np.float32)[:1], ["d_1"])
    vs.save(tmp_path / "vectors.faiss")


def test_vectors_from_another_model_turn_semantic_search_off(tmp_path):
    _index_with_model(tmp_path, "other/model")
    config = Config()
    config.index.directory = tmp_path
    search = HybridSearch(config, load_embedder=False)
    assert search.vector_store is None
    assert "other/model" in search.semantic_status and "rebuild-vectors" in search.semantic_status
    search.close()


def test_unrecorded_model_with_wrong_dimension_is_caught_at_query(tmp_path):
    _index_with_model(tmp_path, None, dimension=3)
    config = Config()
    config.index.directory = tmp_path
    search = HybridSearch(config, load_embedder=False)
    stub = LocalEmbedder.__new__(LocalEmbedder)
    stub.model_name, stub.dimension = config.embeddings.model, 384
    stub.embed_query = lambda q: (_ for _ in ()).throw(AssertionError("must not embed"))
    search.embedder = stub
    search._embedder_ready.set()
    ids, status = search._semantic_search("q", 5, None)
    assert ids == [] and "dimension 3" in status
    search.close()
