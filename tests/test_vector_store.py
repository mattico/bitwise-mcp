"""VectorStore removal, dedupe, filtered search and atomic save."""

import numpy as np

from mcp_embedded_docs.indexing.vector_store import VectorStore


def _unit(rows):
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_remove_dedupe_and_prefix_search(tmp_path):
    vs = VectorStore(dimension=3)
    vs.add_vectors(_unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), ["a_1", "a_2", "b_1"])
    vs.add_vectors(_unit([[0.9, 0.1, 0]]), ["a_1"])  # re-ingest appended a duplicate

    assert vs.dedupe() == 1
    assert vs.ids == ["a_2", "b_1", "a_1"] and len(vs) == 3

    hits = vs.search(_unit([[1, 0, 0]])[0], 5)
    assert [h[0] for h in hits][0] == "a_1"
    assert len({h[0] for h in hits}) == len(hits)

    # prefix filtering scans the whole index, so a far-away match is still found
    hits = vs.search(_unit([[1, 0, 0]])[0], 1, id_prefix="b_")
    assert [h[0] for h in hits] == ["b_1"]

    assert vs.remove_ids({"a_1", "missing"}) == 1
    assert vs.ids == ["a_2", "b_1"] and len(vs) == 2

    path = tmp_path / "v.faiss"
    vs.save(path)
    assert not list(tmp_path.glob("*.tmp"))
    loaded = VectorStore()
    loaded.load(path)
    assert loaded.ids == ["a_2", "b_1"] and loaded.dimension == 3


def test_empty_store_search():
    assert VectorStore(dimension=3).search(np.ones(3, dtype=np.float32), 5) == []


def test_load_rejects_mismatched_pair_and_accepts_old_format(tmp_path):
    import pickle

    import faiss
    import pytest

    a = VectorStore(dimension=3)
    a.add_vectors(_unit([[1, 0, 0]]), ["a_1"])
    a.save(tmp_path / "a.faiss")
    b = VectorStore(dimension=3)
    b.add_vectors(_unit([[0, 1, 0]]), ["b_1"])
    b.save(tmp_path / "b.faiss")

    # new index next to another save's ids: same count, different content
    (tmp_path / "a.ids").replace(tmp_path / "keep.ids")
    (tmp_path / "b.ids").replace(tmp_path / "a.ids")
    with pytest.raises(ValueError, match="do not belong together"):
        VectorStore().load(tmp_path / "a.faiss", wait=0.2)

    # pre-0.4 layout: write_index + a bare pickled id list
    faiss.write_index(a.index, str(tmp_path / "old.faiss"))
    with open(tmp_path / "old.ids", "wb") as f:
        pickle.dump(["a_1"], f)
    old = VectorStore()
    old.load(tmp_path / "old.faiss")
    assert old.ids == ["a_1"] and len(old) == 1


def test_model_is_saved_and_old_files_have_none(tmp_path):
    vs = VectorStore(dimension=3, model="some/model")
    vs.add_vectors(_unit([[1, 0, 0]]), ["a_1"])
    vs.save(tmp_path / "v.faiss")
    loaded = VectorStore()
    loaded.load(tmp_path / "v.faiss")
    assert loaded.model == "some/model"

    VectorStore(dimension=3).save(tmp_path / "none.faiss")
    loaded.load(tmp_path / "none.faiss")
    assert loaded.model is None
