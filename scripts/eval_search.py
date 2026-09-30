"""Known-item retrieval eval: does search_docs rank the target section on top?

Each case names a target section (document filename prefix + SQL LIKE pattern on
the section title) and one or more queries for it. Reports hit@1, hit@5 and MRR@10
per query style, for hybrid and keyword-only search. Run it before and after a
ranking change, against the same index.

    uv run python scripts/eval_search.py [--config path/to/config.yaml] [--cases scripts/evalset_stm32h7.json]

--query-prefix and --rerank override the config, so variants compare on one index:

    uv run python scripts/eval_search.py --query-prefix ""
    uv run python scripts/eval_search.py --rerank cross-encoder/ms-marco-MiniLM-L6-v2

The bundled cases target the STM32H7 document set (reference manual, errata,
datasheet, Cortex-M7 reference, ULPI and USB334x docs).
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help="config.yaml (default: $BITWISE_MCP_CONFIG or ./config.yaml)")
    ap.add_argument("--cases", default=str(Path(__file__).with_name("evalset_stm32h7.json")))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--query-prefix", help='embeddings.query_prefix ("" for none)')
    ap.add_argument("--rerank", metavar="MODEL", help="enable search.rerank with this model")
    ap.add_argument("--rerank-depth", type=int, help="search.rerank_depth")
    args = ap.parse_args()
    if args.config:
        os.environ["BITWISE_MCP_CONFIG"] = args.config
    os.environ.setdefault("BITWISE_MCP_LOG", "0")

    from mcp_embedded_docs.config import Config
    from mcp_embedded_docs.retrieval.hybrid_search import HybridSearch

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    config = Config.load()
    if args.query_prefix is not None:
        config.embeddings.query_prefix = args.query_prefix
    if args.rerank:
        config.search.rerank = True
        config.search.rerank_model = args.rerank
    if args.rerank_depth:
        config.search.rerank_depth = args.rerank_depth
    con = sqlite3.connect(f"file:{config.index.directory / config.index.metadata_db}?mode=ro", uri=True)
    styles = [k for k in cases[0] if k not in ("doc", "section")]

    targets = []
    for c in cases:
        rows = con.execute(
            "SELECT DISTINCT c.doc_id, c.section_hierarchy FROM chunks c JOIN documents d "
            "ON d.id = c.doc_id WHERE d.filename LIKE ? AND c.section_hierarchy LIKE ?",
            (c["doc"] + "%", c["section"])).fetchall()
        if not rows:
            print(f"warning: no section matches {c['doc']!r} / {c['section']!r}; counted as a miss")
        targets.append({(d, s) for d, s in rows})

    for enabled in (True, False):
        config.embeddings.enabled = enabled
        search = HybridSearch(config)
        search.ensure_embedder()
        if config.search.rerank:
            search.ensure_reranker(wait=600)  # first use may download the model
        label = "hybrid" if enabled else "keyword"
        for style in styles:
            h1 = h5 = 0
            mrr = elapsed = 0.0
            for case, target in zip(cases, targets):
                t0 = time.perf_counter()
                results = search.search(case[style], 10)
                elapsed += time.perf_counter() - t0
                rank = next((i for i, r in enumerate(results) if (r.doc_id, r.section) in target), None)
                if rank is not None:
                    h1 += rank == 0
                    h5 += rank < args.top_k
                    mrr += 1 / (rank + 1)
            n = len(cases)
            print(f"{label:8} {style:11} hit@1 {h1 / n:4.0%}  hit@{args.top_k} {h5 / n:4.0%}  "
                  f"MRR@10 {mrr / n:.2f}  avg {elapsed / n * 1000:5.1f} ms")
        search.close()


if __name__ == "__main__":
    main()
