from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import DATA_DIR, KnowledgeBase


def evaluate(kb: KnowledgeBase, dataset: Path, top_k: int = 3) -> dict:
    cases = json.loads(dataset.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        start = time.perf_counter()
        evidence, supported = kb.retrieve(case["question"], top_k)
        elapsed = (time.perf_counter() - start) * 1000
        expected = case.get("expected_source")
        retrieved = [e.source for e in evidence]
        rank = retrieved.index(expected) + 1 if expected in retrieved else None
        rows.append({"question": case["question"], "expected_source": expected, "sources": retrieved, "supported": supported, "hit_at_k": bool(rank), "reciprocal_rank": 1 / rank if rank else 0.0, "abstention_correct": supported == bool(expected), "retrieval_ms": round(elapsed, 3)})
    answerable = [r for r in rows if r["expected_source"]]
    latencies = [r["retrieval_ms"] for r in rows]
    return {
        "dataset": str(dataset), "cases": len(rows), "answerable_cases": len(answerable), "top_k": top_k,
        "embedding": kb.embedding_name, "reranker": kb.reranker_name,
        "hit_at_k": round(sum(r["hit_at_k"] for r in answerable) / len(answerable), 4) if answerable else None,
        "mrr_at_k": round(sum(r["reciprocal_rank"] for r in answerable) / len(answerable), 4) if answerable else None,
        "abstention_accuracy": round(sum(r["abstention_correct"] for r in rows) / len(rows), 4) if rows else None,
        "retrieval_mean_ms": round(float(np.mean(latencies)), 3) if rows else None,
        "retrieval_p95_ms": round(float(np.percentile(latencies, 95)), 3) if rows else None,
        "scope": "Synthetic fixture retrieval only; excludes indexing, model download and API generation.",
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Reproducible retrieval and evidence demonstration")
    commands = parser.add_subparsers(dest="command", required=True)
    ask_parser = commands.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", choices=["offline", "deepseek"], default="offline")
    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--dataset", type=Path, default=DATA_DIR / "eval.json")
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument("--top-k", type=int, choices=range(1, 9), default=3)
    commands.add_parser("index")
    args = parser.parse_args()
    started = time.perf_counter()
    kb = KnowledgeBase.from_env()
    if args.command == "ask":
        result = kb.ask(args.question, mode=args.mode)
    elif args.command == "eval":
        result = evaluate(kb, args.dataset, args.top_k)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        result = {"documents": len({c.source for c in kb.chunks}), "chunks": len(kb.chunks), "index_ms": round((time.perf_counter() - started) * 1000, 3), "embedding": kb.embedding_name, "reranker": kb.reranker_name}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
