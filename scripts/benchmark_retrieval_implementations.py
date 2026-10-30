import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.benchmark_corpus import load_benchmark_corpus
from scripts.evaluate_retrieval import (
    _hardware_summary,
    _unanswerable_statuses,
    validate_dataset,
)
from src.embedding import EMBEDDING_MODEL, _get_model, embed_text
from src.evaluation import evaluate
from src.hybrid_retrieval import BM25Index, bm25_search, reciprocal_rank_fusion
from src.indexer import load_or_create_index
from src.retriever import Retriever
from src.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "benchmark_corpus_v2.json"
DEFAULT_DATASET = ROOT / "data" / "retrieval_evaluation_v4.json"
DEFAULT_REPORT = ROOT / "reports" / "retrieval_performance_v3.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RANKING_TOP_K = 10
METHODS = (
    "original_dense",
    "original_hybrid",
    "optimized_dense",
    "optimized_hybrid",
)


_REFERENCE_STORE = VectorStore()


def original_dense_search(query_embedding, chunks, embeddings, metadata, top_k):
    results = []
    for chunk, embedding, item_metadata in zip(chunks, embeddings, metadata):
        score = _REFERENCE_STORE.cosine_similarity(query_embedding, embedding)
        results.append(
            {"score": score, "text": chunk, "metadata": item_metadata}
        )
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


def run_original_hybrid(query, query_embedding, chunks, embeddings, metadata, top_k):
    dense = original_dense_search(
        query_embedding, chunks, embeddings, metadata, len(chunks)
    )
    lexical = bm25_search(query, chunks, metadata, len(chunks))
    return reciprocal_rank_fusion(dense, lexical, top_k)


def result_identity(results):
    return [
        (
            item["metadata"].get("source"),
            item["metadata"].get("page"),
            item["metadata"].get("chunk_id"),
        )
        for item in results
    ]


def summarize(samples):
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_nearest_rank_ms": ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)]
        * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
        "minimum_ms": min(samples) * 1000,
        "maximum_ms": max(samples) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original and optimized dense/hybrid retrieval on the same "
            "corpus and query set."
        )
    )
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.repetitions < 10:
        parser.error("--repetitions must be at least 10")

    try:
        dataset_path = args.dataset.expanduser().resolve()
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        documents, corpus_info = load_benchmark_corpus(
            corpus_manifest=args.corpus_manifest
        )
        cases, statuses = validate_dataset(dataset, documents)
        unanswerable, unanswerable_statuses = _unanswerable_statuses(dataset)
        if statuses | unanswerable_statuses != {"pending"}:
            raise ValueError(
                "This provisional performance run expects all labels pending."
            )

        cache_path = ROOT / "data" / ".rag_cache" / "performance" / "v3.npz"
        chunks, embeddings, metadata = load_or_create_index(
            documents,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            cache_path,
        )

        optimized_store = VectorStore()
        for chunk, embedding, item_metadata in zip(chunks, embeddings, metadata):
            optimized_store.add(chunk, embedding, item_metadata)
        optimized_dense_init_started = perf_counter()
        optimized_store.prepare_dense_index()
        optimized_dense_index_build_seconds = (
            perf_counter() - optimized_dense_init_started
        )

        bm25_build_started = perf_counter()
        optimized_bm25_index = BM25Index(chunks, metadata)
        bm25_index_build_seconds = perf_counter() - bm25_build_started
        optimized_retriever = Retriever(
            optimized_store, chunks, metadata, optimized_bm25_index
        )

        _get_model()
        embed_text("warmup query")
        # Warm all four paths, including first-use runtime and BLAS setup.
        warm_question = cases[0]["question"]
        warm_embedding = embed_text(warm_question)
        original_dense_search(
            warm_embedding, chunks, embeddings, metadata, RANKING_TOP_K
        )
        run_original_hybrid(
            warm_question, warm_embedding, chunks, embeddings, metadata,
            RANKING_TOP_K,
        )
        optimized_retriever.search_with_embedding(
            warm_question, warm_embedding, RANKING_TOP_K, "dense"
        )
        optimized_retriever.search_with_embedding(
            warm_question, warm_embedding, RANKING_TOP_K, "hybrid"
        )

        raw = {
            method: {
                "query_embedding_seconds": [],
                "search_seconds": [],
                "total_seconds": [],
                "per_query": [],
            }
            for method in METHODS
        }
        quality_results = {
            method: {"development": None, "held-out-test": None}
            for method in METHODS
        }
        all_results = {method: {} for method in METHODS}
        for case in cases:
            query = case["question"]
            raw_vectors = []
            raw_embedding_seconds = []
            for _ in range(args.repetitions):
                started = perf_counter()
                query_embedding = embed_text(query)
                raw_vectors.append(query_embedding)
                raw_embedding_seconds.append(perf_counter() - started)

            for repetition, query_embedding in enumerate(raw_vectors):
                method_order = list(METHODS)
                random.Random(20261019 + repetition * 1009 + len(query)).shuffle(
                    method_order
                )
                embedding_seconds = raw_embedding_seconds[repetition]
                for method in method_order:
                    started = perf_counter()
                    if method == "original_dense":
                        results = original_dense_search(
                            query_embedding, chunks, embeddings, metadata,
                            RANKING_TOP_K,
                        )
                    elif method == "original_hybrid":
                        results = run_original_hybrid(
                            query, query_embedding, chunks, embeddings, metadata,
                            RANKING_TOP_K,
                        )
                    elif method == "optimized_dense":
                        results = optimized_retriever.search_with_embedding(
                            query, query_embedding, RANKING_TOP_K, "dense"
                        )
                    else:
                        results = optimized_retriever.search_with_embedding(
                            query, query_embedding, RANKING_TOP_K, "hybrid"
                        )
                    search_seconds = perf_counter() - started
                    raw[method]["query_embedding_seconds"].append(
                        embedding_seconds
                    )
                    raw[method]["search_seconds"].append(search_seconds)
                    raw[method]["total_seconds"].append(
                        embedding_seconds + search_seconds
                    )
                    raw[method]["per_query"].append(
                        {
                            "id": case["id"],
                            "split": case["split"],
                            "repetition": repetition + 1,
                            "method_order": method_order,
                            "query_embedding_ms": embedding_seconds * 1000,
                            "search_ms": search_seconds * 1000,
                            "total_ms": (
                                embedding_seconds + search_seconds
                            )
                            * 1000,
                        }
                    )
                    all_results[method].setdefault(case["id"], []).append(results)

        equivalence = {}
        for original, optimized in (
            ("original_dense", "optimized_dense"),
            ("original_hybrid", "optimized_hybrid"),
        ):
            same_ranking = True
            maximum_score_difference = 0.0
            for case in cases:
                original_runs = all_results[original][case["id"]]
                optimized_runs = all_results[optimized][case["id"]]
                for expected, actual in zip(original_runs, optimized_runs):
                    if result_identity(expected) != result_identity(actual):
                        same_ranking = False
                    difference = max(
                        (
                            abs(float(left["score"]) - float(right["score"]))
                            for left, right in zip(expected, actual)
                        ),
                        default=0.0,
                    )
                    maximum_score_difference = max(
                        maximum_score_difference, difference
                    )
            equivalence[f"{original}_vs_{optimized}"] = {
                "identical_top_10_ranking": same_ranking,
                "maximum_absolute_score_difference": maximum_score_difference,
                "score_tolerance": 1e-6,
                "equivalent": same_ranking and maximum_score_difference <= 1e-6,
            }

        for case in cases:
            for method in METHODS:
                # Each method's final-pass rankings are evaluated; timing repetitions
                # are not treated as independent labeled questions.
                all_results[method][case["id"]] = all_results[method][case["id"]][-1]
        for split in ("development", "held-out-test"):
            split_cases = [case for case in cases if case["split"] == split]
            for method in METHODS:
                quality_results[method][split] = evaluate(
                    split_cases,
                    lambda question, result_map=all_results[method]: (
                        result_map[
                            next(
                                item["id"]
                                for item in split_cases
                                if item["question"] == question
                            )
                        ]
                    ),
                )

        report = {
            "benchmark_version": "2.0",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset.get("dataset_version"),
            "dataset_path": str(dataset_path),
            "dataset_sha256": hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest(),
            "label_review_status": sorted(statuses | unanswerable_statuses),
            "provisional": True,
            "embedding_model": EMBEDDING_MODEL,
            "corpus": corpus_info,
            "chunk_counts": {
                "indexed": len(chunks),
                "distinct_text": len(set(chunks)),
            },
            "question_counts": {
                "answerable": len(cases),
                "development": sum(
                    case["split"] == "development" for case in cases
                ),
                "held_out_test": sum(
                    case["split"] == "held-out-test" for case in cases
                ),
                "unanswerable_not_scored": len(unanswerable),
            },
            "configuration": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": RANKING_TOP_K,
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
                "rrf_constant": 60,
                "candidate_scope": "all indexed chunks",
                "repetitions": args.repetitions,
                "randomized_method_order_seed": 20261019,
                "timed_unit": (
                    "query-embedding time is measured once per query/repetition "
                    "and reused across the four methods; total is that embedding "
                    "duration plus each method's search duration."
                ),
            },
            "index_initialization_seconds_separate": {
                "optimized_dense_preparation": optimized_dense_index_build_seconds,
                "optimized_bm25_precomputation": bm25_index_build_seconds,
            },
            "equivalence": equivalence,
            "latency": {
                method: {
                    "query_embedding": summarize(
                        raw[method]["query_embedding_seconds"]
                    ),
                    "search": summarize(raw[method]["search_seconds"]),
                    "total_retrieval": summarize(raw[method]["total_seconds"]),
                    "raw": raw[method],
                }
                for method in METHODS
            },
            "quality": quality_results,
            "hardware": _hardware_summary(),
            "definition": {
                "original_dense": (
                    "Per-document scalar cosine loop and stable Python list sort; "
                    "all methods return the top ten for quality cutoffs."
                ),
                "original_hybrid": (
                    "Original scalar dense ranking, original per-query BM25 "
                    "tokenization/document-frequency calculation, and RRF."
                ),
                "optimized_dense": (
                    "Pre-normalized document matrix, vectorized dot product, "
                    "stable NumPy sort."
                ),
                "optimized_hybrid": (
                    "Optimized dense matrix ranking and reusable precomputed "
                    "BM25 inverted index, combined with the same RRF."
                ),
                "unanswerable": (
                    "Not scored by retrieval quality; answer generation and "
                    "abstention require separate evaluation."
                ),
            },
        }
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Could not benchmark retrieval implementations: {error}")

    print(f"BM25 index build: {bm25_index_build_seconds * 1000:.3f} ms")
    for method in METHODS:
        latency = report["latency"][method]
        print(
            f"{method}: search median/p95="
            f"{latency['search']['median_ms']:.3f}/"
            f"{latency['search']['p95_nearest_rank_ms']:.3f} ms; "
            f"embedding median={latency['query_embedding']['median_ms']:.3f} ms; "
            f"total median/p95="
            f"{latency['total_retrieval']['median_ms']:.3f}/"
            f"{latency['total_retrieval']['p95_nearest_rank_ms']:.3f} ms"
        )
    print(f"Equivalence: {equivalence}")
    print(f"Raw report: {report_path}")


if __name__ == "__main__":
    main()
