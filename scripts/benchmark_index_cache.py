import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.benchmark_corpus import load_benchmark_corpus
from src.chunker import chunk_text
from src.embedding import EMBEDDING_MODEL, _get_model, embed_text, embed_texts
from src.indexer import load_or_create_index
from src.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_MANIFEST = ROOT / "data" / "benchmark_corpus_v1.json"
DEFAULT_DATASET_FILE = ROOT / "data" / "retrieval_evaluation.json"
REPORT_FILE = ROOT / "reports" / "index_cache_benchmark_v3.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
TARGET_CHUNK_COUNTS = (100, 1000)


def build_store(index):
    chunks, embeddings, metadata = index
    store = VectorStore()
    for chunk, embedding, item_metadata in zip(chunks, embeddings, metadata):
        store.add(chunk, embedding, item_metadata)
    return store


def distinct_chunks(documents):
    return {
        chunk
        for document in documents
        for chunk in chunk_text(
            document["text"],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    }


def select_natural_subset(documents, target_count):
    by_source = {}
    for document in documents:
        by_source.setdefault(document["source"], []).append(document)

    selected = []
    for source_documents in by_source.values():
        selected.extend(source_documents)
        if len(distinct_chunks(selected)) >= target_count:
            return selected
    return None


def assert_equivalent(index, reference, query_embeddings):
    chunks, embeddings, metadata = index
    reference_chunks, reference_embeddings, reference_metadata = reference
    if chunks != reference_chunks or metadata != reference_metadata:
        raise RuntimeError("Rebuilt/cached index chunks or metadata differ.")
    if not np.array_equal(embeddings, reference_embeddings):
        raise RuntimeError("Rebuilt/cached index embeddings differ.")

    actual_store = build_store(index)
    reference_store = build_store(reference)
    for question, query_embedding in query_embeddings.items():
        actual = actual_store.search(query_embedding, top_k=TOP_K)
        expected = reference_store.search(query_embedding, top_k=TOP_K)
        actual_ids = [
            (
                item["metadata"].get("source"),
                item["metadata"].get("page"),
                item["metadata"].get("chunk_id"),
            )
            for item in actual
        ]
        expected_ids = [
            (
                item["metadata"].get("source"),
                item["metadata"].get("page"),
                item["metadata"].get("chunk_id"),
            )
            for item in expected
        ]
        if actual_ids != expected_ids or not np.allclose(
            [item["score"] for item in actual],
            [item["score"] for item in expected],
            rtol=0,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"Rebuilt/cached retrieval rankings differ for {question!r}."
            )


def hardware_summary():
    cpu_model = platform.processor()
    if platform.system() == "Darwin":
        cpu_model = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "python": sys.version.split()[0],
    }


def measure_index_preparation(documents, cache_path, repetitions, reference, queries):
    timings = {"rebuild": [], "cache_load": []}
    measurements = []
    order = ["rebuild", "cache_load"] * repetitions
    random.Random(20261019).shuffle(order)

    for run_order, condition in enumerate(order, start=1):
        if condition == "rebuild":
            cache_path.unlink(missing_ok=True)
        started = perf_counter()
        index = load_or_create_index(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            cache_path=cache_path,
        )
        elapsed = perf_counter() - started
        assert_equivalent(index, reference, queries)
        timings[condition].append(elapsed)
        measurements.append(
            {"run_order": run_order, "condition": condition, "seconds": elapsed}
        )

    return timings, measurements


def summarize(timings):
    rebuild = timings["rebuild"]
    cached = timings["cache_load"]
    rebuild_median = statistics.median(rebuild)
    cache_median = statistics.median(cached)
    saved_seconds = rebuild_median - cache_median
    return {
        "rebuild_median_seconds": rebuild_median,
        "cache_load_median_seconds": cache_median,
        "absolute_milliseconds_saved": saved_seconds * 1000,
        "reduction_percent": (
            saved_seconds / rebuild_median * 100 if rebuild_median else 0
        ),
        "rebuild_variability": {
            "sample_standard_deviation_seconds": statistics.stdev(rebuild),
            "minimum_seconds": min(rebuild),
            "maximum_seconds": max(rebuild),
        },
        "cache_load_variability": {
            "sample_standard_deviation_seconds": statistics.stdev(cached),
            "minimum_seconds": min(cached),
            "maximum_seconds": max(cached),
        },
    }


def build_scale_subsets(documents):
    subsets = [
        {
            "subset_id": "full-selected-corpus",
            "documents": documents,
            "requested_targets": [],
        }
    ]
    unavailable_targets = []
    for target in TARGET_CHUNK_COUNTS:
        selection = select_natural_subset(documents, target)
        if selection is None:
            unavailable_targets.append(
                {
                    "target_distinct_chunks": target,
                    "available_distinct_chunks": len(distinct_chunks(documents)),
                    "status": "unavailable-in-selected-corpus",
                }
            )
            continue

        selected_sources = tuple(
            dict.fromkeys(document["source"] for document in selection)
        )
        existing = next(
            (
                subset
                for subset in subsets
                if tuple(dict.fromkeys(
                    document["source"] for document in subset["documents"]
                )) == selected_sources
            ),
            None,
        )
        if existing:
            existing["requested_targets"].append(target)
        else:
            subsets.append(
                {
                    "subset_id": f"at-least-{target}-distinct-chunks",
                    "documents": selection,
                    "requested_targets": [target],
                }
            )
    return subsets, unavailable_targets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare warm-model index rebuild and valid-cache load on naturally "
            "sized corpus subsets."
        )
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Randomized repetitions per condition; minimum 10 (default: 10).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_FILE,
        help=f"Path for raw JSON results (default: {REPORT_FILE.name}).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="Versioned evaluation-question JSON file for retrieval-equivalence checks.",
    )
    corpus_group = parser.add_mutually_exclusive_group()
    corpus_group.add_argument(
        "--documents-dir",
        type=Path,
        help="Index every supported document under this directory.",
    )
    corpus_group.add_argument(
        "--corpus-manifest",
        type=Path,
        help="JSON manifest explicitly listing every document to index.",
    )
    args = parser.parse_args()
    if args.repetitions < 10:
        parser.error("--repetitions must be at least 10.")

    try:
        dataset_path = args.dataset.expanduser().resolve()
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        answerable = dataset.get("answerable", [])
        unanswerable = dataset.get("unanswerable", [])
        questions = [
            case["question"] for case in [*answerable, *unanswerable]
        ]
        if not questions:
            raise ValueError("Dataset must include at least one question.")

        documents, corpus_info = load_benchmark_corpus(
            documents_dir=args.documents_dir,
            corpus_manifest=(
                args.corpus_manifest
                or (None if args.documents_dir else DEFAULT_CORPUS_MANIFEST)
            ),
        )
        subsets, unavailable_targets = build_scale_subsets(documents)

        model_start = perf_counter()
        _get_model()
        model_startup_seconds = perf_counter() - model_start
        embed_texts(["index benchmark model warm-up"])
        query_embeddings = {
            question: embed_text(question) for question in questions
        }

        subset_results = []
        with tempfile.TemporaryDirectory(
            prefix="rag-index-cache-benchmark-"
        ) as temp_dir:
            temp_root = Path(temp_dir)
            for subset in subsets:
                selected_documents = subset["documents"]
                index_chunks = distinct_chunks(selected_documents)
                cache_path = temp_root / f"{subset['subset_id']}.npz"
                reference = load_or_create_index(
                    selected_documents,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    cache_path=cache_path,
                )
                timings, measurements = measure_index_preparation(
                    selected_documents,
                    cache_path,
                    args.repetitions,
                    reference,
                    query_embeddings,
                )
                subset_results.append(
                    {
                        "subset_id": subset["subset_id"],
                        "requested_targets": subset["requested_targets"],
                        "corpus": {
                            "document_count": len(
                                {document["source"] for document in selected_documents}
                            ),
                            "page_or_text_records": len(selected_documents),
                            "indexed_chunks": len(reference[0]),
                            "distinct_chunk_texts": len(index_chunks),
                            "manifest": [
                                item
                                for item in corpus_info["manifest"]
                                if item["source"]
                                in {
                                    document["source"]
                                    for document in selected_documents
                                }
                            ],
                        },
                        "raw_seconds": timings,
                        "raw_measurements_in_run_order": measurements,
                        "summary": summarize(timings),
                        "retrieval_equivalent": True,
                    }
                )

        report = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_version": "4.0",
            "dataset_version": dataset.get("dataset_version"),
            "dataset_path": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "dataset_counts": {
                "answerable": len(answerable),
                "unanswerable": len(unanswerable),
                "total_questions": len(questions),
                "review_statuses": sorted(
                    {
                        case.get("review_status", "missing")
                        for case in [*answerable, *unanswerable]
                    }
                ),
            },
            "requirements_sha256": hashlib.sha256(
                (ROOT / "requirements.txt").read_bytes()
            ).hexdigest(),
            "package_versions": {
                name: version(name)
                for name in ("sentence-transformers", "numpy", "torch", "pypdf")
            },
            "embedding_model": EMBEDDING_MODEL,
            "configuration": {
                "chunk_size_characters": CHUNK_SIZE,
                "chunk_overlap_characters": CHUNK_OVERLAP,
                "repetitions_per_condition": args.repetitions,
                "retrieval_equivalence_top_k": TOP_K,
                "randomized_condition_order_seed": 20261019,
                "requested_natural_subset_sizes": list(TARGET_CHUNK_COUNTS),
            },
            "corpus": corpus_info,
            "method": {
                "model_loaded_before_timing": True,
                "model_warmup_excluded": True,
                "model_startup_includes": "SentenceTransformer construction",
                "timed_operation": (
                    "load_or_create_index: cache-key hashing; on rebuild, chunking, "
                    "embedding and cache serialization/write; on cache hit, cache "
                    "read and deserialization."
                ),
                "excluded_stages": [
                    "Document discovery and parsing (performed once before timing)",
                    "VectorStore construction",
                    "Query embedding and retrieval search",
                    "LLM generation",
                ],
                "timing_scope": "index preparation; not end-to-end answer latency",
                "retrieval_equivalence": (
                    "Exact chunk/metadata/embedding equality and matching top-five "
                    "rankings and scores for every answerable and unanswerable "
                    "evaluation question."
                ),
            },
            "model_startup_seconds_separate": model_startup_seconds,
            "unavailable_target_sizes": unavailable_targets,
            "subsets": subset_results,
            "hardware": hardware_summary(),
        }
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Could not benchmark index cache: {error}")

    for subset in subset_results:
        summary = subset["summary"]
        print(
            f"{subset['subset_id']}: "
            f"{subset['corpus']['document_count']} documents, "
            f"{subset['corpus']['distinct_chunk_texts']} distinct chunks; "
            f"rebuild {summary['rebuild_median_seconds'] * 1000:.3f} ms, "
            f"cache {summary['cache_load_median_seconds'] * 1000:.3f} ms, "
            f"saved {summary['absolute_milliseconds_saved']:.3f} ms "
            f"({summary['reduction_percent']:.1f}%)."
        )
    print(f"Unavailable requested sizes: {unavailable_targets}")
    print(f"Model startup (separate): {model_startup_seconds:.3f}s")
    print(f"Raw results: {report_path}")


if __name__ == "__main__":
    main()
