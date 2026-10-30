import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from src.embedding import EMBEDDING_MODEL, embed_text
from src.evaluation import evaluate
from src.indexer import load_or_create_index
from src.retriever import build_retriever

from scripts.benchmark_corpus import load_benchmark_corpus


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_FILE = ROOT / "data" / "retrieval_evaluation.json"
DEFAULT_CORPUS_MANIFEST = ROOT / "data" / "benchmark_corpus_v1.json"
REPORT_FILE = ROOT / "reports" / "retrieval_benchmark_v4.json"
BASELINE_NAME = "current-500-char-100-overlap"
CONFIGURATIONS = (
    {
        "name": BASELINE_NAME,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "retrieval_method": "dense",
    },
    {
        "name": "whole-document-dense",
        "chunk_size": None,
        "chunk_overlap": 0,
        "retrieval_method": "dense",
    },
    {
        "name": "250-char-no-overlap-dense",
        "chunk_size": 250,
        "chunk_overlap": 0,
        "retrieval_method": "dense",
    },
    {
        "name": "250-char-50-overlap-dense",
        "chunk_size": 250,
        "chunk_overlap": 50,
        "retrieval_method": "dense",
    },
    {
        "name": "500-char-no-overlap-dense",
        "chunk_size": 500,
        "chunk_overlap": 0,
        "retrieval_method": "dense",
    },
    {
        "name": "750-char-100-overlap-dense",
        "chunk_size": 750,
        "chunk_overlap": 100,
        "retrieval_method": "dense",
    },
    {
        "name": "current-500-char-100-overlap-dense-bm25-rrf",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "retrieval_method": "dense_bm25_rrf",
    },
    {
        "name": "500-char-no-overlap-dense-bm25-rrf",
        "chunk_size": 500,
        "chunk_overlap": 0,
        "retrieval_method": "dense_bm25_rrf",
    },
)


def validate_dataset(dataset, documents):
    cases = dataset["answerable"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must contain a non-empty answerable list.")

    document_by_location = {
        (document["source"], document.get("page")): document["text"]
        for document in documents
    }
    case_ids = set()
    normalized_questions = set()
    evidence_ids = set()
    group_splits = {}
    review_statuses = set()

    for case in cases:
        if not case.get("id") or case["id"] in case_ids:
            raise ValueError(f"Missing or duplicate evaluation case ID: {case.get('id')!r}")
        case_ids.add(case["id"])
        normalized_question = " ".join(case["question"].casefold().split())
        if normalized_question in normalized_questions:
            raise ValueError(f"Duplicate evaluation question: {case['question']!r}")
        normalized_questions.add(normalized_question)
        if case.get("split") not in {"development", "held-out-test"}:
            raise ValueError(f"Invalid split for case {case['id']!r}.")
        if not case.get("group"):
            raise ValueError(f"Missing topic group for case {case['id']!r}.")
        previous_split = group_splits.setdefault(case["group"], case["split"])
        if previous_split != case["split"]:
            raise ValueError(
                f"Topic group {case['group']!r} crosses development and test."
            )
        if not case.get("expected_answer"):
            raise ValueError(f"Missing expected answer for case {case['id']!r}.")
        if not case.get("evidence"):
            raise ValueError(f"Missing evidence for case {case['id']!r}.")

        review_statuses.add(case.get("review_status", "missing"))
        for evidence in case["evidence"]:
            if not evidence.get("id") or evidence["id"] in evidence_ids:
                raise ValueError(f"Missing or duplicate evidence ID: {evidence.get('id')!r}")
            evidence_ids.add(evidence["id"])
            location = (evidence["source"], evidence.get("page"))
            source_text = document_by_location.get(location)
            normalized_source = (
                " ".join(source_text.casefold().split())
                if source_text is not None
                else None
            )
            normalized_anchor = " ".join(evidence["anchor"].casefold().split())
            if normalized_source is None or normalized_anchor not in normalized_source:
                raise ValueError(
                    f"Evidence anchor for {evidence['id']!r} does not match "
                    "the cited source/page."
                )
            quotation = evidence.get("quotation")
            normalized_quotation = (
                " ".join(quotation.casefold().split()) if quotation else ""
            )
            if not quotation or normalized_anchor not in normalized_quotation:
                raise ValueError(
                    f"Evidence {evidence['id']!r} needs a quotation containing "
                    "its chunk-size-independent anchor."
                )
            if normalized_quotation not in normalized_source:
                raise ValueError(
                    f"Evidence quotation for {evidence['id']!r} does not match "
                    "the cited source/page."
                )

    return cases, review_statuses


def index_documents(documents, chunk_size, chunk_overlap, cache_path):
    if chunk_size is None:
        chunk_size = max(len(document["text"]) for document in documents)
    chunks, embeddings, metadata = load_or_create_index(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        cache_path=cache_path,
    )
    retriever = build_retriever(chunks, embeddings, metadata)
    store = retriever.vector_store
    return chunks, store, metadata, retriever


def run_configuration(documents, cases, configuration, cache_dir):
    query_embeddings = {
        case["question"]: embed_text(case["question"]) for case in cases
    }
    chunks, store, metadata, retriever = index_documents(
        documents,
        configuration["chunk_size"],
        configuration["chunk_overlap"],
        cache_dir / f"{configuration['name']}.npz",
    )
    def search(question):
        mode = (
            "hybrid"
            if configuration["retrieval_method"] == "dense_bm25_rrf"
            else "dense"
        )
        return retriever.search_with_embedding(
            question, query_embeddings[question], top_k=max(10, len(chunks)),
            mode=mode,
        )

    metrics = evaluate(cases, search)
    return {
        "name": configuration["name"],
        **configuration,
        "chunk_count": len(chunks),
        **metrics,
    }


def _selection_score(result):
    return (
        result["hit_at_k"],
        result["mrr_at_k"],
        result["evidence_recall_at_k"],
    )


def _select_context_bounded_candidate(results, baseline, tolerance_percent=1.0):
    maximum_context_chars = baseline["mean_context_chars_at_k"] * (
        1 + tolerance_percent / 100
    )
    eligible = [
        result
        for result in results
        if result["mean_context_chars_at_k"] <= maximum_context_chars
    ]
    best = max(eligible, key=_selection_score)
    selected = (
        best
        if _selection_score(best) > _selection_score(baseline)
        else baseline
    )
    excluded = [
        {
            "name": result["name"],
            "mean_context_chars_at_k": result["mean_context_chars_at_k"],
            "maximum_allowed_context_chars_at_k": maximum_context_chars,
        }
        for result in results
        if result["mean_context_chars_at_k"] > maximum_context_chars
    ]
    return selected, excluded, maximum_context_chars


def _metric_delta(baseline, final):
    change = final - baseline
    return {
        "baseline": baseline,
        "final": final,
        "percentage_points": round(change * 100, 2),
        "relative_percent": (
            round(change / baseline * 100, 2) if baseline else None
        ),
    }


def _hardware_summary():
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


def _unanswerable_statuses(dataset):
    cases = dataset.get("unanswerable", [])
    if not isinstance(cases, list):
        raise ValueError("Dataset unanswerable field must be a list.")
    ids = set()
    questions = set()
    groups = {}
    statuses = set()
    for case in cases:
        if not case.get("id") or case["id"] in ids:
            raise ValueError("Unanswerable cases need unique non-empty IDs.")
        ids.add(case["id"])
        normalized = " ".join(case["question"].casefold().split())
        if normalized in questions:
            raise ValueError(f"Duplicate evaluation question: {case['question']!r}")
        questions.add(normalized)
        if case.get("split") not in {"development", "held-out-test"}:
            raise ValueError(f"Invalid split for case {case['id']!r}.")
        group = case.get("group")
        if not group:
            raise ValueError(f"Missing topic group for case {case['id']!r}.")
        previous_split = groups.setdefault(group, case["split"])
        if previous_split != case["split"]:
            raise ValueError(f"Topic group {group!r} crosses splits.")
        if not case.get("corpus_absence_rationale"):
            raise ValueError(
                f"Unanswerable case {case['id']!r} needs a corpus-absence rationale."
            )
        statuses.add(case.get("review_status", "missing"))
    return cases, statuses


def _validate_question_set(answerable, unanswerable):
    seen_questions = set()
    group_splits = {}
    for case in [*answerable, *unanswerable]:
        normalized = " ".join(case["question"].casefold().split())
        if normalized in seen_questions:
            raise ValueError(f"Duplicate evaluation question: {case['question']!r}")
        seen_questions.add(normalized)
        previous_split = group_splits.setdefault(case["group"], case["split"])
        if previous_split != case["split"]:
            raise ValueError(
                f"Topic group {case['group']!r} crosses development and test."
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare retrieval configurations on development questions, then "
            "evaluate the selected configuration on grouped held-out questions."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_FILE,
        help=f"Path for the raw JSON results (default: {REPORT_FILE.name}).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVALUATION_FILE,
        help="Versioned evaluation-question JSON file.",
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
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help=(
            "Run a provisional diagnostic with assistant-drafted labels. "
            "Results are not evaluation evidence until reviewed by a person."
        ),
    )
    args = parser.parse_args()

    try:
        dataset_path = args.dataset.expanduser().resolve()
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        documents, corpus_info = load_benchmark_corpus(
            documents_dir=args.documents_dir,
            corpus_manifest=(
                args.corpus_manifest
                or (None if args.documents_dir else DEFAULT_CORPUS_MANIFEST)
            ),
        )
        cases, answerable_review_statuses = validate_dataset(dataset, documents)
        unanswerable, unanswerable_review_statuses = _unanswerable_statuses(dataset)
        _validate_question_set(cases, unanswerable)
        review_statuses = answerable_review_statuses | unanswerable_review_statuses
        evidence_sources = {
            evidence["source"]
            for case in cases
            for evidence in case["evidence"]
        }
        loaded_sources = {document["source"] for document in documents}
        missing_sources = evidence_sources - loaded_sources
        if missing_sources:
            raise ValueError(
                "Evaluation documents are missing: "
                + ", ".join(sorted(missing_sources))
            )

        unreviewed = sorted(
            status for status in review_statuses if status != "reviewed"
        )
        if unreviewed and not args.allow_unreviewed:
            raise ValueError(
                "Question/evidence labels are not human-reviewed "
                f"({', '.join(unreviewed)}). Review the dataset first, or use "
                "--allow-unreviewed for a provisional diagnostic."
            )

        development = [case for case in cases if case["split"] == "development"]
        held_out = [case for case in cases if case["split"] == "held-out-test"]
        if not development or not held_out:
            raise ValueError("Both development and held-out-test splits are required.")

        started_at = datetime.now(timezone.utc).isoformat()
        started = perf_counter()
        cache_dir = ROOT / "data" / ".rag_cache" / "evaluation"
        development_results = [
            run_configuration(documents, development, configuration, cache_dir)
            for configuration in CONFIGURATIONS
        ]
        baseline_result = next(
            result for result in development_results
            if result["name"] == BASELINE_NAME
        )
        best_candidate, context_excluded, maximum_context_chars = (
            _select_context_bounded_candidate(
                development_results, baseline_result
            )
        )
        chosen_name = best_candidate["name"]
        final_configuration = next(
            configuration for configuration in CONFIGURATIONS
            if configuration["name"] == chosen_name
        )
        baseline_configuration = next(
            configuration for configuration in CONFIGURATIONS
            if configuration["name"] == BASELINE_NAME
        )
        held_out_baseline = run_configuration(
            documents, held_out, baseline_configuration, cache_dir
        )
        held_out_final = run_configuration(
            documents, held_out, final_configuration, cache_dir
        )
        elapsed_seconds = perf_counter() - started

        report = {
            "started_at_utc": started_at,
            "benchmark_version": "4.0",
            "dataset_version": dataset.get("dataset_version"),
            "dataset_path": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "requirements_sha256": hashlib.sha256(
                (ROOT / "requirements.txt").read_bytes()
            ).hexdigest(),
            "package_versions": {
                name: version(name)
                for name in ("sentence-transformers", "numpy", "torch", "pypdf")
            },
            "label_review_status": sorted(review_statuses),
            "provisional_unreviewed_diagnostic": bool(unreviewed),
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_cutoffs": {
                "hit_at": 5,
                "mrr_at": 10,
                "evidence_recall_at": 5,
                "context_characters_at": 5,
            },
            "evidence_recall_denominator": (
                "total labeled evidence items across answerable questions"
            ),
            "hit_definition": (
                "A question is a hit if any labeled evidence item is retrieved "
                "within the top 5."
            ),
            "context_length_definition": (
                "Sum of characters in the top 5 retrieved chunks, averaged "
                "across questions."
            ),
            "context_length_fairness": {
                "selection_guard": (
                    "A candidate is eligible only if its development mean "
                    "top-five context length is at most 1% above the baseline."
                ),
                "baseline_development_mean_chars": baseline_result[
                    "mean_context_chars_at_k"
                ],
                "maximum_candidate_development_mean_chars": maximum_context_chars,
                "excluded_candidates": context_excluded,
            },
            "latency_scope": (
                "Per-question retrieval/search only; query embedding, corpus "
                "parsing, index construction and answer generation are excluded."
            ),
            "retrieval_method_parameters": {
                "dense": "Cosine similarity using the existing vector store.",
                "dense_bm25_rrf": {
                    "bm25_k1": 1.5,
                    "bm25_b": 0.75,
                    "reciprocal_rank_fusion_constant": 60,
                    "candidate_scope": "All indexed chunks for both rankers.",
                },
            },
            "corpus": corpus_info,
            "dataset_counts": {
                "answerable_total": len(cases),
                "development": len(development),
                "held_out_test": len(held_out),
                "unanswerable": len(unanswerable),
                "topic_groups": len(
                    {case["group"] for case in [*cases, *unanswerable]}
                ),
            },
            "unanswerable_evaluation": {
                "status": "not_scored",
                "question_count": len(unanswerable),
                "reason": (
                    "Unanswerable questions require a reviewed abstention rubric "
                    "and generation evaluation; retrieval metrics apply only to "
                    "answerable questions."
                ),
            },
            "development_configurations": development_results,
            "chosen_configuration": {
                "name": final_configuration["name"],
                **final_configuration,
                "selection_rule": (
                    "Keep the baseline unless a candidate strictly improves "
                    "the development tuple (Hit@5, MRR@10, Evidence Recall@5) "
                    "while remaining within the context-length guard."
                ),
            },
            "held_out_comparison": {
                "baseline": held_out_baseline,
                "final": held_out_final,
                "delta": {
                    "hit_at_5": _metric_delta(
                        held_out_baseline["hit_at_k"],
                        held_out_final["hit_at_k"],
                    ),
                    "mrr_at_10": _metric_delta(
                        held_out_baseline["mrr_at_k"],
                        held_out_final["mrr_at_k"],
                    ),
                    "evidence_recall_at_5": _metric_delta(
                        held_out_baseline["evidence_recall_at_k"],
                        held_out_final["evidence_recall_at_k"],
                    ),
                },
                "context_length": {
                    "baseline_mean_top_5_chars": (
                        held_out_baseline["mean_context_chars_at_k"]
                    ),
                    "final_mean_top_5_chars": (
                        held_out_final["mean_context_chars_at_k"]
                    ),
                    "absolute_change_chars": (
                        held_out_final["mean_context_chars_at_k"]
                        - held_out_baseline["mean_context_chars_at_k"]
                    ),
                    "relative_change_percent": (
                        (
                            held_out_final["mean_context_chars_at_k"]
                            - held_out_baseline["mean_context_chars_at_k"]
                        )
                        / held_out_baseline["mean_context_chars_at_k"]
                        * 100
                        if held_out_baseline["mean_context_chars_at_k"]
                        else None
                    ),
                },
            },
            "elapsed_seconds": round(elapsed_seconds, 3),
            "hardware": _hardware_summary(),
            "answer_quality_evaluation": {
                "status": "not_measured",
                "reason": (
                    "No human-reviewed answer rubric and fixed generation/API "
                    "configuration were provided."
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
        parser.error(f"Could not evaluate retrieval: {error}")

    print(f"Corpus documents: {corpus_info['document_count']}")
    print(f"Questions: {len(cases)} (development {len(development)}, held-out {len(held_out)})")
    print(f"Unanswerable questions (separate, not scored): {len(unanswerable)}")
    print(f"Human-reviewed: {not bool(unreviewed)}")
    print(f"Selected configuration: {final_configuration['name']}")
    print("\nHeld-out metrics (provisional if labels are unreviewed):")
    print(
        f"  {'Version':<9} {'Hit@5':>8} {'MRR@10':>8} "
        f"{'Evidence Recall@5':>19} {'Mean top-5 chars':>18}"
    )
    for name, result in (
        ("baseline", held_out_baseline),
        ("final", held_out_final),
    ):
        print(
            f"  {name:<9} {result['hit_at_k']:>7.1%} "
            f"{result['mrr_at_k']:>8.3f} "
            f"{result['evidence_recall_at_k']:>18.1%} "
            f"{result['mean_context_chars_at_k']:>18.1f}"
        )
    print(f"\nDetailed JSON report: {report_path}")


if __name__ == "__main__":
    main()
