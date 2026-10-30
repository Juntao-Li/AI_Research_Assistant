from math import ceil
from statistics import mean, median
from time import perf_counter


def _normalize_evidence_text(text):
    return " ".join(text.casefold().split())


def _matches_evidence(result, evidence):
    metadata = result["metadata"]
    return (
        metadata.get("source") == evidence["source"]
        and metadata.get("page") == evidence.get("page")
        and _normalize_evidence_text(evidence["anchor"])
        in _normalize_evidence_text(result["text"])
    )


def evaluate(cases, search, hit_k=5, mrr_k=10, evidence_k=5):
    if not cases:
        raise ValueError("Evaluation set must contain at least one question.")
    if min(hit_k, mrr_k, evidence_k) <= 0:
        raise ValueError("Evaluation cutoffs must be greater than 0.")

    outcomes = []
    reciprocal_ranks = []
    search_latencies_ms = []
    found_evidence_count = 0
    expected_evidence_count = 0

    for case in cases:
        evidence = case["evidence"]
        if not evidence:
            raise ValueError(
                f"Answerable case {case.get('id', case['question'])!r} "
                "must label at least one evidence passage."
            )

        search_started = perf_counter()
        results = search(case["question"])
        search_latency_ms = (perf_counter() - search_started) * 1000
        search_latencies_ms.append(search_latency_ms)
        evidence_ranks = []
        for evidence_item in evidence:
            rank = next(
                (
                    index
                    for index, result in enumerate(results, start=1)
                    if _matches_evidence(result, evidence_item)
                ),
                None,
            )
            evidence_ranks.append(
                {"evidence_id": evidence_item["id"], "rank": rank}
            )

        relevant_ranks = [
            item["rank"] for item in evidence_ranks if item["rank"] is not None
        ]
        first_relevant_rank = min(relevant_ranks, default=None)
        hit = any(
            rank is not None and rank <= hit_k
            for rank in (item["rank"] for item in evidence_ranks)
        )
        found_at_evidence_k = sum(
            rank is not None and rank <= evidence_k
            for rank in (item["rank"] for item in evidence_ranks)
        )
        expected_evidence_count += len(evidence)
        found_evidence_count += found_at_evidence_k
        reciprocal_ranks.append(
            1 / first_relevant_rank
            if first_relevant_rank is not None
            and first_relevant_rank <= mrr_k
            else 0.0
        )
        outcomes.append(
            {
                "id": case.get("id"),
                "group": case.get("group"),
                "split": case.get("split"),
                "question": case["question"],
                "search_latency_ms": search_latency_ms,
                "evidence_ranks": evidence_ranks,
                "first_relevant_rank": first_relevant_rank,
                "hit_at_k": hit,
                "evidence_found_at_k": found_at_evidence_k,
                "evidence_count": len(evidence),
                "context_chars_at_evidence_k": sum(
                    len(result["text"]) for result in results[:evidence_k]
                ),
            }
        )

    count = len(outcomes)
    hit_rate = sum(outcome["hit_at_k"] for outcome in outcomes) / count
    mean_reciprocal_rank = sum(reciprocal_ranks) / count
    sorted_latencies = sorted(search_latencies_ms)
    return {
        "question_count": count,
        "hit_k": hit_k,
        "hit_at_k": hit_rate,
        "mrr_k": mrr_k,
        "mrr_at_k": mean_reciprocal_rank,
        "evidence_recall_k": evidence_k,
        "evidence_found": found_evidence_count,
        "evidence_denominator": expected_evidence_count,
        "evidence_recall_at_k": (
            found_evidence_count / expected_evidence_count
        ),
        "context_k": evidence_k,
        "mean_context_chars_at_k": (
            sum(outcome["context_chars_at_evidence_k"] for outcome in outcomes)
            / count
        ),
        "search_latency_ms": {
            "mean": mean(search_latencies_ms),
            "median": median(search_latencies_ms),
            "p95_nearest_rank": sorted_latencies[
                max(0, ceil(0.95 * count) - 1)
            ],
            "per_question": search_latencies_ms,
            "scope": "vector search or dense-plus-BM25 retrieval; query embedding excluded",
        },
        "outcomes": outcomes,
    }
