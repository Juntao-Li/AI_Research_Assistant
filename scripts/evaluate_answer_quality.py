import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from scripts.benchmark_corpus import load_benchmark_corpus
from scripts.evaluate_retrieval import (
    _unanswerable_statuses,
    validate_dataset,
)
from src.embedding import EMBEDDING_MODEL
from src.generator import SYSTEM_PROMPT, generate_answer
from src.indexer import load_or_create_index
from src.retriever import build_retriever


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "benchmark_corpus_v2.json"
DEFAULT_DATASET = ROOT / "data" / "retrieval_evaluation_v4.json"
DEFAULT_REPORT = ROOT / "reports" / "answer_quality_v1.json"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TEMPERATURE = 0
SYSTEM_PROMPT_VERSION = "local-rag-answer-v1"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate answer-quality examples for human review using fixed "
            "retrieval and generation settings."
        )
    )
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="dense")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Required while any question label is pending.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("Set OPENAI_API_KEY before generating answer-quality results.")

    try:
        dataset_path = args.dataset.expanduser().resolve()
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        documents, corpus_info = load_benchmark_corpus(
            corpus_manifest=args.corpus_manifest
        )
        answerable, answerable_statuses = validate_dataset(dataset, documents)
        unanswerable, unanswerable_statuses = _unanswerable_statuses(dataset)
        statuses = answerable_statuses | unanswerable_statuses
        if statuses != {"reviewed"} and not args.allow_unreviewed:
            parser.error(
                "Question labels are not fully reviewed; use --allow-unreviewed "
                "to generate provisional examples."
            )

        chunks, embeddings, metadata = load_or_create_index(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            cache_path=ROOT / "data" / ".rag_cache" / "answer_quality_v1.npz",
        )
        retriever = build_retriever(chunks, embeddings, metadata)
        outputs = []
        for answerable_case, cases in (
            (True, answerable),
            (False, unanswerable),
        ):
            for case in cases:
                results = retriever.search(
                    case["question"], top_k=args.top_k, mode=args.retrieval_mode
                )
                context_parts = []
                for index, result in enumerate(results, start=1):
                    source = result["metadata"]
                    page = f", page {source['page']}" if source.get("page") else ""
                    context_parts.append(
                        f"[{index}] Source: {source['source']} "
                        f"(chunk {source['chunk_id']}{page})\n{result['text']}"
                    )
                answer = generate_answer(
                    question=case["question"],
                    context="\n\n".join(context_parts),
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    temperature=TEMPERATURE,
                )
                cited_ranks = sorted(
                    {int(number) for number in CITATION_PATTERN.findall(answer)}
                )
                outputs.append(
                    {
                        "id": case["id"],
                        "question": case["question"],
                        "answerable": answerable_case,
                        "split": case["split"],
                        "expected_answer": case.get("expected_answer"),
                        "evidence": case.get("evidence", []),
                        "corpus_absence_rationale": case.get(
                            "corpus_absence_rationale"
                        ),
                        "generated_answer": answer,
                        "retrieved_passages": [
                            {
                                "rank": rank,
                                "source": result["metadata"]["source"],
                                "page": result["metadata"].get("page"),
                                "chunk_id": result["metadata"]["chunk_id"],
                                "score": result["score"],
                                "score_type": result["score_type"],
                                "text": result["text"],
                            }
                            for rank, result in enumerate(results, start=1)
                        ],
                        "cited_ranks": cited_ranks,
                        "citation_ranks_valid": all(
                            1 <= rank <= len(results) for rank in cited_ranks
                        ),
                        "review_status": "pending",
                        "review": {
                            "answer_correctness": None,
                            "citation_support": None,
                            "appropriate_abstention": None,
                            "unnecessary_refusal": None,
                            "review_notes": "",
                        },
                    }
                )

        report = {
            "report_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provisional": True,
            "question_label_review_status": sorted(statuses),
            "dataset_version": dataset.get("dataset_version"),
            "dataset_path": str(dataset_path),
            "dataset_sha256": hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest(),
            "corpus": corpus_info,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval": {
                "mode": args.retrieval_mode,
                "top_k": args.top_k,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "candidate_scope": "all indexed chunks",
            },
            "generation": {
                "provider_base_url": args.base_url,
                "model": args.model,
                "temperature": TEMPERATURE,
                "system_prompt_version": SYSTEM_PROMPT_VERSION,
                "system_prompt_sha256": hashlib.sha256(
                    SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
            },
            "review_rubric": {
                "answer_correctness": (
                    "Mark correct only when the response completely and accurately "
                    "answers the question using the source documents."
                ),
                "citation_support": (
                    "Judge each cited rank against the exact retrieved passage; "
                    "mark supported only when it entails the associated claim."
                ),
                "appropriate_abstention": (
                    "For unanswerable questions, mark yes only if the response "
                    "clearly states that the corpus excerpts do not establish the answer."
                ),
                "unnecessary_refusal": (
                    "For answerable questions, mark yes if the response refuses or "
                    "claims insufficient evidence despite adequate retrieved evidence."
                ),
                "allowed_values": ["yes", "no", "uncertain"],
            },
            "counts": {
                "documents": corpus_info["document_count"],
                "chunks": len(chunks),
                "answerable": len(answerable),
                "unanswerable": len(unanswerable),
                "generated": len(outputs),
            },
            "excluded_stages": [
                "document parsing is excluded from reported answer quality",
                "index preparation latency is not end-to-end answer latency",
            ],
            "results": outputs,
        }
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Could not generate answer-quality review set: {error}")

    print(
        f"Wrote {len(outputs)} pending answer-quality examples "
        f"({len(answerable)} answerable, {len(unanswerable)} unanswerable) "
        f"to {report_path}"
    )


if __name__ == "__main__":
    main()
