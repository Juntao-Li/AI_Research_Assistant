import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "data" / "retrieval_evaluation_v2.json"
CORPUS_DIR = ROOT / "data" / "benchmark_documents"


CORRECTIONS = {
    "argparse-q4": {
        "evidence": (
            "Some command-line arguments should be selected from a restricted "
            "set of values."
        ),
        "question": "How can argparse restrict a command-line argument to a defined set of values?",
        "expected_answer": (
            "Pass a sequence as the choices keyword to add_argument(); parsed "
            "values outside that set produce an error."
        ),
        "reason": (
            "The v2 quotation described suggest_on_error suggestions, not the "
            "choices option's validation behavior."
        ),
    },
    "asyncio_tasks-q1": {
        "evidence": (
            "The :func:`asyncio.run` function to run the top-level entry point"
        ),
        "question": "What role does asyncio.run() serve in an asyncio program?",
        "expected_answer": (
            "It runs the program's top-level coroutine entry point."
        ),
        "reason": (
            "The v2 evidence was only example output and did not support the "
            "event-loop-management claim in the expected answer."
        ),
    },
    "asyncio_tasks-q2": {
        "evidence": (
            "When a coroutine is wrapped into a *Task* with functions like"
        ),
        "question": "How does asyncio.create_task() schedule a coroutine?",
        "expected_answer": (
            "It wraps the coroutine in a Task and schedules it to run soon."
        ),
        "reason": (
            "Replaced a broad introductory mention with the Task documentation "
            "that directly describes coroutine scheduling."
        ),
    },
    "asyncio_tasks-q4": {
        "evidence": "The order of result values",
        "question": (
            "In what order does asyncio.gather() return successful results?"
        ),
        "expected_answer": (
            "The result values follow the order of the awaitables supplied to "
            "gather(), not their completion order."
        ),
        "reason": (
            "The v2 evidence was an example invocation and did not establish "
            "the documented result-order guarantee."
        ),
    },
    "asyncio_tasks-q5": {
        "evidence": "Event loops use cooperative scheduling",
        "question": (
            "Why can another asyncio task run while a task awaits a Future?"
        ),
        "expected_answer": (
            "A Task suspends while awaiting a Future, so cooperative scheduling "
            "lets the event loop run other tasks."
        ),
        "reason": (
            "The original quotation defined coroutine syntax but did not support "
            "the claim about suspension and cooperative scheduling."
        ),
    },
    "asyncio_streams-q1": {
        "evidence": (
            "reader* and *writer* objects are instances of"
        ),
        "question": "What does asyncio.open_connection() return?",
        "expected_answer": (
            "It returns a StreamReader and StreamWriter pair for the connection."
        ),
        "reason": (
            "Replaced the function-name-only anchor with the API documentation "
            "that states the returned object types."
        ),
    },
    "asyncio_streams-q4": {
        "evidence": "*limit* determines the buffer size limit used by the",
        "question": "What does the limit parameter of asyncio.open_connection() control?",
        "expected_answer": (
            "It controls the buffer-size limit of the StreamReader returned by "
            "the stream API."
        ),
        "reason": (
            "The v2 quote only mentioned the parameter and did not state what "
            "it controls."
        ),
    },
    "asyncio_streams-q5": {
        "evidence": "Start a socket server.",
        "question": "What does asyncio.start_server() start?",
        "expected_answer": "It starts a socket server.",
        "reason": (
            "The replacement quotation explicitly states that start_server "
            "starts a socket server, matching the expected answer."
        ),
    },
    "asyncio_queues-q3": {
        "evidence": "Block until all items in the queue have been received and processed.",
        "question": "What condition makes asyncio.Queue.join() unblock?",
        "expected_answer": "It unblocks after all queue items have been received and processed.",
        "reason": (
            "The v2 anchor named join but did not state its completion condition."
        ),
    },
}


def normalize(text):
    return " ".join(text.split())


def source_paragraph(source, anchor):
    text = (CORPUS_DIR / source).read_text(encoding="utf-8")
    normalized_anchor = normalize(anchor).casefold()
    normalized_text = normalize(text)
    position = normalized_text.casefold().find(normalized_anchor)
    if position < 0:
        raise ValueError(f"Could not find {anchor!r} in {source}.")
    paragraphs = [normalize(part) for part in text.split("\n\n") if part.strip()]
    for paragraph in paragraphs:
        if normalized_anchor in paragraph.casefold():
            return paragraph
    excerpt_start = max(0, position - 200)
    return normalized_text[excerpt_start:position + len(anchor) + 300]


def main():
    parser = argparse.ArgumentParser(
        description="Create a source-corrected pending retrieval dataset version."
    )
    parser.add_argument(
        "--version", type=int, choices=(3, 4), default=3,
        help="Dataset version to create (default: 3).",
    )
    args = parser.parse_args()
    dataset_path = ROOT / "data" / f"retrieval_evaluation_v{args.version}.json"
    review_path = ROOT / "data" / f"retrieval_evaluation_v{args.version}_review.md"
    correction_log = (
        ROOT / "data" / f"retrieval_evaluation_v{args.version}_corrections.md"
    )
    corrections_for_version = CORRECTIONS
    if args.version == 3:
        original_v3_corrections = {
            "argparse-q4",
            "asyncio_tasks-q1",
            "asyncio_tasks-q4",
            "asyncio_streams-q1",
        }
        corrections_for_version = {
            key: value
            for key, value in CORRECTIONS.items()
            if key in original_v3_corrections
        }

    dataset = json.loads(SOURCE_DATASET.read_text(encoding="utf-8"))
    corrections = []
    for case in dataset["answerable"]:
        update = corrections_for_version.get(case["id"])
        if not update:
            continue
        evidence = case["evidence"][0]
        evidence["anchor"] = update["evidence"]
        evidence["quotation"] = source_paragraph(
            evidence["source"], update["evidence"]
        )
        case["question"] = update["question"]
        case["expected_answer"] = update["expected_answer"]
        case["review_status"] = "pending"
        case["label_origin"] = "corrected assistant draft; human review pending"
        corrections.append(
            {
                "id": case["id"],
                "source": evidence["source"],
                "reason": update["reason"],
                "new_anchor": evidence["anchor"],
                "new_quotation": evidence["quotation"],
            }
        )

    duplicate_pairs = [
        {
            "ids": ["argparse-q1", "argparse-q2"],
            "relationship": "near-duplicate",
            "split": "development",
            "action": (
                "Both remain in the same development group/split; identified "
                "to prevent treating them as independent topic coverage."
            ),
        },
        {
            "ids": ["traceback-q1", "traceback-q4"],
            "relationship": "related but non-duplicate",
            "split": "development",
            "action": (
                "Retained in the same split: one asks about printing the active "
                "exception traceback, the other about returning it as text."
            ),
        },
        {
            "ids": ["unittest-q2", "unittest-q3"],
            "relationship": "paired lifecycle questions",
            "split": "held-out-test",
            "action": (
                "Retained in the same group and split; setUp and tearDown ask "
                "distinct lifecycle facts."
            ),
        },
    ]
    for item in duplicate_pairs:
        if item["relationship"] != "near-duplicate":
            continue
        for case_id in item["ids"]:
            next(
                case for case in dataset["answerable"] if case["id"] == case_id
            )["near_duplicate_group"] = item["ids"][0]
    dataset.update(
        {
            "dataset_version": f"{args.version}.0",
            "parent_dataset_version": "2.0",
            "label_origin": (
                "Audited and corrected assistant drafts; all labels remain "
                "pending human review."
            ),
            "audit_note": (
                "Source content, not retriever output, was used to correct "
                "evidence. No review status was promoted automatically."
            ),
            "corrections": corrections,
            "near_duplicate_and_related_question_groups": duplicate_pairs,
        }
    )
    for case in dataset["answerable"] + dataset["unanswerable"]:
        case["review_status"] = "pending"

    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    review_lines = [
        f"# Retrieval evaluation v{args.version}.0 — pending human review",
        "",
        f"Answerable: {len(dataset['answerable'])}; unanswerable: {len(dataset['unanswerable'])}.",
        "All entries are assistant-drafted or corrected and remain `pending`; none are human-reviewed.",
        "",
        "## Review instructions",
        "",
        "Verify every expected answer against the source quotation and full local source. For unanswerable questions, search all 22 selected source files. Fix or remove incorrect cases; only you may mark a case reviewed. Keep grouped and near-duplicate questions in their current split.",
        "",
        "## Answerable questions",
        "",
    ]
    for case in dataset["answerable"]:
        review_lines.extend(
            [
                f"### {case['id']} — {case['split']} / {case['group']} / pending",
                f"- **Question:** {case['question']}",
                f"- **Expected answer:** {case['expected_answer']}",
            ]
        )
        for evidence in case["evidence"]:
            review_lines.extend(
                [
                    f"- **Evidence:** [{evidence['source']}](benchmark_documents/{evidence['source']})",
                    f"- **Anchor:** `{evidence['anchor']}`",
                    f"- **Quotation:** “{evidence['quotation']}”",
                ]
            )
        review_lines.append("")
    review_lines.extend(["## Unanswerable questions", ""])
    for case in dataset["unanswerable"]:
        review_lines.extend(
            [
                f"### {case['id']} — {case['split']} / pending",
                f"- **Question:** {case['question']}",
                f"- **Absence rationale:** {case['corpus_absence_rationale']}",
                "",
            ]
        )
    review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    log_lines = [
        f"# Retrieval dataset v{args.version} correction and audit log",
        "",
        "All changed labels remain pending. Corrections use the corpus source, not a retriever's output.",
        "",
    ]
    for item in corrections:
        log_lines.extend(
            [
                f"## {item['id']}",
                f"- **Source:** [{item['source']}](benchmark_documents/{item['source']})",
                f"- **Correction:** {item['reason']}",
                f"- **Replacement anchor:** `{item['new_anchor']}`",
                f"- **Replacement quotation:** “{item['new_quotation']}”",
                "",
            ]
        )
    log_lines.extend(["## Duplicate/related question audit", ""])
    for item in duplicate_pairs:
        log_lines.append(
            f"- `{', '.join(item['ids'])}` ({item['relationship']}): {item['action']}"
        )
    log_lines.extend(
        [
            "",
            "The dataset validator confirms unique normalized question strings, valid evidence source/anchors/quotations, and no topic group crossing development and held-out splits. These mechanical checks do not replace human semantic review.",
        ]
    )
    correction_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(
        f"Wrote dataset v{args.version} with {len(corrections)} evidence corrections; "
        f"{len(dataset['answerable'])} answerable and "
        f"{len(dataset['unanswerable'])} unanswerable labels remain pending."
    )


if __name__ == "__main__":
    main()
