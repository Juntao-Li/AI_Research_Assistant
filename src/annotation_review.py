import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


_WRITE_LOCK = threading.RLock()
REVIEWABLE_STATUSES = {"pending", "reviewed", "needs_correction"}

SYNCHRONIZATION_CORRECTIONS = {
    "asyncio_synchronization-q1": {
        "anchor": "An asyncio lock can be used to guarantee exclusive access to a",
        "expected_answer": (
            "An asyncio.Lock can guarantee exclusive access to a shared resource."
        ),
        "reason": (
            "The original evidence cited a source-code path, which does not "
            "support the lock's purpose."
        ),
    },
    "asyncio_synchronization-q2": {
        "anchor": "All tasks waiting for event to be set will be immediately",
        "expected_answer": (
            "Calling set() immediately awakens every task waiting for the event."
        ),
        "reason": (
            "The original evidence only listed synchronization primitive names "
            "and did not support the expected answer."
        ),
    },
    "asyncio_synchronization-q3": {
        "anchor": "A semaphore manages an internal counter which is decremented",
        "expected_answer": (
            "A semaphore uses a counter, and acquire() blocks when the counter "
            "reaches zero."
        ),
        "reason": (
            "The original evidence listed primitive names rather than explaining "
            "how a semaphore limits concurrent acquisition."
        ),
    },
}


def atomic_json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _quotation_from_source(document_root, source, anchor):
    source_path = Path(document_root, source).resolve()
    root = Path(document_root).resolve()
    if root not in source_path.parents or not source_path.is_file():
        raise ValueError(f"Evidence source is not in the selected corpus: {source}")
    text = source_path.read_text(encoding="utf-8")
    normalized_anchor = " ".join(anchor.split()).casefold()
    for paragraph in text.split("\n\n"):
        normalized_paragraph = " ".join(paragraph.split())
        if normalized_anchor in normalized_paragraph.casefold():
            return normalized_paragraph
    raise ValueError("Evidence anchor was not found in the selected source.")


def ensure_review_dataset(source_path, output_path, document_root):
    source_path = Path(source_path)
    output_path = Path(output_path)
    with _WRITE_LOCK:
        if output_path.exists():
            return json.loads(output_path.read_text(encoding="utf-8"))

        dataset = json.loads(source_path.read_text(encoding="utf-8"))
        if dataset.get("dataset_version") != "4.0":
            raise ValueError("Annotation review requires retrieval dataset v4.0.")

        correction_records = []
        for case in dataset["answerable"]:
            correction = SYNCHRONIZATION_CORRECTIONS.get(case["id"])
            if correction is None:
                continue
            evidence = case["evidence"][0]
            evidence["anchor"] = correction["anchor"]
            evidence["quotation"] = _quotation_from_source(
                document_root, evidence["source"], correction["anchor"]
            )
            case["expected_answer"] = correction["expected_answer"]
            case["review_status"] = "pending"
            case["review_notes"] = ""
            case["correction_reason"] = correction["reason"]
            case["label_origin"] = (
                "proposed source-audit correction; human review pending"
            )
            correction_records.append(
                {
                    "question_id": case["id"],
                    "action": "proposed_correction",
                    "reason": correction["reason"],
                    "review_status": "pending",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

        dataset["dataset_version"] = "5.0"
        dataset["parent_dataset_version"] = "4.0"
        dataset["label_origin"] = (
            "v4 labels plus pending source-audit proposals; approval requires "
            "explicit human action in the review interface."
        )
        dataset["review_instructions"] = (
            "Review every question and its evidence. Approve only after source "
            "verification; use Needs correction with a reason otherwise. "
            "Topic groups and original development/test assignments are frozen."
        )
        dataset["review_log"] = correction_records
        for case in [*dataset["answerable"], *dataset["unanswerable"]]:
            case.setdefault("review_notes", "")
            case.setdefault("correction_reason", "")
            if case.get("review_status") not in REVIEWABLE_STATUSES:
                case["review_status"] = "pending"

        atomic_json_write(output_path, dataset)
        return dataset


def update_review_case(
    dataset_path,
    log_path,
    question_id,
    *,
    action,
    review_notes,
    reviewer,
    document_root,
    reason="",
    expected_answer=None,
    evidence=None,
    corpus_search_rationale=None,
):
    if action not in {"approve", "needs_correction", "leave_pending"}:
        raise ValueError("Unsupported review action.")
    if action == "needs_correction" and not reason.strip():
        raise ValueError("A correction reason is required.")

    dataset_path = Path(dataset_path)
    log_path = Path(log_path)
    with _WRITE_LOCK:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        answerable_ids = {item["id"] for item in dataset["answerable"]}
        unanswerable_ids = {item["id"] for item in dataset["unanswerable"]}
        case = next(
            (
                item
                for item in [*dataset["answerable"], *dataset["unanswerable"]]
                if item["id"] == question_id
            ),
            None,
        )
        if case is None:
            raise ValueError(f"Unknown evaluation question ID: {question_id}")
        previous_status = case.get("review_status", "pending")
        before = {
            "expected_answer": case.get("expected_answer"),
            "evidence": case.get("evidence"),
        }
        if expected_answer is not None:
            if not expected_answer.strip():
                raise ValueError("Expected answer cannot be empty.")
            case["expected_answer"] = expected_answer.strip()
        if evidence is not None:
            if question_id not in answerable_ids:
                raise ValueError("Unanswerable questions do not have evidence fields.")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("At least one supporting evidence item is required.")
            for item in evidence:
                if not all(
                    item.get(field) for field in ("id", "source", "anchor", "quotation")
                ):
                    raise ValueError(
                        "Evidence needs an ID, source, anchor, and quotation."
                    )
                source_path = Path(document_root, item["source"]).resolve()
                root = Path(document_root).resolve()
                if root not in source_path.parents or not source_path.is_file():
                    raise ValueError(
                        "Evidence source is outside the selected corpus."
                    )
                source_text = " ".join(
                    source_path.read_text(encoding="utf-8").casefold().split()
                )
                quotation = " ".join(item["quotation"].casefold().split())
                anchor = " ".join(item["anchor"].casefold().split())
                if anchor not in quotation or quotation not in source_text:
                    raise ValueError(
                        "Evidence quotation and anchor must match the selected source."
                    )
            case["evidence"] = evidence
        evidence_changed = before["evidence"] != case.get("evidence")
        answer_changed = before["expected_answer"] != case.get("expected_answer")
        edited = evidence_changed or answer_changed
        if edited and previous_status == "reviewed":
            action = "leave_pending"
            reason = reason or "Approval reset because approved evidence was edited."
        if edited and not reason.strip():
            raise ValueError("A reason is required for answer or evidence corrections.")

        if question_id in unanswerable_ids:
            if corpus_search_rationale is None or not corpus_search_rationale.strip():
                raise ValueError(
                    "Record why the requested answer is absent from the corpus."
                )
            case["corpus_absence_rationale"] = corpus_search_rationale.strip()

        case["review_notes"] = review_notes.strip()
        case["correction_reason"] = reason.strip()
        if action == "approve":
            case["review_status"] = "reviewed"
        elif action == "needs_correction":
            case["review_status"] = "needs_correction"
        else:
            case["review_status"] = "pending"

        record = {
            "question_id": question_id,
            "action": action,
            "previous_status": previous_status,
            "new_status": case["review_status"],
            "reviewer": reviewer.strip() or "local reviewer",
            "reason": reason.strip(),
            "answer_edited": answer_changed,
            "evidence_edited": evidence_changed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        dataset.setdefault("review_log", []).append(record)
        atomic_json_write(dataset_path, dataset)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as review_log:
            review_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            review_log.flush()
            os.fsync(review_log.fileno())
        return dataset, record
