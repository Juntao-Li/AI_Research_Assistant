import html
import json
import os
import re
from pathlib import Path

import streamlit as st

from scripts.benchmark_corpus import load_benchmark_corpus
from src.annotation_review import ensure_review_dataset, update_review_case


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "data" / "retrieval_evaluation_v4.json"
REVIEW_ROOT = Path(os.getenv("REVIEW_DATA_ROOT", ROOT / "data")).resolve()
REVIEW_DATASET_V5 = REVIEW_ROOT / "retrieval_evaluation_v5.json"
REVIEW_DATASET_V7 = REVIEW_ROOT / "retrieval_evaluation_v7_corrections_applied.json"
REVIEW_LOG_ROOT = REVIEW_ROOT / "review_logs"
CORPUS_MANIFEST = ROOT / "data" / "benchmark_corpus_v2.json"
CORPUS_ROOT = ROOT / "data" / "benchmark_documents"


st.set_page_config(
    page_title="Annotation Review",
    page_icon="📝",
    layout="wide",
)
st.title("Annotation review")
st.caption(
    "Review assistant-drafted labels against the complete source. Approvals are "
    "explicit actions; editing an approved answer or evidence resets its approval."
)

dataset_options = {
    "v5 — original human review": REVIEW_DATASET_V5,
}
if REVIEW_DATASET_V7.is_file():
    dataset_options[
        "v7 — user-requested corrections, awaiting re-review"
    ] = REVIEW_DATASET_V7
selected_dataset_label = st.selectbox(
    "Dataset version",
    list(dataset_options),
    index=1 if REVIEW_DATASET_V7.is_file() else 0,
)
ACTIVE_DATASET = dataset_options[selected_dataset_label]
ACTIVE_REVIEW_LOG = (
    REVIEW_LOG_ROOT / f"{ACTIVE_DATASET.stem}.jsonl"
)

try:
    if ACTIVE_DATASET == REVIEW_DATASET_V5:
        dataset = ensure_review_dataset(
            SOURCE_DATASET, ACTIVE_DATASET, CORPUS_ROOT
        )
    else:
        dataset = json.loads(ACTIVE_DATASET.read_text(encoding="utf-8"))
        if dataset.get("dataset_version") != "7.0-user-requested-corrections":
            raise ValueError(
                f"Unexpected dataset version in {ACTIVE_DATASET}."
            )
    documents, corpus_info = load_benchmark_corpus(
        corpus_manifest=CORPUS_MANIFEST
    )
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the review dataset or source corpus: {error}")
    st.stop()

st.caption(
    f"Active dataset: `{dataset['dataset_version']}` · `{ACTIVE_DATASET}`"
)
all_cases = [*dataset["answerable"], *dataset["unanswerable"]]
reviewed_count = sum(item.get("review_status") == "reviewed" for item in all_cases)
correction_count = sum(
    item.get("review_status") == "needs_correction" for item in all_cases
)
pending_count = len(all_cases) - reviewed_count - correction_count
st.progress(reviewed_count / len(all_cases) if all_cases else 0)
st.caption(
    f"{reviewed_count}/{len(all_cases)} approved · {correction_count} need "
    f"correction · {pending_count} pending"
)
st.info(
    "AI-applied corrections remain pending until you inspect the source and "
    "explicitly approve them. The v5 dataset and its review history are preserved."
)

case_by_id = {case["id"]: case for case in all_cases}
if "review_case_id" not in st.session_state or (
    st.session_state.review_case_id not in case_by_id
):
    st.session_state.review_case_id = all_cases[0]["id"]

selected_id = st.selectbox(
    "Question",
    list(case_by_id),
    index=list(case_by_id).index(st.session_state.review_case_id),
    format_func=lambda case_id: (
        f"{case_id} · {case_by_id[case_id]['split']} · "
        f"{case_by_id[case_id].get('review_status', 'pending')}"
    ),
)
st.session_state.review_case_id = selected_id
case = case_by_id[selected_id]
is_answerable = case in dataset["answerable"]
case_index = list(case_by_id).index(selected_id)

previous_col, next_col, reviewer_col = st.columns([1, 1, 3])
with previous_col:
    if st.button("Previous", disabled=case_index == 0):
        st.session_state.review_case_id = list(case_by_id)[case_index - 1]
        st.rerun()
with next_col:
    if st.button("Next", disabled=case_index == len(case_by_id) - 1):
        st.session_state.review_case_id = list(case_by_id)[case_index + 1]
        st.rerun()
with reviewer_col:
    reviewer = st.text_input("Reviewer name (local log only)", value="local reviewer")

st.subheader(case["question"])
st.write(
    f"**Question ID:** `{case['id']}` · **Split:** `{case['split']}` · "
    f"**Topic group:** `{case.get('group', 'unanswerable')}` · "
    f"**Status:** `{case.get('review_status', 'pending')}`"
)

if is_answerable:
    evidence_items = case["evidence"]
    evidence_index = st.selectbox(
        "Evidence passage",
        range(len(evidence_items)),
        format_func=lambda index: (
            f"{index + 1}. {evidence_items[index]['source']}"
            + (
                f", page {evidence_items[index]['page']}"
                if evidence_items[index].get("page") is not None
                else ""
            )
        ),
    )
    evidence = evidence_items[evidence_index]
    answerable_sources = sorted(
        entry["source"] for entry in corpus_info["manifest"]
    )
    source_index = (
        answerable_sources.index(evidence["source"])
        if evidence["source"] in answerable_sources
        else 0
    )
    selected_source = st.selectbox(
        "Evidence source", answerable_sources, index=source_index
    )
    source_path = (CORPUS_ROOT / selected_source).resolve()
    if CORPUS_ROOT.resolve() not in source_path.parents:
        st.error("Evidence source escaped the selected corpus.")
        st.stop()
    source_text = source_path.read_text(encoding="utf-8")

    left, right = st.columns([1, 2])
    with left:
        expected_answer = st.text_area(
            "Expected answer",
            value=case["expected_answer"],
            height=160,
        )
        evidence_anchor = st.text_area(
            "Evidence anchor",
            value=evidence["anchor"],
            height=100,
        )
        evidence_quotation = st.text_area(
            "Evidence quotation",
            value=evidence["quotation"],
            height=200,
        )
    with right:
        st.markdown("**Complete source document — cited anchor highlighted**")
        anchor_parts = evidence["anchor"].split()
        anchor_pattern = r"\s+".join(re.escape(part) for part in anchor_parts)
        anchor_match = (
            re.search(anchor_pattern, source_text, flags=re.IGNORECASE)
            if anchor_pattern
            else None
        )
        if anchor_match:
            before = html.escape(source_text[: anchor_match.start()])
            highlighted = html.escape(
                source_text[anchor_match.start() : anchor_match.end()]
            )
            after = html.escape(source_text[anchor_match.end() :])
            source_html = (
                "<pre style='white-space:pre-wrap;max-height:70vh;overflow:auto'>"
                f"{before}<mark>{highlighted}</mark>{after}</pre>"
            )
        else:
            source_html = (
                "<pre style='white-space:pre-wrap;max-height:70vh;overflow:auto'>"
                f"{html.escape(source_text)}</pre>"
            )
        st.html(source_html)
        st.download_button(
            "Download complete source text",
            data=source_text,
            file_name=f"{Path(selected_source).stem}.source.txt",
            mime="text/plain",
            key=f"source-download-{selected_id}",
        )
        st.caption(f"Local corpus file: `{source_path}`")
    st.caption(f"Quoted evidence: “{evidence['quotation']}”")
else:
    expected_answer = None
    evidence_anchor = ""
    evidence_quotation = ""
    st.warning(
        "This question is labeled unanswerable. Search is performed over every "
        "document selected by the full corpus manifest."
    )
    search_query = st.text_input(
        "Search the full selected corpus",
        value=case["question"],
    )
    if st.button("Search all corpus sources", type="secondary"):
        terms = [term for term in re.findall(r"\w+", search_query.casefold()) if len(term) > 2]
        matches = []
        for document in documents:
            text = document["text"]
            lowered = text.casefold()
            for term in terms:
                start = 0
                while (position := lowered.find(term, start)) >= 0:
                    matches.append(
                        {
                            "source": document["source"],
                            "page": document.get("page"),
                            "term": term,
                            "excerpt": text[max(0, position - 160) : position + len(term) + 220],
                        }
                    )
                    start = position + len(term)
        st.session_state[f"corpus_matches_{selected_id}"] = matches[:100]
    matches = st.session_state.get(f"corpus_matches_{selected_id}", [])
    if matches:
        st.write(f"Found {len(matches)} matching passage excerpt(s), capped at 100.")
        for match in matches:
            st.markdown(
                f"**{match['source']}**"
                + (
                    f" · page {match['page']}"
                    if match["page"] is not None
                    else ""
                )
                + f" · matched `{match['term']}`"
            )
            st.code(match["excerpt"])
    else:
        st.caption("No search run yet, or no matching terms were found.")
    corpus_search_rationale = st.text_area(
        "Why is the requested answer absent from this corpus?",
        value=case.get("corpus_absence_rationale", ""),
        height=100,
    )

review_notes = st.text_area(
    "Review notes",
    value=case.get("review_notes", ""),
    height=100,
)
correction_reason = st.text_input(
    "Reason for a correction (required for Needs correction)",
    value=case.get("correction_reason", ""),
)

action_columns = st.columns(3)
actions = (
    ("approve", "Approve"),
    ("needs_correction", "Needs correction"),
    ("leave_pending", "Leave pending"),
)
for column, (action, label) in zip(action_columns, actions):
    with column:
        if st.button(label, type="primary" if action == "approve" else "secondary"):
            evidence_update = None
            if is_answerable:
                evidence_update = [
                    dict(item) for item in case["evidence"]
                ]
                evidence_update[evidence_index]["source"] = selected_source
                evidence_update[evidence_index]["anchor"] = evidence_anchor.strip()
                evidence_update[evidence_index]["quotation"] = (
                    evidence_quotation.strip()
                )
            try:
                dataset, record = update_review_case(
                    ACTIVE_DATASET,
                    ACTIVE_REVIEW_LOG,
                    selected_id,
                    action=action,
                    review_notes=review_notes,
                    reviewer=reviewer,
                    reason=correction_reason,
                    expected_answer=expected_answer,
                    evidence=evidence_update,
                    corpus_search_rationale=(
                        corpus_search_rationale if not is_answerable else None
                    ),
                    document_root=CORPUS_ROOT,
                )
            except (OSError, ValueError, KeyError) as error:
                st.error(f"Could not save review decision: {error}")
            else:
                st.session_state.review_case_id = selected_id
                if record["new_status"] == "pending" and record["previous_status"] == "reviewed":
                    st.warning(
                        "This case was approved before its answer/evidence changed. "
                        "Approval was reset to pending; review and approve it again."
                    )
                else:
                    st.success(f"Saved: {record['new_status']}.")
                st.rerun()

st.divider()
if ACTIVE_DATASET == REVIEW_DATASET_V7:
    try:
        dataset_argument = str(ACTIVE_DATASET.relative_to(ROOT))
    except ValueError:
        dataset_argument = str(ACTIVE_DATASET)
    evaluation_command = (
        "python -m scripts.evaluate_retrieval "
        "--corpus-manifest data/benchmark_corpus_v2.json "
        f"--dataset {dataset_argument} "
        "--report reports/retrieval_benchmark_v9_v7_corrections_applied.json "
        "--allow-unreviewed"
    )
    st.markdown(
        "**Provisional diagnostic command** — the corrected labels still need "
        "your explicit approval; `--allow-unreviewed` preserves that warning "
        "in the report."
    )
else:
    evaluation_command = (
        "python -m scripts.evaluate_retrieval "
        "--corpus-manifest data/benchmark_corpus_v2.json "
        "--dataset data/retrieval_evaluation_v5.json "
        "--report reports/retrieval_benchmark_v8_reviewed.json"
    )
    st.markdown("**Reviewed evaluation command**")
st.code(evaluation_command)
