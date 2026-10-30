import os
from pathlib import Path

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=120.0, pool=5.0)


st.set_page_config(
    page_title="Local Research Assistant",
    page_icon="📚",
    layout="wide",
)


def api_request(method, endpoint, **kwargs):
    try:
        response = httpx.request(
            method,
            f"{API_BASE_URL}{endpoint}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "The API request timed out. The first request may be downloading "
            "the local embedding model; check API logs and retry."
        ) from error
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Could not reach the backend API at {API_BASE_URL}: {error}"
        ) from error
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")
    return response.json()


def render_passages(passages):
    if not passages:
        st.warning("The backend found no matching passages.")
        return
    st.subheader("Sources and retrieved passages")
    for passage in passages:
        metadata = passage["metadata"]
        page = (
            f", page {metadata['page']}"
            if metadata.get("page") is not None
            else ""
        )
        source = (
            f"{metadata['source']} (chunk {metadata['chunk_id']}{page})"
        )
        with st.expander(
            f"[{passage['rank']}] {source} · {passage['score_type']} "
            f"{passage['score']:.3f}",
            expanded=passage["rank"] == 1,
        ):
            st.write(passage["text"])


st.title("📚 Local Research Assistant")
st.caption(
    "The browser talks only to the local FastAPI backend. Documents and "
    "embeddings are processed by the backend; provider keys never enter this UI."
)

with st.sidebar:
    st.header("Backend")
    st.code(API_BASE_URL)
    if st.button("Refresh API status"):
        st.rerun()
    try:
        health = api_request("GET", "/health")
        readiness = api_request("GET", "/ready")
    except RuntimeError as error:
        st.error(str(error))
        st.warning("Start the API service, then refresh this page.")
        st.stop()
    st.success(f"API {health['status']}")
    if readiness["ready"]:
        st.success(
            f"Index ready · {readiness['document_count']} documents · "
            f"{readiness['chunk_count']} chunks"
        )
        st.caption(f"Index version `{readiness['index_version'][:12]}`")
    else:
        st.warning(readiness["detail"])

    mode = st.radio(
        "Retrieval mode",
        ("dense", "hybrid"),
        help=(
            "Dense scores are cosine similarity. Hybrid scores are reciprocal-rank "
            "fusion scores."
        ),
    )
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=4)
    answer_mode = st.toggle("Generate an answer with the configured backend LLM", False)
    st.caption(
        "LLM credentials are configured only in the API runtime. Retrieval works "
        "without them."
    )

st.subheader("Add documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, Markdown, or text files",
    type=["pdf", "docx", "md", "txt"],
    accept_multiple_files=True,
    max_upload_size=20,
)
upload_col, rebuild_col = st.columns(2)
with upload_col:
    if st.button("Upload selected documents", disabled=not uploaded_files):
        with st.spinner("Uploading documents to the backend..."):
            successes = 0
            for uploaded in uploaded_files or []:
                try:
                    result = api_request(
                        "POST",
                        "/documents",
                        files={
                            "file": (
                                Path(uploaded.name).name,
                                uploaded.getvalue(),
                                uploaded.type or "application/octet-stream",
                            )
                        },
                    )
                except RuntimeError as error:
                    st.error(f"{uploaded.name}: {error}")
                else:
                    successes += 1
                    st.success(
                        f"{Path(uploaded.name).name} stored as "
                        f"{result['stored_name']} ({result['size_bytes']} bytes)."
                    )
            if successes:
                st.info("Rebuild the index to search newly uploaded documents.")
with rebuild_col:
    if st.button("Rebuild index", type="primary"):
        with st.spinner(
            "Building a replacement index. The current index remains available "
            "until the replacement is ready..."
        ):
            try:
                rebuilt = api_request("POST", "/index/rebuild")
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.success(
                    f"Activated index {rebuilt['index_version'][:12]}: "
                    f"{rebuilt['document_count']} documents, "
                    f"{rebuilt['chunk_count']} chunks."
                )
                st.rerun()

st.divider()
with st.form("question_form"):
    question = st.text_input(
        "Ask a question about the indexed documents",
        max_chars=5000,
        placeholder="For example: What does dynamic programming solve?",
    )
    submitted = st.form_submit_button(
        "Ask" if answer_mode else "Retrieve passages",
        type="primary",
        disabled=not readiness["ready"],
    )

if submitted:
    if not question.strip():
        st.error("Enter a question.")
    else:
        endpoint = "/ask" if answer_mode else "/retrieve"
        with st.spinner(
            "Retrieving evidence and generating an answer..."
            if answer_mode
            else "Retrieving passages..."
        ):
            try:
                result = api_request(
                    "POST",
                    endpoint,
                    json={
                        "question": question,
                        "mode": mode,
                        "top_k": top_k,
                    },
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.caption(
                    f"Index version `{result['index_version'][:12]}` · "
                    f"mode `{result['mode']}`"
                )
                if answer_mode:
                    st.subheader("Answer")
                    st.write(result["answer"])
                    st.caption(
                        "Answer citations use passage ranks such as [1]; inspect "
                        "the retrieved source passages below."
                    )
                render_passages(result["passages"])

st.divider()
st.caption(
    f"API documentation: {API_BASE_URL}/docs · Annotation review: use the "
    "Annotation Review page in the navigation menu."
)
