# Local RAG Research Assistant

The local research assistant searches `.txt`, `.md`, `.docx`, and text-based
`.pdf` files, retrieves passages with local embeddings, and can send retrieved
context to a configured OpenAI-compatible API for answer generation. PDF
citations include page numbers. Word citations include filename and chunk;
Word page numbers are not stable across layouts.

## Setup

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Add non-empty `.txt`, `.md`, `.docx` (Word), or `.pdf` files to
   `data/documents/` (subdirectories are searched too). PDFs must contain
   selectable text; scanned/image-only PDFs need OCR before they can be indexed.
   Legacy `.doc` files are not supported; save them as `.docx` first.
   Personal documents in this folder are excluded from Git by default. Only the
   bundled sample Markdown and PDF are included in the repository.

3. Set your API key in the shell. Do not put it in source code:

   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

   The default API URL is `https://api.openai.com/v1` and the default chat model
   is `gpt-4o-mini`. For another OpenAI-compatible service, set
   `OPENAI_BASE_URL` and, if needed, `OPENAI_MODEL`.

## Ask a question

Run from the project directory:

```bash
python main.py --question "What does my research say about retrieval?"
```

## Web app

Start the local browser interface from the project directory:

```bash
streamlit run app.py
```

Start the backend and UI in two terminals:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
streamlit run app.py
```

Streamlit prints `http://localhost:8501`; the API docs are at
`http://localhost:8000/docs`. The UI chooses dense or hybrid retrieval, uploads
documents, rebuilds the backend index, and retrieves or generates answers. The
provider key is configured only in the backend environment, never in the
browser. To enable generation, set `OPENAI_API_KEY` in the shell before
starting Uvicorn; retrieval works without it. The local server is intended for
trusted local use and is not hardened for public exposure.

Omit `--question` to be prompted interactively. Useful options:

```text
--documents-dir PATH   Documents directory (default: data/documents)
--top-k N              Number of passages to retrieve (default: 4)
--retrieval-mode MODE  dense or hybrid (default: dense)
--corpus-manifest PATH Use an explicit document manifest instead of data/documents
--chunk-size N         Maximum chunk size in characters (default: 500)
--chunk-overlap N      Target overlap in characters (default: 100)
--retrieve-only        Print matching passages without calling the chat API
--cache-path PATH      Embedding cache path (default: data/.rag_cache/index.npz)
--model NAME           Chat model (default: OPENAI_MODEL or gpt-4o-mini)
--base-url URL         API base URL (default: OPENAI_BASE_URL or OpenAI)
```

The assistant prints an answer and the retrieved source filenames and chunk
numbers. PDF sources also include the page number for each passage. If the
retrieved passages do not support an answer, it is instructed to say so.

Document chunks and embeddings are cached at `data/.rag_cache/index.npz` after
the first run. The cache is refreshed automatically when document contents,
chunk settings, or the embedding model change.

To inspect local retrieval without making a chat API request, use:

```bash
python main.py --retrieve-only --question "What do my documents say about heat?"
```

The CLI, Streamlit app, and retrieval evaluator use the same initialized
retriever. Dense is the default; choose hybrid explicitly with
`--retrieval-mode hybrid`. Hybrid scores are labeled reciprocal-rank fusion
scores, not cosine similarities.

## Annotation review

Open **Annotation Review** from the Streamlit navigation at
`http://localhost:8501/Annotation_Review`. Select **v7 — user-requested
corrections, awaiting re-review**. The page loads
`data/retrieval_evaluation_v7_corrections_applied.json` directly and writes
decisions to its matching append-only log under `data/review_logs/`. The v5
dataset and its review history remain preserved.

The v7 dataset retains the 114 previously reviewed cases. Eighteen requested
corrections were applied, and two cases were verified as no-change; all 20
affected cases remain pending. Inspect every pending case against the complete
source. Use **Approve** only when the answer and all evidence are supported.
For unanswerable questions, search the full corpus and record why the
information is absent. Choose **Needs correction** to record a reason and
corrected answer/evidence, or **Leave pending** if unresolved. Editing an
approved answer or evidence resets that label to pending. Automated tests use
fixture datasets and do not approve real labels.

Run the provisional diagnostic with:

```bash
python -m scripts.evaluate_retrieval \
  --corpus-manifest data/benchmark_corpus_v2.json \
  --dataset data/retrieval_evaluation_v7_corrections_applied.json \
  --allow-unreviewed \
  --report reports/retrieval_benchmark_v9_v7_corrections_applied.json
```

Only after all 134 cases have been explicitly reviewed, run the final evaluation
without `--allow-unreviewed`:

```bash
python -m scripts.evaluate_retrieval \
  --corpus-manifest data/benchmark_corpus_v2.json \
  --dataset data/retrieval_evaluation_v7_corrections_applied.json \
  --report reports/retrieval_benchmark_v10_reviewed.json
```

The v7 correction history, held-out metrics, exact source ranges, and review
instructions are documented in
[`reports/retrieval_evaluation_v7_corrections_applied.md`](reports/retrieval_evaluation_v7_corrections_applied.md).

## FastAPI backend

Run the API and Streamlit UI in separate terminals:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
streamlit run app.py
```

The UI uses the backend at `http://localhost:8000` by default. API health and
readiness are available at `/health` and `/ready`; interactive OpenAPI
documentation is at `http://localhost:8000/docs`. Retrieval is available
without generation credentials. To enable `/ask` and answer generation, set
`OPENAI_API_KEY` only in the API process environment. Do not expose this local
development server to an untrusted network.

Uploads are limited to supported file types and 20 MiB per file by default.
Files are given server-generated names in the API's document root; use
**Rebuild index** to activate uploaded files. The active index is replaced only
after a complete rebuild, so requests can continue using the previous index
while rebuilding.

### HTTP retrieval timing

The HTTP benchmark includes loopback HTTP, validation/serialization, query
embedding, and retrieval. It excludes model/process startup, index preparation,
generation, and browser rendering. The measurement is not end-to-end answer
latency, and query-embedding time is included rather than separately measured.
Run it against a ready local API with:

```bash
python -m scripts.benchmark_http_retrieval \
  --base-url http://127.0.0.1:8000 \
  --question "How does asyncio.create_task schedule a coroutine?" \
  --top-k 4 --warmups 5 --repetitions 30 \
  --report reports/http_retrieval_latency_v1.json
```

On the local Apple Silicon macOS machine, with the sample index loaded (2
documents, 6 chunks), 30 requests per mode produced dense median/p95 latency of
7.39/8.42 ms and hybrid median/p95 latency of 7.05/8.11 ms. These small-corpus
loopback measurements are a smoke-test result, not a general performance claim.

## Retrieval evaluation

The expanded, shareable benchmark corpus is versioned independently of the
original sample corpus. `data/benchmark_corpus_v2.json` selects 22 distinct
official CPython 3.14 standard-library reference pages; it records each source
URL, retrieval date, relative file path, and SHA-256 checksum. The retrieved
reStructuredText files are under `data/benchmark_documents/`, with the
upstream license notices retained in `LICENSE.python.txt`. Private coursework
is not included.

Run the expanded benchmark commands from the project root, explicitly selecting
the corpus manifest and question dataset:

```bash
python -m scripts.evaluate_retrieval \
  --corpus-manifest data/benchmark_corpus_v2.json \
  --dataset data/retrieval_evaluation_v7_corrections_applied.json \
  --report reports/retrieval_benchmark_v9_v7_corrections_applied.json \
  --allow-unreviewed

python -m scripts.benchmark_index_cache \
  --repetitions 10 \
  --corpus-manifest data/benchmark_corpus_v2.json \
  --dataset data/retrieval_evaluation_v2.json \
  --report reports/index_cache_benchmark_v3.json

python -m scripts.benchmark_retrieval_implementations \
  --corpus-manifest data/benchmark_corpus_v2.json \
  --dataset data/retrieval_evaluation_v4.json \
  --report reports/retrieval_performance_v3.json \
  --repetitions 10

```

The scripts retain the original sample corpus/dataset as their defaults. Every
document in an explicitly selected manifest is indexed, including distractor
documents without labeled answers. To select every supported file under another
directory, use `--documents-dir`; that directory must contain only the corpus
you intend to benchmark. To use another versioned question file, pass
`--dataset PATH`. For example:

```bash
python -m scripts.evaluate_retrieval \
  --corpus-manifest data/my_corpus_manifest.json \
  --dataset data/my_evaluation_v1.json \
  --allow-unreviewed \
  --report reports/retrieval_benchmark_v4_my_corpus.json
python -m scripts.benchmark_index_cache \
  --corpus-manifest data/my_corpus_manifest.json \
  --dataset data/my_evaluation_v1.json \
  --repetitions 10 \
  --report reports/index_cache_benchmark_v3_my_corpus.json
```

`data/retrieval_evaluation_v4.json` and its review/correction files are
preserved historical versions. The current dataset is
`data/retrieval_evaluation_v7_corrections_applied.json`; it contains 114
answerable questions (72 development, 42 held-out) and 20 unanswerable
questions. Its 114 earlier reviewed cases remain unchanged, while 20
correction targets await re-review. Use the **Annotation Review** page and
follow the v7 instructions in
[`reports/retrieval_evaluation_v7_corrections_applied.md`](reports/retrieval_evaluation_v7_corrections_applied.md);
do not edit review statuses directly or rerun the old dataset creator over
manual review work. `--allow-unreviewed` is only for provisional diagnostics.

The retrieval benchmark compares chunk sizes/overlaps and dense vs.
dense+BM25 reciprocal-rank fusion using **development questions only**. It
keeps the 500-character/100-overlap dense baseline unless a candidate strictly
improves the lexicographically ordered development tuple (Hit@5, MRR@10,
Evidence Recall@5) while keeping mean top-five context within 1% of baseline.
The held-out set is used only for final comparison. Results include:

- **Hit@5:** fraction of questions with any labeled evidence in the top five.
- **MRR@10:** mean reciprocal rank of the first labeled evidence, with zero if
  none appears in the first ten.
- **Evidence Recall@5:** labeled evidence items retrieved in the top five
  divided by all labeled evidence items across answerable questions.
- Mean character count of the top five retrieved chunks, to expose context-size
  differences between configurations.
- Mean, median, and p95 retrieval/search latency per question. Query-embedding
  latency, document parsing, index construction, and answer generation are
  outside this metric.

The versioned JSON report at `reports/retrieval_benchmark_v7.json` includes
per-question ranks/latencies, checksums for the **full selected corpus**,
dataset/configuration versions, package versions, hardware, held-out
baseline/final scores, percentage-point deltas, relative changes, and context
length. The implementation comparison at
`reports/retrieval_performance_v3.json` includes ten randomized timing passes,
separate query-embedding/search/total latency, and raw per-query measurements.
Unanswerable questions are counted separately but not scored: abstention and
generated-answer quality are evaluated separately below.
Earlier sample reports and `reports/baseline/` remain preserved. The
intermediate v3-draft runs are retained as
`reports/retrieval_benchmark_v6_intermediate_v3_draft.json` and
`reports/retrieval_performance_v2_intermediate_v3_draft.json`; use v7 and
performance v3 for the finalized dataset v4 comparison.

The cache command runs at least 10 repetitions per condition in randomized
condition order and measures naturally accumulated whole-document subsets
targeting 100 and 1,000 distinct chunks. It does not duplicate text to reach
those sizes and reports unavailable targets when the corpus is too small. It
separates model startup, verifies chunk/embedding and top-five retrieval
equivalence, and reports median, sample standard deviation, min/max, absolute
milliseconds saved, and percent reduction. The timed rebuild includes cache-key
hashing, chunking, embedding, and cache serialization/write; a cache hit
includes key hashing, cache read, and deserialization. **Excluded** are document
discovery/parsing (done once before timing), vector-store construction, query
embedding/search, and LLM generation. This is index-preparation time, not
end-to-end answer speedup. The expanded raw report is
`reports/index_cache_benchmark_v3.json`.

### Historical v3 result and current v7 result

The selected benchmark corpus contains **22 distinct documents**, 22 extracted
text records, and **3,520 distinct chunks** (3,523 indexed chunks, including
exact duplicate chunk text across source pages). The v3 historical question
snapshot contained **114 answerable** (72 development / 42 held-out) and **20
unanswerable** questions across source/topic groups. Its labels and scores are
preserved in the v3 report and are not the current dataset result.

In that historical v3 comparison, the dense baseline scored
**Hit@5 40.5% (17/42), MRR@10 0.284, and Evidence Recall@5 38.6% (17/44
labeled evidence items)**. Dense+BM25 reciprocal-rank fusion, selected using
development questions only, scored **50.0% (21/42), 0.318, and 47.7% (21/44)**.
The provisional changes are **+9.52, +3.45, and +9.09 percentage points**
(relative changes **+23.53%, +12.15%, and +23.53%**). Mean top-five context
increased from **2,046 to 2,091 characters** (+45 characters, +2.2%). That
historical comparison does not supersede the v7 evaluation below.

The current v7 dataset has the same 22-document corpus and **3,523 indexed
chunks**. It contains 114 answerable questions (72 development / 42 held-out)
and 20 unanswerable questions. Of the 20 cases changed or explicitly checked
for this update, all remain pending; the other 114 retain their existing
reviewed status. The v7 evaluation is therefore still a provisional diagnostic.

On the same 42 held-out answerable questions, dense baseline scored **Hit@5
40.5% (17/42), MRR@10 0.315, and Evidence Recall@5 39.1% (18/46 labeled
evidence items)**. Dense+BM25 reciprocal-rank fusion, selected using
development questions only, scored **57.1% (24/42), 0.369, and 54.3% (25/46)**.
Changes were **+16.67, +5.47, and +15.22 percentage points**, respectively.
Mean top-five retrieved context increased from **2,046.26 to 2,091.38
characters** (+45.12, +2.20%). These are provisional retrieval-label metrics,
not answer accuracy; do not treat the gains as fully human-validated.

The application default remains dense. Although the hybrid candidate was
selected from development results, changing the application default still
requires a decision based on reviewed labels and measured latency.

The performance report compares original dense, original hybrid, optimized
dense, and optimized hybrid on the same corpus and v3 questions (22 documents,
3,520 unique chunk texts / 3,523 indexed chunks). The optimized dense and
hybrid methods had identical top-ten rankings to their original counterparts;
maximum dense score difference was 1.2e-7 and hybrid score difference was
zero. Across ten repetitions with randomized method order, search median/p95
was **5.295/5.621 ms original dense vs. 0.204/0.238 ms optimized dense**
(96.1% lower median search time), and **48.311/53.228 ms original hybrid vs.
4.464/5.631 ms optimized hybrid** (90.8% lower). Query embedding was measured
separately at 4.387 ms median in each method; median total retrieval time
(embedding plus search) changed from **9.706 to 4.578 ms dense** and **53.177
to 9.054 ms hybrid**. Precomputing the BM25 index took 47.519 ms, reported
separately from query latency. These are retrieval-label diagnostics on
Python standard-library documentation, not broad-domain RAG claims. All
retrieval quality metrics remain provisional until labels are reviewed.

The cache benchmark used the same model and 500/100 chunk settings, with model
startup measured separately (**2.533 s**). It ran ten rebuild and ten cached
loads per subset, with retrieval equivalence verified:

| Corpus subset | Documents | Distinct chunks | Rebuild median | Cached median | Saved | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| Full selected corpus | 22 | 3,520 | 3,896.575 ms | 17.634 ms | 3,878.941 ms | 99.5% |
| Natural subset reaching 100 | 1 | 278 | 301.716 ms | 1.820 ms | 299.896 ms | 99.4% |
| Natural subset reaching 1,000 | 10 | 1,074 | 1,155.414 ms | 5.706 ms | 1,149.708 ms | 99.5% |

The 100/1,000 targets were reached by including whole documents, so actual
subset counts are reported rather than represented as exact-size samples.
Parsing, vector-store construction, query embedding/search, and LLM generation
are excluded. This is index-preparation savings, not end-to-end latency.
Hardware for both reports: Apple M5 Pro, arm64 macOS, Python 3.14.6. Raw
repetitions, variability, checksums, and full settings are in the JSON reports.

### Answer-quality evaluation

`scripts/evaluate_answer_quality.py` runs the same shared retriever and the
application's fixed system prompt, with temperature set to zero, then saves
answerable and unanswerable examples for human scoring. It records citations,
retrieved passages, expected answers, evidence, and a rubric for correctness,
citation support, appropriate abstention, and unnecessary refusal. The report
does not automatically assign quality scores; review fields stay pending.
Generation could not be run in this environment because `OPENAI_API_KEY` is
not configured. After setting an API key, run (this may incur API charges):

```bash
export OPENAI_API_KEY="your-api-key"
python -m scripts.evaluate_answer_quality \
  --corpus-manifest data/benchmark_corpus_v2.json \
  --dataset data/retrieval_evaluation_v7_corrections_applied.json \
  --retrieval-mode dense \
  --top-k 5 \
  --model gpt-4o-mini \
  --allow-unreviewed \
  --report reports/answer_quality_v2_v7.json
```

The fixed prompt, model, temperature, corpus, dataset, retrieval settings, and
per-question outputs are recorded. The report remains provisional until the
question labels and generated-answer rubric are reviewed by a person. Answer
correctness, citation support, abstention, and unnecessary refusal are
currently unmeasured.

### What is needed for résumé-grade results

The prepared corpus meets the requested document-count range, but it is a
single-source Python standard-library corpus, not a broad research collection.
The next required step is to review the 20 pending v7 cases in the review page
and rerun retrieval evaluation without `--allow-unreviewed`. Live answer-quality
generation also remains unmeasured because no provider key was configured.
Add further sources only for claims that should generalize beyond this
Python-documentation domain.

## Tests

```bash
python -m unittest discover -s tests
```
