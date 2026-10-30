import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "benchmark_corpus_v2.json"
DATASET_PATH = ROOT / "data" / "retrieval_evaluation_v2.json"
REVIEW_PATH = ROOT / "data" / "retrieval_evaluation_v2_review.md"


# Each anchor is a literal phrase from its primary-source document. The builder
# extracts the containing paragraph so reviewers can verify the full context.
FACTS = {
    "argparse.txt": [
        ("What object is the central container for an argparse command-line interface?", "The ArgumentParser instance contains the command-line argument specifications.", "container for"),
        ("How does argparse describe the role of ArgumentParser in a command-line interface?", "An ArgumentParser instance is the container for argument specifications.", "instance of :class:`argparse.ArgumentParser`"),
        ("What can the add_argument method define for a command-line parser?", "It defines the individual arguments that the parser accepts.", "add_argument"),
        ("How can a parser restrict an argument to a finite set of values?", "Use the choices argument to restrict the accepted values.", "choices"),
        ("Which parser method turns the command-line arguments into parsed values?", "Use parse_args() to parse the command line and return the resulting namespace.", "parse_args"),
    ],
    "asyncio_tasks.txt": [
        ("What does asyncio.run() do for a top-level coroutine?", "It runs the coroutine and manages the event loop for that top-level operation.", "asyncio.run"),
        ("How can an asyncio coroutine be scheduled to run as a Task?", "Use asyncio.create_task() to schedule it on the running event loop.", "create_task"),
        ("What does TaskGroup provide for related asynchronous tasks?", "TaskGroup provides structured concurrency for a group of tasks.", "TaskGroup"),
        ("How does asyncio.gather() return successful results from multiple awaitables?", "It returns their results in the order of the original awaitables.", "gather"),
        ("What must a coroutine do to let other tasks run while it waits?", "It must await an operation that suspends, allowing the event loop to run other tasks.", "await"),
    ],
    "asyncio_streams.txt": [
        ("What pair of objects does asyncio.open_connection() return?", "It returns a StreamReader and StreamWriter pair.", "open_connection"),
        ("Which stream object is used to read incoming data?", "StreamReader reads data from an asynchronous stream.", "StreamReader"),
        ("How should an asyncio stream writer be closed cleanly?", "Call close() and then await wait_closed().", "wait_closed"),
        ("What does the limit parameter of stream APIs control?", "It sets the stream reader's buffer limit.", "limit"),
        ("What can asyncio.start_server() create?", "It starts a TCP server that invokes a callback for each client connection.", "start_server"),
    ],
    "asyncio_synchronization.txt": [
        ("What is an asyncio.Lock used to protect?", "It provides mutual exclusion among asyncio tasks.", "Lock"),
        ("How does an asyncio.Event notify waiting tasks?", "Calling set() makes the event set and wakes tasks waiting for it.", "Event"),
        ("What does an asyncio.Semaphore limit?", "It limits the number of tasks that can access a resource concurrently.", "Semaphore"),
        ("Are asyncio synchronization primitives safe to use from multiple OS threads?", "No. asyncio synchronization primitives are not thread-safe.", "not thread-safe"),
        ("How can a coroutine wait for an operation only up to a time limit?", "Use asyncio.wait_for() to apply a timeout to an awaitable.", "wait_for"),
    ],
    "asyncio_queues.txt": [
        ("What is the default capacity of an asyncio.Queue?", "A queue with maxsize zero or less has unlimited capacity.", "maxsize"),
        ("What happens when put() is called on a full bounded asyncio queue?", "The coroutine waits until an item is removed and space becomes available.", "put"),
        ("What does Queue.join() wait for?", "It waits until every enqueued item has been marked as processed with task_done().", "join"),
        ("What does qsize() report?", "It reports the number of items currently in the queue.", "qsize"),
        ("Do asyncio queue methods accept a timeout parameter directly?", "No. Apply a timeout using asyncio.wait_for() around the queue operation.", "asyncio queues don't have a *timeout* parameter"),
    ],
    "collections.txt": [
        ("What does collections.Counter count?", "It counts hashable objects and stores their counts.", "Counter"),
        ("How does defaultdict handle a missing key?", "It calls its default_factory to create and insert a default value.", "default_factory"),
        ("Why use deque instead of a list for frequent operations at both ends?", "A deque supports fast, memory-efficient appends and pops at either end.", "deque"),
        ("What does namedtuple add to a tuple?", "It creates a tuple subclass whose fields can be accessed by name.", "namedtuple"),
        ("When several mappings are combined with ChainMap, where do writes go?", "Writes and updates affect only the first mapping in the chain.", "ChainMap"),
    ],
    "concurrent_futures.txt": [
        ("What high-level programming interface does concurrent.futures provide?", "It provides an interface for asynchronously executing callables.", "asynchronously executing"),
        ("What does Executor.submit() return?", "It returns a Future representing the scheduled call.", "submit"),
        ("What does Future.result() do before returning a result?", "It waits for the call to finish and then returns its result or raises its exception.", "result"),
        ("What is the main difference between ThreadPoolExecutor and ProcessPoolExecutor?", "ThreadPoolExecutor uses threads, while ProcessPoolExecutor uses separate processes.", "ProcessPoolExecutor"),
        ("What restriction applies to callables and values used with ProcessPoolExecutor?", "They must be picklable so they can be transferred to worker processes.", "picklable"),
    ],
    "csv.txt": [
        ("What type of values does csv.reader normally return for each row?", "It returns each row as a list of strings.", "list of strings"),
        ("How does csv.DictReader represent a row?", "It maps field names to values for each row.", "DictReader"),
        ("Why does the csv documentation recommend opening files with newline=''?", "It lets the csv module handle newline processing itself and avoids extra newline translation.", "newline=''"),
        ("What does csv.writer.writerow() write?", "It writes a single row of fields to the CSV output.", "writerow"),
        ("How does csv.DictWriter determine which fields to output?", "Its fieldnames sequence defines the output columns and their order.", "fieldnames"),
    ],
    "dataclasses.txt": [
        ("What does the dataclass decorator generate from annotated fields?", "It can generate methods such as __init__ and __repr__ from the declared fields.", "adding generated"),
        ("How should a dataclass field create a fresh mutable default per instance?", "Use field(default_factory=...) to create a new value for each instance.", "default_factory"),
        ("What effect does frozen=True have on a dataclass instance?", "It prevents normal assignment to its fields after initialization.", "frozen"),
        ("What does dataclasses.asdict() produce?", "It recursively converts a dataclass instance into a dictionary.", "asdict"),
        ("Which annotations does dataclasses use to identify fields?", "It uses annotations on the class to identify declared fields.", "annotations"),
    ],
    "json.txt": [
        ("Which function converts a Python object into a JSON-formatted string?", "json.dumps() serializes a Python object to a JSON string.", "json.dumps"),
        ("Which function reads JSON text from a file-like object?", "json.load() deserializes JSON from a file-like object.", "json.load"),
        ("What Python value represents JSON null after decoding?", "JSON null is decoded as Python None.", "null"),
        ("What does the sort_keys option change when encoding a dictionary?", "It sorts dictionary keys in the encoded JSON output.", "sort_keys"),
        ("Can JSON represent tuples as a distinct data type?", "No. Tuples are encoded as JSON arrays and decode as lists.", "tuple"),
    ],
    "logging.txt": [
        ("How can separate Python modules participate in the same application log?", "Modules can obtain named loggers that propagate records through the logging hierarchy.", "all Python modules can participate in logging"),
        ("What does logging.basicConfig() configure?", "It provides a simple way to configure the root logger and its handlers.", "basicConfig"),
        ("Which logging level is more severe: WARNING or ERROR?", "ERROR is more severe than WARNING.", "ERROR"),
        ("What is a logging Handler responsible for?", "A handler sends log records to a destination such as a stream or file.", "Handler"),
        ("What is the recommended way to include variable data in a logging message?", "Pass a format string and its arguments separately to the logging call.", "format string"),
    ],
    "pathlib.txt": [
        ("What does pathlib.Path represent?", "Path represents a concrete filesystem path with operations that access the filesystem.", "class:`Path`"),
        ("How do pure paths differ from concrete Path objects?", "Pure paths perform path manipulation without filesystem access.", "PurePath"),
        ("Which Path method finds files matching a pattern beneath a directory?", "Path.glob() yields paths matching a glob pattern.", "glob"),
        ("What does Path.read_text() return?", "It reads a file and returns its contents as text.", "read_text"),
        ("How can a path be combined with a child path using pathlib?", "Use the / operator to join path components.", " / "),
    ],
    "regular_expressions.txt": [
        ("What does re.search() do?", "It scans through a string looking for the first location that matches a pattern.", "search"),
        ("How does re.match() differ from re.search()?", "re.match() checks for a match only at the beginning of the string.", "match"),
        ("What does the DOTALL flag change for the dot metacharacter?", "With DOTALL, a dot matches newline characters as well.", "DOTALL"),
        ("What does re.compile() provide?", "It compiles a regular expression pattern into a reusable pattern object.", "re.compile"),
        ("What is the purpose of a named group in a regular expression?", "A named group lets matched text be retrieved using its assigned name.", "?P<"),
    ],
    "shutil.txt": [
        ("What does shutil.copytree() copy?", "It recursively copies a directory tree to a destination directory.", "copytree"),
        ("How does shutil.move() behave when moving a path?", "It moves a file or directory, using rename when possible and copy-then-remove otherwise.", "move"),
        ("What information does shutil.disk_usage() return?", "It reports total, used, and free space for a filesystem path.", "disk_usage"),
        ("Does shutil.copy2() guarantee that every filesystem metadata field is preserved?", "No. Some metadata cannot be copied even by copy2().", "metadata"),
        ("What is shutil.which() used to find?", "It locates an executable by searching the directories in the system path.", "which"),
    ],
    "sqlite3.txt": [
        ("Which database interface standard does Python's sqlite3 module follow?", "The module provides a DB-API 2.0 interface to SQLite.", "DB-API 2.0 interface"),
        ("How should values be supplied to an SQLite query to avoid string-built SQL?", "Use placeholders in the SQL statement and pass values separately as parameters.", "placeholder"),
        ("What does Connection.row_factory control?", "It determines how rows returned by a connection are represented.", "row_factory"),
        ("What does the Connection context manager do for a transaction?", "It commits a successful transaction and rolls back if an exception occurs.", "context manager"),
        ("Can sqlite3 create a backup while the database is in use?", "Yes. The backup API can copy a database while it is being accessed.", "backup"),
    ],
    "subprocess.txt": [
        ("What is the recommended high-level function for running a subprocess?", "subprocess.run() is the recommended high-level interface for many subprocess use cases.", "subprocess.run"),
        ("What does check=True do in subprocess.run()?", "It raises CalledProcessError when the process exits with a non-zero status.", "check=True"),
        ("How can subprocess.run() capture a child's standard output?", "Set capture_output=True or direct stdout to PIPE.", "capture_output"),
        ("What risk can arise from using shell=True with untrusted input?", "It can expose the application to shell injection unless the input is safely handled.", "shell=True"),
        ("How can a caller limit how long it waits for a subprocess?", "Pass a timeout; the call raises TimeoutExpired if the limit is exceeded.", "timeout"),
    ],
    "tempfile.txt": [
        ("What does TemporaryDirectory provide?", "It creates a temporary directory that is cleaned up when the context ends or the object is cleaned up.", "TemporaryDirectory"),
        ("What is the purpose of NamedTemporaryFile?", "It creates a temporary file with a visible filesystem name.", "NamedTemporaryFile"),
        ("How can a caller choose the directory used for a temporary file?", "Pass the desired directory using the dir parameter.", "dir="),
        ("Why is mktemp() unsafe for creating temporary files?", "It returns a name without atomically creating the file, leaving a race condition.", "mktemp"),
        ("How can temporary files be cleaned up reliably?", "Use the provided context manager so cleanup occurs when the block exits.", "context manager"),
    ],
    "traceback.txt": [
        ("What does traceback.print_exc() print?", "It prints the traceback for the most recently handled exception.", "print_exc"),
        ("What is TracebackException designed to represent?", "It captures exception information in a form that can be formatted without retaining the original traceback objects.", "TracebackException"),
        ("How can traceback output include chained exceptions?", "The formatting functions include exception chaining by default unless chain is disabled.", "chain"),
        ("What does traceback.format_exc() return?", "It returns the current exception's traceback as a formatted string.", "format_exc"),
        ("Can traceback utilities print a stack even when no exception is active?", "Yes. Stack-printing helpers can display the current call stack without an active exception.", "print_stack"),
    ],
    "typing.txt": [
        ("Does Python enforce type annotations at runtime?", "No. Type annotations are not enforced by the Python runtime.", "The Python runtime does not enforce function"),
        ("What does TypeVar let a type annotation express?", "It introduces a type variable so a function or class can relate types across inputs and outputs.", "TypeVar"),
        ("What is a Protocol used to describe?", "A Protocol describes a structural interface based on the operations an object supports.", "Protocol"),
        ("What does typing.Optional[T] mean?", "It means a value can be T or None.", "Optional"),
        ("How does the | operator express a union type in modern Python annotations?", "It combines types into a union, such as int | str.", "Union"),
    ],
    "unittest.txt": [
        ("What is a unittest.TestCase?", "It is the base class used to define individual test cases and their assertions.", "TestCase"),
        ("When does setUp() run?", "setUp() runs before each test method in the test case.", "setUp"),
        ("When does tearDown() run?", "tearDown() runs after each test method, including when the test raises an error.", "tearDown"),
        ("What does unittest discovery look for by default?", "It searches for test modules matching the test*.py pattern.", "test*.py"),
        ("What is the difference between assertEqual and assertIs?", "assertEqual checks equality; assertIs checks that two references are the same object.", "assertIs"),
    ],
    "urllib_request.txt": [
        ("What does urllib.request.urlopen() return for a successful request?", "It returns a response object that can be read like a file.", "urlopen"),
        ("Why create a urllib.request.Request object instead of passing only a URL?", "A Request can carry details such as headers, method, and request data.", "Request"),
        ("How can urllib.request send data with an HTTP request?", "Pass encoded data in the request data argument; this typically makes the request a POST.", "data"),
        ("What is an opener in urllib.request?", "An opener combines handlers that customize how URLs are opened.", "OpenerDirector"),
        ("How can a caller specify a custom HTTP header for a request?", "Add the header to a Request object before opening it.", "add_header"),
    ],
    "venv.txt": [
        ("How does the documentation characterize a virtual environment?", "It is considered disposable and should be straightforward to recreate.", "Considered as disposable"),
        ("Should a virtual environment be moved or copied to another location?", "No. Environments are generally non-portable and should be recreated at the new location.", "non-portable"),
        ("What does the --system-site-packages option allow?", "It gives the environment access to packages installed in the base Python environment.", "--system-site-packages"),
        ("What does the --clear option do when creating an environment?", "It removes the contents of an existing target environment directory before creating the environment.", "--clear"),
        ("How can a virtual environment be recreated consistently?", "Record its dependencies and recreate it using the intended Python interpreter and those dependencies.", "recreate"),
    ],
}


GROUPS = {
    "argparse.txt": ("command-line-parsing", "development"),
    "asyncio_tasks.txt": ("asyncio", "held-out-test"),
    "asyncio_streams.txt": ("asyncio", "held-out-test"),
    "asyncio_synchronization.txt": ("asyncio", "held-out-test"),
    "asyncio_queues.txt": ("asyncio", "held-out-test"),
    "collections.txt": ("containers", "development"),
    "concurrent_futures.txt": ("concurrency", "development"),
    "csv.txt": ("structured-data", "held-out-test"),
    "dataclasses.txt": ("records", "development"),
    "json.txt": ("structured-data", "held-out-test"),
    "logging.txt": ("observability", "development"),
    "pathlib.txt": ("filesystem", "development"),
    "regular_expressions.txt": ("text-patterns", "development"),
    "shutil.txt": ("filesystem", "development"),
    "sqlite3.txt": ("sqlite", "held-out-test"),
    "subprocess.txt": ("processes", "development"),
    "tempfile.txt": ("filesystem", "development"),
    "traceback.txt": ("observability", "development"),
    "typing.txt": ("type-system", "development"),
    "unittest.txt": ("testing", "held-out-test"),
    "urllib_request.txt": ("http-client", "development"),
    "venv.txt": ("environments", "development"),
}


MULTI_EVIDENCE = [
    {
        "id": "cross-async-q1",
        "question": "How could an asynchronous application read a network stream and hand work to a bounded queue?",
        "expected_answer": "Read incoming data with a stream reader, then await putting work into a bounded queue; the queue operation waits when it is full.",
        "group": "cross-source-async",
        "split": "held-out-test",
        "sources": [
            ("asyncio_streams.txt", "StreamReader"),
            ("asyncio_queues.txt", "put"),
        ],
    },
    {
        "id": "cross-files-q1",
        "question": "How can pathlib locate matching files before shutil copies a directory tree?",
        "expected_answer": "Use Path.glob() to find matching paths, then shutil.copytree() to recursively copy a directory tree.",
        "group": "cross-source-files",
        "split": "development",
        "sources": [
            ("pathlib.txt", "glob"),
            ("shutil.txt", "copytree"),
        ],
    },
    {
        "id": "cross-data-q1",
        "question": "Which standard-library tools can read row-oriented CSV data and deserialize JSON from a file?",
        "expected_answer": "Use csv.DictReader to map CSV field names to values, and json.load() to deserialize JSON from a file-like object.",
        "group": "cross-source-data",
        "split": "held-out-test",
        "sources": [
            ("csv.txt", "DictReader"),
            ("json.txt", "json.load"),
        ],
    },
    {
        "id": "cross-process-q1",
        "question": "How can a Python program isolate temporary files while running an external command?",
        "expected_answer": "Create a TemporaryDirectory for temporary files and use subprocess.run() for the external command.",
        "group": "cross-source-process",
        "split": "development",
        "sources": [
            ("tempfile.txt", "TemporaryDirectory"),
            ("subprocess.txt", "subprocess.run"),
        ],
    },
]


UNANSWERABLE = [
    ("Which PyTorch API moves a tensor onto a CUDA device?", "pytorch", "development"),
    ("How does PyTorch automatically differentiate a tensor computation graph?", "pytorch", "development"),
    ("Which scikit-learn estimator should be used for multiclass logistic regression?", "scikit-learn", "development"),
    ("How should a pandas DataFrame perform a groupby aggregation?", "pandas", "development"),
    ("What is the syntax for defining a Kubernetes liveness probe in a Pod manifest?", "kubernetes", "development"),
    ("How does Kubernetes roll back a failed Deployment rollout?", "kubernetes", "development"),
    ("Which PostgreSQL index type is best for a JSONB containment query?", "postgresql", "development"),
    ("How do you create a Docker multi-stage build for a Python web service?", "docker", "development"),
    ("What is the syntax for registering a GitHub Actions matrix strategy?", "github-actions", "development"),
    ("How does Terraform import an existing cloud resource into state?", "terraform", "development"),
    ("What is the latest stable version of the Rust compiler?", "rust", "held-out-test"),
    ("How does Rust's borrow checker infer a lifetime for a returned reference?", "rust", "held-out-test"),
    ("Which Go package provides goroutine channels and select statements?", "golang", "held-out-test"),
    ("How should a React component use the useEffect hook to subscribe to events?", "react", "held-out-test"),
    ("Which FastAPI decorator declares a path parameter constraint?", "fastapi", "held-out-test"),
    ("How do you configure a Redis cluster for automatic failover?", "redis", "held-out-test"),
    ("What command creates an AWS S3 bucket with versioning enabled?", "aws", "held-out-test"),
    ("How should an OpenAI API request specify a JSON response format?", "openai-api", "held-out-test"),
    ("Which vector database supports HNSW indexing with tenant-level namespaces?", "vector-databases", "held-out-test"),
    ("What is the measured Recall@10 of this application's embedding index?", "project-benchmark-results", "held-out-test"),
]


def normalize(text):
    return " ".join(text.split())


def paragraph_containing(text, anchor):
    normalized_text = normalize(text)
    normalized_anchor = normalize(anchor)
    if normalized_anchor.casefold() not in normalized_text.casefold():
        raise ValueError(f"Anchor not found: {anchor!r}")
    original_index = text.casefold().find(anchor.casefold())
    if original_index < 0:
        # Fall back to whitespace-normalized matching for RST line wraps.
        collapsed = re.sub(r"\s+", " ", text)
        collapsed_index = collapsed.casefold().find(normalized_anchor.casefold())
        if collapsed_index < 0:
            raise ValueError(f"Anchor not found after whitespace folding: {anchor!r}")
        return normalized_text[max(0, collapsed_index - 300):collapsed_index + 700]
    start = text.rfind("\n\n", 0, original_index)
    end = text.find("\n\n", original_index)
    paragraph = text[start + 2 if start >= 0 else 0:end if end >= 0 else len(text)]
    return normalize(paragraph)


def stable_anchor_excerpt(text, phrase):
    normalized_text = normalize(text)
    normalized_phrase = normalize(phrase)
    position = normalized_text.casefold().find(normalized_phrase.casefold())
    if position < 0:
        raise ValueError(f"Anchor phrase not found: {phrase!r}")
    start = max(0, position - 35)
    end = min(len(normalized_text), position + len(normalized_phrase) + 35)
    if start:
        start = normalized_text.rfind(" ", 0, start) + 1
    following_space = normalized_text.find(" ", end)
    if following_space >= 0:
        end = following_space
    return normalized_text[start:end]


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    text_by_source = {
        entry["source"]: (ROOT / "data" / entry["path"]).read_text(encoding="utf-8")
        for entry in manifest["documents"]
    }
    missing_anchors = [
        (source, anchor)
        for source, facts in FACTS.items()
        for _, _, anchor in facts
        if normalize(anchor).casefold()
        not in normalize(text_by_source[source]).casefold()
    ]
    if missing_anchors:
        raise ValueError(f"Anchors not found in corpus: {missing_anchors}")
    answerable = []
    for source, facts in FACTS.items():
        group, split = GROUPS[source]
        for index, (question, answer, anchor) in enumerate(facts, start=1):
            evidence_id = f"{source.removesuffix('.txt')}-e{index}"
            quote = paragraph_containing(text_by_source[source], anchor)
            stable_anchor = stable_anchor_excerpt(quote, anchor)
            answerable.append(
                {
                    "id": f"{source.removesuffix('.txt')}-q{index}",
                    "question": question,
                    "expected_answer": answer,
                    "group": group,
                    "split": split,
                    "question_style": "fact-retrieval",
                    "review_status": "pending",
                    "evidence": [
                        {
                            "id": evidence_id,
                            "source": source,
                            "page": None,
                            "anchor": stable_anchor,
                            "quotation": quote,
                        }
                    ],
                }
            )

    for case in MULTI_EVIDENCE:
        evidence = []
        for index, (source, anchor) in enumerate(case["sources"], start=1):
            quote = paragraph_containing(text_by_source[source], anchor)
            stable_anchor = stable_anchor_excerpt(quote, anchor)
            evidence.append(
                {
                    "id": f"{case['id']}-e{index}",
                    "source": source,
                    "page": None,
                    "anchor": stable_anchor,
                    "quotation": quote,
                }
            )
        answerable.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "group": case["group"],
                "split": case["split"],
                "question_style": "multi-evidence-synthesis",
                "review_status": "pending",
                "evidence": evidence,
            }
        )

    unanswerable = [
        {
            "id": f"unanswerable-{index:02}",
            "question": question,
            "group": f"absent-{group}",
            "split": split,
            "review_status": "pending",
            "corpus_absence_rationale": (
                f"The selected corpus consists of Python standard-library "
                f"documentation and contains no {group} documentation or "
                "project-specific benchmark answer."
            ),
        }
        for index, (question, group, split) in enumerate(UNANSWERABLE, start=1)
    ]

    dataset = {
        "dataset_version": "2.0",
        "corpus_manifest": "benchmark_corpus_v2.json",
        "label_origin": "assistant-drafted; all labels require human review",
        "review_instructions": (
            "For every answerable case, verify the answer against the exact "
            "quotation and source; confirm the anchor occurs in the cited "
            "document; correct or remove inaccurate cases. For unanswerable "
            "cases, search the full selected corpus and confirm the information "
            "is absent. Only after personally completing review, change that "
            "case's review_status from pending to reviewed. Keep all topic-group "
            "splits unchanged; do not tune on held-out-test questions."
        ),
        "answerable": answerable,
        "unanswerable": unanswerable,
    }
    DATASET_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Evaluation dataset v2.0 — human review required",
        "",
        f"Drafted labels: {len(answerable)} answerable; {len(unanswerable)} unanswerable.",
        "Every label is pending. No question counts as human-reviewed until you personally verify it.",
        "",
        "## How to review",
        "",
        "1. Open `retrieval_evaluation_v2.json` and inspect each answerable question, expected answer, quotation, and source.",
        "2. Confirm the quotation is exact and supports the expected answer; check the full source document, not just the extracted excerpt.",
        "3. Search the selected corpus for each unanswerable question's subject and confirm the requested fact is absent.",
        "4. Correct or delete bad cases. After personally checking a case, change only its `review_status` from `pending` to `reviewed`.",
        "5. Preserve each `group` and `split`; use development questions for tuning and held-out-test only for final reporting.",
        "6. Run the benchmark without `--allow-unreviewed` after all included labels are reviewed.",
        "",
        "## Answerable questions",
        "",
    ]
    for case in answerable:
        evidence = case["evidence"][0]
        lines.extend(
            [
                f"### {case['id']} [{case['split']}; {case['group']}; pending]",
                f"- **Question:** {case['question']}",
                f"- **Expected answer:** {case['expected_answer']}",
            ]
        )
        for evidence in case["evidence"]:
            lines.extend(
                [
                    f"- **Evidence:** `{evidence['source']}` (text record; no page number), anchor `{evidence['anchor']}`",
                    f"- **Exact source quotation:** “{evidence['quotation']}”",
                ]
            )
        lines.append("")
    lines.extend(["## Unanswerable questions", ""])
    for case in unanswerable:
        lines.extend(
            [
                f"### {case['id']} [{case['split']}; {case['group']}; pending]",
                f"- **Question:** {case['question']}",
                f"- **Absence rationale:** {case['corpus_absence_rationale']}",
                "",
            ]
        )
    REVIEW_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(answerable)} answerable and {len(unanswerable)} "
        f"unanswerable drafts; all review statuses are pending."
    )


if __name__ == "__main__":
    main()
