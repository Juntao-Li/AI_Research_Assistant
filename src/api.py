import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from .documents import SUPPORTED_SUFFIXES, load_document_file
from .embedding import _get_model
from .generator import generate_answer
from .indexer import load_or_create_index
from .retriever import Retriever, build_retriever, retrieval_fingerprint


DEFAULT_DOCUMENT_ROOT = Path(
    os.getenv("DOCUMENT_ROOT", "data/uploads")
).expanduser().resolve()
DEFAULT_SAMPLE_ROOT = Path(
    os.getenv("SAMPLE_DOCUMENT_ROOT", "data/sample_documents")
).expanduser().resolve()
DEFAULT_CACHE_ROOT = Path(
    os.getenv("INDEX_CACHE_ROOT", "data/.rag_cache/api")
).expanduser().resolve()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))


class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=5000)
    mode: Literal["dense", "hybrid"] = "dense"
    top_k: int = Field(default=4, ge=1, le=MAX_TOP_K)


class SourceMetadata(BaseModel):
    source: str
    page: int | None = None
    chunk_id: int


class PassageResponse(BaseModel):
    rank: int
    text: str
    metadata: SourceMetadata
    score: float
    score_type: str


class RetrieveResponse(BaseModel):
    question: str
    mode: Literal["dense", "hybrid"]
    index_version: str
    passages: list[PassageResponse]


class AskResponse(RetrieveResponse):
    answer: str
    citations: list[SourceMetadata]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    ready: bool
    document_count: int
    chunk_count: int
    index_version: str | None = None
    detail: str


class UploadResponse(BaseModel):
    stored_name: str
    size_bytes: int
    message: str


class RebuildResponse(BaseModel):
    ready: bool
    document_count: int
    chunk_count: int
    index_version: str
    message: str


@dataclass(frozen=True)
class IndexSnapshot:
    retriever: Retriever
    version: str
    document_count: int
    chunk_count: int


class AssistantRuntime:
    def __init__(
        self,
        document_root=DEFAULT_DOCUMENT_ROOT,
        sample_root=DEFAULT_SAMPLE_ROOT,
        cache_root=DEFAULT_CACHE_ROOT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        api_key=None,
        base_url=None,
        model=None,
        max_upload_bytes=MAX_UPLOAD_BYTES,
    ):
        self.document_root = Path(document_root).resolve()
        self.sample_root = Path(sample_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_upload_bytes = max_upload_bytes
        self._snapshot: IndexSnapshot | None = None
        self._rebuild_lock = asyncio.Lock()
        self._last_error: str | None = None
        self.document_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot(self):
        return self._snapshot

    @property
    def last_error(self):
        return self._last_error

    def _read_documents(self):
        documents = []
        seen_sources = set()
        for root in (self.sample_root, self.document_root):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                source = path.relative_to(root).as_posix()
                if source in seen_sources:
                    source = f"{root.name}/{source}"
                seen_sources.add(source)
                documents.extend(load_document_file(path, source))
        return documents

    def _build_snapshot_sync(self):
        documents = self._read_documents()
        if not documents:
            raise ValueError(
                "No supported documents are available. Upload a PDF, DOCX, "
                "Markdown, or text document, then rebuild the index."
            )
        corpus_signature = retrieval_fingerprint(
            documents, self.chunk_size, self.chunk_overlap
        )
        cache_path = self.cache_root / f"{corpus_signature}.npz"
        chunks, embeddings, metadata = load_or_create_index(
            documents,
            self.chunk_size,
            self.chunk_overlap,
            cache_path,
        )
        retriever = build_retriever(chunks, embeddings, metadata)
        return IndexSnapshot(
            retriever=retriever,
            version=corpus_signature,
            document_count=len({item["source"] for item in documents}),
            chunk_count=len(chunks),
        )

    async def rebuild(self):
        async with self._rebuild_lock:
            try:
                replacement = await asyncio.to_thread(self._build_snapshot_sync)
            except (OSError, RuntimeError, ValueError) as error:
                self._last_error = str(error)
                raise
            self._snapshot = replacement
            self._last_error = None
            return replacement

    def retrieve_sync(self, request: QuestionRequest):
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("The retrieval index is not ready. Rebuild it first.")
        results = snapshot.retriever.search(
            request.question.strip(),
            top_k=request.top_k,
            mode=request.mode,
        )
        passages = [
            PassageResponse(
                rank=rank,
                text=result["text"],
                metadata=SourceMetadata(**result["metadata"]),
                score=float(result["score"]),
                score_type=result["score_type"],
            )
            for rank, result in enumerate(results, start=1)
        ]
        return snapshot, passages

    def ask_sync(self, request: QuestionRequest):
        if not self.api_key:
            raise PermissionError(
                "Answer generation is not configured. Set OPENAI_API_KEY in the "
                "backend runtime environment; retrieval remains available."
            )
        snapshot, passages = self.retrieve_sync(request)
        context = "\n\n".join(
            f"[{passage.rank}] Source: {passage.metadata.source} "
            f"(chunk {passage.metadata.chunk_id}"
            f"{', page ' + str(passage.metadata.page) if passage.metadata.page else ''})\n"
            f"{passage.text}"
            for passage in passages
        )
        answer = generate_answer(
            question=request.question.strip(),
            context=context,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )
        return snapshot, passages, answer

    async def save_upload(self, upload: UploadFile):
        original_name = Path(upload.filename or "").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                "Unsupported upload type. Supported types are PDF, DOCX, "
                "Markdown, and plain text."
            )
        temporary_path = None
        total_size = 0
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.document_root,
                prefix=".upload-",
                suffix=suffix,
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                while chunk := await upload.read(1024 * 1024):
                    total_size += len(chunk)
                    if total_size > self.max_upload_bytes:
                        raise ValueError(
                            f"Upload exceeds the {self.max_upload_bytes}-byte limit."
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total_size == 0:
                raise ValueError("Uploaded document is empty.")
            await asyncio.to_thread(
                load_document_file, temporary_path, temporary_path.name
            )
            stored_name = f"{uuid.uuid4().hex}{suffix}"
            destination = self.document_root / stored_name
            os.replace(temporary_path, destination)
            temporary_path = None
            return stored_name, total_size
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            await upload.close()


def get_runtime(request: Request) -> AssistantRuntime:
    return request.app.state.runtime


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse, tags=["service"])
def ready(request: Request):
    runtime = get_runtime(request)
    snapshot = runtime.snapshot
    if snapshot is None:
        return ReadinessResponse(
            ready=False,
            document_count=0,
            chunk_count=0,
            index_version=None,
            detail=runtime.last_error or "No retrieval index has been built yet.",
        )
    return ReadinessResponse(
        ready=True,
        document_count=snapshot.document_count,
        chunk_count=snapshot.chunk_count,
        index_version=snapshot.version,
        detail="Retrieval index is ready.",
    )


@router.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
async def retrieve(request: QuestionRequest, http_request: Request):
    runtime = get_runtime(http_request)
    try:
        snapshot, passages = await asyncio.to_thread(runtime.retrieve_sync, request)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return RetrieveResponse(
        question=request.question.strip(),
        mode=request.mode,
        index_version=snapshot.version,
        passages=passages,
    )


@router.post("/ask", response_model=AskResponse, tags=["generation"])
async def ask(request: QuestionRequest, http_request: Request):
    runtime = get_runtime(http_request)
    if not runtime.api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Answer generation is not configured. Set OPENAI_API_KEY in "
                "backend runtime configuration; /retrieve remains available."
            ),
        )
    try:
        snapshot, passages, answer = await asyncio.to_thread(
            runtime.ask_sync, request
        )
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return AskResponse(
        question=request.question.strip(),
        mode=request.mode,
        index_version=snapshot.version,
        passages=passages,
        answer=answer,
        citations=[passage.metadata for passage in passages],
    )


@router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=201,
    tags=["documents"],
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    runtime = get_runtime(request)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > runtime.max_upload_bytes + 1024 * 1024:
                raise HTTPException(status_code=413, detail="Upload is too large.")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length.") from error
    try:
        stored_name, size_bytes = await runtime.save_upload(file)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not store uploaded document.") from error
    return UploadResponse(
        stored_name=stored_name,
        size_bytes=size_bytes,
        message="Document uploaded. Rebuild the index to make it searchable.",
    )


@router.post("/index/rebuild", response_model=RebuildResponse, tags=["index"])
async def rebuild_index(request: Request):
    runtime = get_runtime(request)
    try:
        snapshot = await runtime.rebuild()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed; the previous index remains active: {error}",
        ) from error
    return RebuildResponse(
        ready=True,
        document_count=snapshot.document_count,
        chunk_count=snapshot.chunk_count,
        index_version=snapshot.version,
        message="Index rebuilt and activated.",
    )


def create_app(
    runtime_factory=AssistantRuntime,
    *,
    load_embedding_model=True,
    rebuild_on_startup=True,
):
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime = runtime_factory()
        application.state.runtime = runtime
        if load_embedding_model:
            await asyncio.to_thread(_get_model)
        if rebuild_on_startup:
            try:
                await runtime.rebuild()
            except ValueError:
                # Empty document stores are valid; the UI can upload first.
                pass
        yield

    application = FastAPI(
        title="Local Research Assistant API",
        description=(
            "Local document retrieval and grounded answer generation. Dense "
            "and hybrid retrieval are available without generation credentials."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
