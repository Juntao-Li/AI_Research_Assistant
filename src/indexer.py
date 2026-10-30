import hashlib
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .chunker import chunk_text
from .embedding import EMBEDDING_MODEL, embed_texts


_CACHE_VERSION = 1


def _cache_key(documents, chunk_size, chunk_overlap):
    contents = {
        "version": _CACHE_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": documents,
    }
    serialized = json.dumps(
        contents, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_cache(cache_path, cache_key):
    if not cache_path.is_file():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            version = int(cached["version"][0])
            stored_key = str(cached["key"][0])
            chunks = cached["chunks"].tolist()
            embeddings = cached["embeddings"]
            metadata_bytes = cached["metadata"].tobytes()
        if version != _CACHE_VERSION or stored_key != cache_key:
            return None
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        if (
            not isinstance(metadata, list)
            or len(chunks) != len(metadata)
            or embeddings.ndim != 2
            or len(embeddings) != len(chunks)
        ):
            raise ValueError("Cached index has inconsistent data.")
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        logging.warning("Ignoring invalid embedding cache %s: %s", cache_path, error)
        return None

    return chunks, embeddings, metadata


def _save_cache(cache_path, cache_key, chunks, embeddings, metadata):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                version=np.array([_CACHE_VERSION], dtype=np.int64),
                key=np.array([cache_key]),
                chunks=np.asarray(chunks, dtype=np.str_),
                embeddings=np.asarray(embeddings, dtype=np.float32),
                metadata=np.frombuffer(
                    json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
                    dtype=np.uint8,
                ),
            )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_or_create_index(
    documents, chunk_size, chunk_overlap, cache_path
):
    cache_path = Path(cache_path)
    key = _cache_key(documents, chunk_size, chunk_overlap)
    cached_index = _load_cache(cache_path, key)
    if cached_index is not None:
        return cached_index

    chunks = []
    metadata = []
    for document in documents:
        document_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(document_chunks)
        metadata.extend(
            {
                "source": document["source"],
                "page": document.get("page"),
                "chunk_id": chunk_id,
            }
            for chunk_id in range(len(document_chunks))
        )

    embeddings = np.asarray(embed_texts(chunks), dtype=np.float32)
    _save_cache(cache_path, key, chunks, embeddings, metadata)
    return chunks, embeddings, metadata
