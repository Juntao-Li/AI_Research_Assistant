import hashlib
import json
from pathlib import Path

from src.documents import load_document_file, load_documents


def load_benchmark_corpus(documents_dir=None, corpus_manifest=None):
    if (documents_dir is None) == (corpus_manifest is None):
        raise ValueError(
            "Choose exactly one corpus source: a documents directory or a manifest."
        )

    if corpus_manifest is not None:
        manifest_path = Path(corpus_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("documents")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Corpus manifest needs a non-empty documents list.")
        corpus_root = manifest_path.parent
        selected_files = []
        for entry in entries:
            relative_path = Path(entry["path"])
            if relative_path.is_absolute():
                raise ValueError("Manifest document paths must be relative.")
            path = (corpus_root / relative_path).resolve()
            if corpus_root not in path.parents:
                raise ValueError("Manifest document path escapes its directory.")
            if not path.is_file():
                raise FileNotFoundError(f"Manifest document does not exist: {path}")
            source = entry.get("source")
            if not source or Path(source).is_absolute():
                raise ValueError("Each manifest document needs a relative source name.")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            expected_checksum = entry.get("sha256")
            if expected_checksum and checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for manifest document {source!r}: "
                    f"expected {expected_checksum}, got {checksum}."
                )
            selected_files.append(
                (path, source, relative_path.as_posix(), entry)
            )
    else:
        corpus_root = Path(documents_dir).expanduser().resolve()
        if not corpus_root.is_dir():
            raise FileNotFoundError(f"Corpus directory does not exist: {corpus_root}")
        selected_files = [
            (
                path,
                path.relative_to(corpus_root).as_posix(),
                path.relative_to(corpus_root).as_posix(),
            )
            for path in sorted(corpus_root.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in {".docx", ".md", ".pdf", ".txt"}
        ]
        if not selected_files:
            raise ValueError(f"No supported documents found in {corpus_root}.")
        selected_files = [(*item, {}) for item in selected_files]

    source_names = [source for _, source, _, _ in selected_files]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Corpus selection contains duplicate source names.")
    file_paths = [path for path, _, _, _ in selected_files]
    if len(file_paths) != len(set(file_paths)):
        raise ValueError("Corpus selection contains duplicate document paths.")

    documents = []
    manifest_entries = []
    for path, source, relative_path, origin in selected_files:
        matched = load_document_file(path, source)
        if not matched:
            raise ValueError(f"Could not parse selected document: {path}")
        documents.extend(matched)
        entry = {
            "source": source,
            "path": relative_path,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "page_or_text_records": len(matched),
        }
        for key in ("source_url", "retrieved_date"):
            if key in origin:
                entry[key] = origin[key]
        manifest_entries.append(entry)

    return documents, {
        "root": str(corpus_root),
        "document_count": len(manifest_entries),
        "page_or_text_records": len(documents),
        "manifest": manifest_entries,
    }
