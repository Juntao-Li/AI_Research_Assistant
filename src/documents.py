from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PdfReadError


SUPPORTED_SUFFIXES = {".docx", ".md", ".pdf", ".txt"}


def _load_word_document(path, source):
    try:
        word_document = Document(path)
        text_parts = []
        for block in word_document.iter_inner_content():
            if hasattr(block, "text"):
                text = block.text.strip()
                if text:
                    text_parts.append(text)
            else:
                for row in block.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    ]
                    if cells:
                        text_parts.append(" | ".join(cells))
    except (BadZipFile, KeyError, PackageNotFoundError, ValueError) as error:
        raise ValueError(f"Could not read Word document {path}: {error}") from error

    text = "\n\n".join(text_parts)
    if not text:
        raise ValueError(f"Word document contains no extractable text: {path}")

    return {"source": source, "page": None, "text": text}


def _load_pdf_pages(path, source):
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError(f"PDF is encrypted and cannot be read: {path}")

        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    {
                        "source": source,
                        "page": page_number,
                        "text": text,
                    }
                )
    except (PdfReadError, EOFError) as error:
        raise ValueError(f"Could not read PDF {path}: {error}") from error

    if not documents:
        raise ValueError(
            f"PDF contains no extractable text: {path}. "
            "Scanned PDFs require OCR before they can be indexed."
        )

    return documents


def load_document_file(path, source):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_pages(path, source)
    if suffix == ".docx":
        return [_load_word_document(path, source)]
    if suffix not in {".md", ".txt"}:
        raise ValueError(f"Unsupported document format: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Document is not valid UTF-8: {path}") from error
    if not text.strip():
        return []
    return [{"source": source, "page": None, "text": text}]


def load_documents(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {root}")

    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        source = path.relative_to(root).as_posix()
        documents.extend(load_document_file(path, source))

    if not documents:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"No non-empty {supported} documents found in {root}."
        )

    return documents
