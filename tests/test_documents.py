import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docx import Document
from pypdf.errors import PdfReadError

from src.documents import load_documents


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfReader:
    def __init__(self, pages, is_encrypted=False):
        self.pages = pages
        self.is_encrypted = is_encrypted


class LoadDocumentsTests(unittest.TestCase):
    def test_loads_supported_files_recursively_and_ignores_empty_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "notes.txt").write_text("Research notes", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "summary.MD").write_text(
                "# Summary", encoding="utf-8"
            )
            (root / "empty.txt").write_text("  \n", encoding="utf-8")
            (root / "ignored.odt").write_text("not loaded", encoding="utf-8")

            documents = load_documents(root)

        self.assertEqual(
            documents,
            [
                {"source": "nested/summary.MD", "page": None, "text": "# Summary"},
                {"source": "notes.txt", "page": None, "text": "Research notes"},
            ],
        )

    def test_loads_word_document_paragraphs_and_tables(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            word_path = root / "research.docx"
            word_document = Document()
            word_document.add_paragraph("Research overview")
            table = word_document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Measure"
            table.cell(0, 1).text = "Result"
            word_document.add_paragraph("Final note")
            word_document.save(word_path)

            documents = load_documents(root)

        self.assertEqual(
            documents,
            [
                {
                    "source": "research.docx",
                    "page": None,
                    "text": "Research overview\n\nMeasure | Result\n\nFinal note",
                }
            ],
        )

    def test_reports_invalid_word_document(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "broken.docx").write_text("not a Word file", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not read Word document"):
                load_documents(root)

    @patch("src.documents.PdfReader")
    def test_loads_pdf_text_with_original_one_based_page_numbers(
        self, mock_pdf_reader
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "paper.pdf").write_bytes(b"%PDF-fake")
            mock_pdf_reader.return_value = FakePdfReader(
                [
                    FakePage("First page text"),
                    FakePage(" \n"),
                    FakePage("Third page text"),
                ]
            )

            documents = load_documents(root)

        self.assertEqual(
            documents,
            [
                {
                    "source": "paper.pdf",
                    "page": 1,
                    "text": "First page text",
                },
                {
                    "source": "paper.pdf",
                    "page": 3,
                    "text": "Third page text",
                },
            ],
        )

    @patch("src.documents.PdfReader", side_effect=PdfReadError("invalid PDF"))
    def test_reports_unreadable_pdf(self, mock_pdf_reader):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "broken.pdf").write_bytes(b"not a PDF")

            with self.assertRaisesRegex(ValueError, "Could not read PDF"):
                load_documents(root)
            mock_pdf_reader.assert_called_once()

    @patch("src.documents.PdfReader")
    def test_reports_encrypted_pdf(self, mock_pdf_reader):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "locked.pdf").write_bytes(b"%PDF-fake")
            mock_pdf_reader.return_value = FakePdfReader([], is_encrypted=True)

            with self.assertRaisesRegex(ValueError, "PDF is encrypted"):
                load_documents(root)

    def test_errors_when_directory_has_no_supported_documents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "No non-empty"):
                load_documents(temporary_directory)


if __name__ == "__main__":
    unittest.main()
