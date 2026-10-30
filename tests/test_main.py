import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

import main


class MainTests(unittest.TestCase):
    @patch("main.generate_answer", return_value="Answer supported by [1].")
    @patch("main.retrieve")
    @patch("main.load_or_create_index")
    @patch("main.load_documents")
    def test_passes_pdf_page_into_context_and_prints_page_source(
        self,
        mock_load_documents,
        mock_load_index,
        mock_retrieve,
        mock_generate,
    ):
        mock_load_documents.return_value = [
            {
                "source": "paper.pdf",
                "page": 7,
                "text": "Evidence from page seven.",
            }
        ]
        mock_load_index.return_value = (
            ["Evidence from page seven."],
            [[1.0, 0.0]],
            [{"source": "paper.pdf", "page": 7, "chunk_id": 0}],
        )
        mock_retrieve.return_value = [
            {
                "score": 0.9,
                "text": "Evidence from page seven.",
                "metadata": {
                    "source": "paper.pdf",
                    "page": 7,
                    "chunk_id": 0,
                },
                "score_type": "cosine similarity",
            }
        ]
        output = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                ["main.py", "--question", "What does the paper say?"],
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            contextlib.redirect_stdout(output),
        ):
            main.main()

        context = mock_generate.call_args.kwargs["context"]
        self.assertIn("Source: paper.pdf (chunk 0, page 7)", context)
        self.assertIn("Evidence from page seven.", context)
        self.assertIn("paper.pdf (chunk 0, page 7", output.getvalue())

    @patch("main.generate_answer")
    @patch("main.retrieve")
    @patch("main.load_or_create_index")
    @patch("main.load_documents", return_value=[])
    def test_retrieve_only_does_not_require_or_call_chat_api(
        self,
        _mock_load_documents,
        mock_load_index,
        mock_retrieve,
        mock_generate,
    ):
        mock_load_index.return_value = (
            ["Local evidence."],
            [[1.0, 0.0]],
            [{"source": "notes.txt", "page": None, "chunk_id": 0}],
        )
        mock_retrieve.return_value = [
            {
                "score": 0.8,
                "text": "Local evidence.",
                "metadata": {
                    "source": "notes.txt",
                    "page": None,
                    "chunk_id": 0,
                },
                "score_type": "cosine similarity",
            }
        ]
        output = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "--retrieve-only",
                    "--question",
                    "What is the evidence?",
                ],
            ),
            patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(output),
        ):
            main.main()

        mock_generate.assert_not_called()
        _mock_load_documents.assert_called_once()
        self.assertIn("Retrieved passages:", output.getvalue())
        self.assertIn("Local evidence.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
