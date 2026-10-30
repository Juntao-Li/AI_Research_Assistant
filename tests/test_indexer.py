import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.indexer import load_or_create_index


class LoadOrCreateIndexTests(unittest.TestCase):
    def test_reuses_saved_embeddings_on_second_call(self):
        documents = [
            {
                "source": "notes.txt",
                "page": None,
                "text": "Local indexing avoids repeated embedding.",
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "index.npz"
            with patch(
                "src.indexer.embed_texts",
                return_value=[[0.1, 0.2, 0.3]],
            ) as mock_embed:
                first = load_or_create_index(
                    documents, 500, 100, cache_path
                )
                second = load_or_create_index(
                    documents, 500, 100, cache_path
                )

        mock_embed.assert_called_once()
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[1].tolist(), second[1].tolist())

    def test_rebuilds_cache_when_document_changes(self):
        documents = [
            {
                "source": "notes.txt",
                "page": None,
                "text": "First contents.",
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "index.npz"
            with patch(
                "src.indexer.embed_texts",
                side_effect=[[[0.1, 0.2]], [[0.3, 0.4]]],
            ) as mock_embed:
                first = load_or_create_index(
                    documents, 500, 100, cache_path
                )
                changed_documents = [
                    {
                        **documents[0],
                        "text": "Changed contents.",
                    }
                ]
                second = load_or_create_index(
                    changed_documents, 500, 100, cache_path
                )

        self.assertEqual(mock_embed.call_count, 2)
        self.assertNotEqual(first[0], second[0])

    def test_rebuilds_corrupt_cache(self):
        documents = [
            {
                "source": "notes.txt",
                "page": None,
                "text": "Recover from a corrupt cache.",
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "index.npz"
            cache_path.write_bytes(b"not a valid npz file")
            with patch(
                "src.indexer.embed_texts",
                return_value=[[0.5, 0.6]],
            ) as mock_embed:
                chunks, embeddings, metadata = load_or_create_index(
                    documents, 500, 100, cache_path
                )

        mock_embed.assert_called_once()
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(float(embeddings[0, 0]), 0.5)
        self.assertAlmostEqual(float(embeddings[0, 1]), 0.6)
        self.assertEqual(metadata[0]["source"], "notes.txt")


if __name__ == "__main__":
    unittest.main()
