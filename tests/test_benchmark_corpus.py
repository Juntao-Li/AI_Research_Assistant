import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_corpus import load_benchmark_corpus


class BenchmarkCorpusTests(unittest.TestCase):
    def test_manifest_loads_unlabeled_documents_and_raw_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "answer.md").write_text(
                "A labeled document with evidence.", encoding="utf-8"
            )
            (root / "distractor.md").write_text(
                "An unlabeled but indexed distractor.", encoding="utf-8"
            )
            manifest = root / "corpus.json"
            manifest.write_text(
                json.dumps(
                    {
                        "documents": [
                            {"path": "answer.md", "source": "answer.md"},
                            {
                                "path": "distractor.md",
                                "source": "distractor.md",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            documents, info = load_benchmark_corpus(corpus_manifest=manifest)

        self.assertEqual(
            {document["source"] for document in documents},
            {"answer.md", "distractor.md"},
        )
        self.assertEqual(info["document_count"], 2)
        self.assertEqual(len(info["manifest"]), 2)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in info["manifest"]))

    def test_directory_selection_includes_all_supported_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "labeled.md").write_text("Labeled content.", encoding="utf-8")
            (root / "distractor.txt").write_text(
                "Distractor content.", encoding="utf-8"
            )
            (root / "ignored.bin").write_bytes(b"not a document")

            documents, info = load_benchmark_corpus(documents_dir=root)

        self.assertEqual(
            {document["source"] for document in documents},
            {"labeled.md", "distractor.txt"},
        )
        self.assertEqual(info["document_count"], 2)

    def test_requires_exactly_one_selection_mode_and_rejects_missing_manifest_file(self):
        with self.assertRaisesRegex(ValueError, "Choose exactly one"):
            load_benchmark_corpus()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "corpus.json"
            manifest.write_text(
                json.dumps(
                    {"documents": [{"path": "missing.md", "source": "missing.md"}]}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_benchmark_corpus(corpus_manifest=manifest)


if __name__ == "__main__":
    unittest.main()
