import json
import tempfile
import unittest
from pathlib import Path

from src.annotation_review import ensure_review_dataset, update_review_case


class AnnotationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        (self.corpus / "guide.txt").write_text(
            "Evidence passage supports the expected response.\n",
            encoding="utf-8",
        )
        self.source = self.root / "v4.json"
        self.dataset_path = self.root / "v5.json"
        self.log_path = self.root / "review.jsonl"
        self.source.write_text(
            json.dumps(
                {
                    "dataset_version": "4.0",
                    "answerable": [
                        {
                            "id": "fixture-answerable",
                            "question": "What is supported?",
                            "expected_answer": "A supported response.",
                            "group": "fixture",
                            "split": "development",
                            "review_status": "pending",
                            "evidence": [
                                {
                                    "id": "fixture-evidence",
                                    "source": "guide.txt",
                                    "page": None,
                                    "anchor": "Evidence passage",
                                    "quotation": (
                                        "Evidence passage supports the expected response."
                                    ),
                                }
                            ],
                        }
                    ],
                    "unanswerable": [
                        {
                            "id": "fixture-unanswerable",
                            "question": "What is not in the corpus?",
                            "group": "fixture-absent",
                            "split": "held-out-test",
                            "review_status": "pending",
                            "corpus_absence_rationale": "No evidence is present.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def setUpReviewDataset(self):
        return ensure_review_dataset(
            self.source, self.dataset_path, self.corpus
        )

    def update(self, question_id, action, **kwargs):
        return update_review_case(
            self.dataset_path,
            self.log_path,
            question_id,
            action=action,
            review_notes=kwargs.pop("review_notes", ""),
            reviewer="test reviewer",
            document_root=self.corpus,
            **kwargs,
        )

    def test_initializes_new_version_without_mutating_parent_or_auto_approving(self):
        dataset = self.setUpReviewDataset()
        original = json.loads(self.source.read_text(encoding="utf-8"))
        self.assertEqual(dataset["dataset_version"], "5.0")
        self.assertEqual(
            [item["split"] for item in dataset["answerable"]],
            [item["split"] for item in original["answerable"]],
        )
        self.assertEqual(dataset["answerable"][0]["review_status"], "pending")
        self.assertEqual(
            original["answerable"][0]["expected_answer"],
            "A supported response.",
        )

    def test_explicit_approval_and_edits_reset_existing_approval_to_pending(self):
        self.setUpReviewDataset()
        dataset, record = self.update(
            "fixture-answerable",
            "approve",
            expected_answer="A supported response.",
            evidence=[
                {
                    "id": "fixture-evidence",
                    "source": "guide.txt",
                    "page": None,
                    "anchor": "Evidence passage",
                    "quotation": "Evidence passage supports the expected response.",
                }
            ],
        )
        self.assertEqual(
            dataset["answerable"][0]["review_status"], "reviewed"
        )
        self.assertEqual(record["new_status"], "reviewed")

        dataset, record = self.update(
            "fixture-answerable",
            "approve",
            expected_answer="A corrected supported response.",
        )
        self.assertEqual(dataset["answerable"][0]["review_status"], "pending")
        self.assertEqual(record["new_status"], "pending")
        self.assertTrue(record["answer_edited"])
        self.assertEqual(len(self.log_path.read_text().splitlines()), 2)

    def test_needs_correction_requires_reason_and_evidence_must_match_source(self):
        self.setUpReviewDataset()
        with self.assertRaisesRegex(ValueError, "reason is required"):
            self.update("fixture-answerable", "needs_correction")
        with self.assertRaisesRegex(ValueError, "must match the selected source"):
            self.update(
                "fixture-answerable",
                "leave_pending",
                reason="Correcting the evidence anchor.",
                evidence=[
                    {
                        "id": "fixture-evidence",
                        "source": "guide.txt",
                        "page": None,
                        "anchor": "Unsupported",
                        "quotation": "Unsupported quotation.",
                    }
                ],
            )

    def test_unanswerable_approval_requires_full_corpus_absence_rationale(self):
        self.setUpReviewDataset()
        with self.assertRaisesRegex(ValueError, "Record why"):
            self.update("fixture-unanswerable", "approve")
        dataset, _ = self.update(
            "fixture-unanswerable",
            "approve",
            corpus_search_rationale=(
                "Searched the entire selected corpus; no document discusses this."
            ),
        )
        self.assertEqual(
            dataset["unanswerable"][0]["review_status"], "reviewed"
        )


if __name__ == "__main__":
    unittest.main()
