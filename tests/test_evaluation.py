import unittest

from src.evaluation import evaluate


def result(text, source, page=None):
    return {"text": text, "metadata": {"source": source, "page": page}}


class EvaluateRetrievalTests(unittest.TestCase):
    def test_calculates_hit_mrr_multi_evidence_recall_and_context(self):
        cases = [
            {
                "id": "first",
                "question": "First question",
                "evidence": [
                    {
                        "id": "first-a",
                        "source": "notes.md",
                        "page": None,
                        "anchor": "first evidence",
                    },
                    {
                        "id": "first-b",
                        "source": "notes.md",
                        "page": None,
                        "anchor": "second evidence",
                    },
                ],
            },
            {
                "id": "second",
                "question": "Second question",
                "evidence": [
                    {
                        "id": "second-a",
                        "source": "paper.pdf",
                        "page": 2,
                        "anchor": "page two evidence",
                    }
                ],
            },
        ]
        rankings = {
            "First question": [
                result("Irrelevant passage", "other.md"),
                result("FIRST EVIDENCE is here", "notes.md"),
                result("Wrong page evidence", "paper.pdf", 1),
                result("Second evidence is here", "notes.md"),
            ],
            "Second question": [
                result("Wrong page two evidence", "paper.pdf", 1),
                result("Page two evidence is here", "paper.pdf", 2),
            ],
        }

        report = evaluate(
            cases,
            rankings.__getitem__,
            hit_k=2,
            mrr_k=10,
            evidence_k=2,
        )

        self.assertEqual(report["question_count"], 2)
        self.assertEqual(report["hit_at_k"], 1.0)
        self.assertAlmostEqual(report["mrr_at_k"], 0.5)
        self.assertEqual(report["evidence_found"], 2)
        self.assertEqual(report["evidence_denominator"], 3)
        self.assertAlmostEqual(report["evidence_recall_at_k"], 2 / 3)
        self.assertEqual(
            [outcome["first_relevant_rank"] for outcome in report["outcomes"]],
            [2, 2],
        )
        self.assertEqual(
            [outcome["context_chars_at_evidence_k"] for outcome in report["outcomes"]],
            [40, 48],
        )

    def test_returns_zero_for_missing_evidence_and_rejects_empty_or_unlabeled_sets(self):
        missing_case = [
            {
                "question": "Question",
                "evidence": [
                    {
                        "id": "missing",
                        "source": "notes.md",
                        "page": None,
                        "anchor": "missing phrase",
                    }
                ],
            }
        ]
        report = evaluate(
            missing_case,
            lambda question: [result("Different evidence", "notes.md")],
        )
        self.assertEqual(report["hit_at_k"], 0.0)
        self.assertEqual(report["mrr_at_k"], 0.0)
        self.assertEqual(report["evidence_recall_at_k"], 0.0)

        with self.assertRaisesRegex(ValueError, "at least one question"):
            evaluate([], lambda question: [question])
        with self.assertRaisesRegex(ValueError, "at least one evidence passage"):
            evaluate([{"question": "No label", "evidence": []}], lambda question: [])

    def test_matches_evidence_anchor_across_line_wrapping(self):
        cases = [
            {
                "question": "Wrapped evidence?",
                "evidence": [
                    {
                        "id": "wrapped",
                        "source": "notes.md",
                        "page": None,
                        "anchor": "exact source phrase",
                    }
                ],
            }
        ]
        report = evaluate(
            cases,
            lambda question: [
                result("An exact\nsource phrase appears here.", "notes.md")
            ],
        )

        self.assertEqual(report["hit_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
