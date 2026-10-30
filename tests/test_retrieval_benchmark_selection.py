import unittest

from scripts.evaluate_retrieval import _select_context_bounded_candidate


class RetrievalBenchmarkSelectionTests(unittest.TestCase):
    def test_context_guard_excludes_long_context_candidate(self):
        baseline = {
            "name": "baseline",
            "hit_at_k": 0.7,
            "mrr_at_k": 0.6,
            "evidence_recall_at_k": 0.65,
            "mean_context_chars_at_k": 1000,
        }
        bounded_candidate = {
            "name": "bounded",
            "hit_at_k": 0.8,
            "mrr_at_k": 0.7,
            "evidence_recall_at_k": 0.75,
            "mean_context_chars_at_k": 1005,
        }
        unbounded_candidate = {
            "name": "whole-document",
            "hit_at_k": 1.0,
            "mrr_at_k": 1.0,
            "evidence_recall_at_k": 1.0,
            "mean_context_chars_at_k": 10000,
        }

        selected, excluded, maximum = _select_context_bounded_candidate(
            [baseline, bounded_candidate, unbounded_candidate], baseline
        )

        self.assertEqual(selected["name"], "bounded")
        self.assertEqual([item["name"] for item in excluded], ["whole-document"])
        self.assertEqual(maximum, 1010)


if __name__ == "__main__":
    unittest.main()
