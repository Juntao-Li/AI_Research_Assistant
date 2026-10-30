import unittest
import random

from src.hybrid_retrieval import (
    BM25Index,
    bm25_search,
    reciprocal_rank_fusion,
)


class HybridRetrievalTests(unittest.TestCase):
    def test_bm25_prioritizes_query_terms(self):
        chunks = [
            "Cooling centers offer shade during dangerous heat.",
            "Tree canopy cools neighborhoods through shade.",
            "Libraries may act as cooling centers.",
        ]
        metadata = [
            {"source": f"{index}.md", "page": None, "chunk_id": 0}
            for index in range(len(chunks))
        ]

        results = bm25_search("cooling centers", chunks, metadata, top_k=3)

        self.assertEqual(results[0]["metadata"]["source"], "2.md")
        self.assertEqual(results[0]["score"] > 0, True)

    def test_reciprocal_rank_fusion_uses_stable_source_keys(self):
        dense = [
            {
                "text": "A",
                "score": 0.9,
                "metadata": {"source": "a.md", "page": None, "chunk_id": 0},
            },
            {
                "text": "B",
                "score": 0.8,
                "metadata": {"source": "b.md", "page": None, "chunk_id": 0},
            },
        ]
        lexical = [
            {
                "text": "B",
                "score": 1.0,
                "metadata": {"source": "b.md", "page": None, "chunk_id": 0},
            },
            {
                "text": "A",
                "score": 0.1,
                "metadata": {"source": "a.md", "page": None, "chunk_id": 0},
            },
        ]

        results = reciprocal_rank_fusion(dense, lexical, top_k=2)

        self.assertEqual(
            [item["metadata"]["source"] for item in results],
            ["a.md", "b.md"],
        )

    def test_precomputed_bm25_matches_original_scores_and_stable_rankings(self):
        randomizer = random.Random(73)
        vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
        chunks = [
            " ".join(randomizer.choices(vocabulary, k=randomizer.randint(0, 35)))
            for _ in range(100)
        ]
        metadata = [
            {"source": f"{index}.txt", "page": None, "chunk_id": index}
            for index in range(len(chunks))
        ]
        index = BM25Index(chunks, metadata)
        for query in (
            "alpha",
            "alpha alpha beta",
            "absent term",
            "gamma delta epsilon",
            "",
        ):
            expected = bm25_search(query, chunks, metadata, top_k=len(chunks))
            actual = index.search(query, top_k=len(chunks))
            self.assertEqual(
                [item["metadata"] for item in actual],
                [item["metadata"] for item in expected],
            )
            for left, right in zip(expected, actual):
                self.assertAlmostEqual(left["score"], right["score"], places=12)


if __name__ == "__main__":
    unittest.main()
