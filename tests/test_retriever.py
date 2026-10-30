import unittest
from unittest.mock import patch

import numpy as np

from src.retriever import (
    Retriever,
    build_retriever,
    retrieval_fingerprint,
)
from src.vector_store import VectorStore


class DenseSearchTests(unittest.TestCase):
    def test_vectorized_cosine_matches_scalar_ranking_scores_and_metadata(self):
        embeddings = [
            np.array([1.0, 2.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ]
        store = VectorStore()
        for index, embedding in enumerate(embeddings):
            store.add(
                f"chunk-{index}",
                embedding,
                {"source": f"source-{index}", "chunk_id": index},
            )
        query = np.array([1.0, 2.0, 0.0], dtype=np.float32)

        expected = [
            {
                "score": store.cosine_similarity(query, item["embedding"]),
                "text": item["text"],
                "metadata": item["metadata"],
            }
            for item in store.items
        ]
        expected.sort(key=lambda result: result["score"], reverse=True)
        actual = store.search(query, top_k=4)

        self.assertEqual(
            [item["metadata"] for item in actual],
            [item["metadata"] for item in expected],
        )
        np.testing.assert_allclose(
            [item["score"] for item in actual],
            [item["score"] for item in expected],
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(actual[-1]["metadata"]["source"], "source-3")

    def test_equal_score_ties_retain_insertion_order_and_prepare_invalidates(self):
        store = VectorStore()
        store.add("first", [1.0, 0.0], {"id": 1})
        store.add("second", [1.0, 0.0], {"id": 2})
        self.assertEqual(
            [item["metadata"]["id"] for item in store.search([1.0, 0.0], 2)],
            [1, 2],
        )
        store.add("third", [1.0, 0.0], {"id": 3})
        self.assertEqual(
            [item["metadata"]["id"] for item in store.search([1.0, 0.0], 3)],
            [1, 2, 3],
        )


class SharedRetrieverTests(unittest.TestCase):
    def test_modes_preserve_source_page_chunk_and_label_scores(self):
        retriever = build_retriever(
            ["A cooling center offers shade.", "Trees cool urban streets."],
            [[1.0, 0.0], [0.0, 1.0]],
            [
                {"source": "brief.pdf", "page": 2, "chunk_id": 7},
                {"source": "notes.md", "page": None, "chunk_id": 4},
            ],
        )
        dense = retriever.search_with_embedding(
            "cooling center", [1.0, 0.0], top_k=2, mode="dense"
        )
        hybrid = retriever.search_with_embedding(
            "cooling center", [1.0, 0.0], top_k=2, mode="hybrid"
        )

        self.assertEqual(dense[0]["score_type"], "cosine similarity")
        self.assertEqual(
            hybrid[0]["score_type"], "reciprocal-rank fusion score"
        )
        self.assertEqual(
            dense[0]["metadata"],
            {"source": "brief.pdf", "page": 2, "chunk_id": 7},
        )
        self.assertEqual(
            hybrid[0]["metadata"],
            {"source": "brief.pdf", "page": 2, "chunk_id": 7},
        )

    def test_query_embedding_is_computed_once_and_invalid_mode_is_rejected(self):
        retriever = build_retriever(
            ["alpha beta"],
            [[1.0, 0.0]],
            [{"source": "notes.md", "page": None, "chunk_id": 0}],
        )
        with patch("src.retriever.embed_text", return_value=[1.0, 0.0]) as embed:
            results = retriever.search("alpha", top_k=1, mode="hybrid")
        embed.assert_called_once_with("alpha")
        self.assertEqual(len(results), 1)
        with self.assertRaisesRegex(ValueError, "mode must be"):
            retriever.search("alpha", mode="bm25")

    def test_fingerprint_changes_for_content_location_or_retrieval_settings(self):
        documents = [
            {"source": "a.txt", "page": None, "text": "alpha"}
        ]
        baseline = retrieval_fingerprint(documents, 500, 100)
        changed_text = [{"source": "a.txt", "page": None, "text": "beta"}]
        changed_source = [{"source": "b.txt", "page": None, "text": "alpha"}]
        self.assertNotEqual(baseline, retrieval_fingerprint(changed_text, 500, 100))
        self.assertNotEqual(baseline, retrieval_fingerprint(changed_source, 500, 100))
        self.assertNotEqual(baseline, retrieval_fingerprint(documents, 600, 100))
        self.assertNotEqual(baseline, retrieval_fingerprint(documents, 500, 50))


if __name__ == "__main__":
    unittest.main()
