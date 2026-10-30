import unittest

from src.chunker import chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_does_not_duplicate_chunk_when_overlap_will_not_fit(self):
        chunks = chunk_text(
            "First paragraph\n\nSecond paragraph",
            chunk_size=20,
            chunk_overlap=10,
        )

        self.assertEqual(chunks, ["First paragraph", "Second paragraph"])

    def test_long_text_chunks_respect_size_limit(self):
        chunks = chunk_text(
            "One sentence. Two sentence. Three sentence. Four sentence.",
            chunk_size=20,
            chunk_overlap=5,
        )

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))

    def test_rejects_overlap_at_least_as_large_as_chunk(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
