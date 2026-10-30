import threading
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import AssistantRuntime, IndexSnapshot, create_app
from src.retriever import build_retriever


def make_snapshot(version, chunks=None):
    chunks = chunks or ["The local guide explains a research topic."]
    retriever = build_retriever(
        chunks,
        [[1.0, 0.0] for _ in chunks],
        [
            {"source": f"guide-{index}.txt", "page": None, "chunk_id": index}
            for index in range(len(chunks))
        ],
    )
    return IndexSnapshot(retriever, version, 1, len(chunks))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.sample_root = root / "samples"
        self.upload_root = root / "uploads"
        self.cache_root = root / "cache"
        self.sample_root.mkdir()
        self.upload_root.mkdir()
        self.runtime = AssistantRuntime(
            document_root=self.upload_root,
            sample_root=self.sample_root,
            cache_root=self.cache_root,
            api_key=None,
        )
        self.app = create_app(
            runtime_factory=lambda: self.runtime,
            load_embedding_model=False,
            rebuild_on_startup=False,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.runtime._snapshot = make_snapshot("test-index-v1")

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_health_readiness_and_question_validation(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertTrue(self.client.get("/ready").json()["ready"])
        invalid = self.client.post(
            "/retrieve",
            json={"question": "  ", "mode": "dense", "top_k": 0},
        )
        self.assertEqual(invalid.status_code, 422)

    def test_dense_and_hybrid_return_typed_sources_and_correct_score_labels(self):
        with patch("src.retriever.embed_text", return_value=[1.0, 0.0]):
            dense = self.client.post(
                "/retrieve",
                json={"question": "local guide", "mode": "dense", "top_k": 1},
            ).json()
            hybrid = self.client.post(
                "/retrieve",
                json={"question": "local guide", "mode": "hybrid", "top_k": 1},
            ).json()
        self.assertEqual(dense["index_version"], "test-index-v1")
        self.assertEqual(dense["passages"][0]["score_type"], "cosine similarity")
        self.assertEqual(
            hybrid["passages"][0]["score_type"],
            "reciprocal-rank fusion score",
        )
        self.assertEqual(
            dense["passages"][0]["metadata"],
            {"source": "guide-0.txt", "page": None, "chunk_id": 0},
        )
        self.assertEqual(
            hybrid["passages"][0]["metadata"]["source"], "guide-0.txt"
        )

    def test_generation_credentials_are_backend_only_and_retrieval_still_works(self):
        with patch("src.retriever.embed_text", return_value=[1.0, 0.0]):
            retrieved = self.client.post(
                "/retrieve", json={"question": "topic"}
            )
            ask = self.client.post("/ask", json={"question": "topic"})
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(ask.status_code, 503)
        self.assertIn("Set OPENAI_API_KEY", ask.json()["detail"])

    def test_upload_refreshes_index_and_rejects_unsupported_or_oversized_files(self):
        response = self.client.post(
            "/documents",
            files={"file": ("../../not-used.txt", b"Uploadable guide text.", "text/plain")},
        )
        self.assertEqual(response.status_code, 201)
        stored_name = response.json()["stored_name"]
        self.assertEqual(Path(stored_name).name, stored_name)
        self.assertTrue((self.upload_root / stored_name).is_file())

        with (
            patch(
                "src.api.load_or_create_index",
                return_value=(
                    ["Uploadable guide text."],
                    [[1.0, 0.0]],
                    [{"source": stored_name, "page": None, "chunk_id": 0}],
                ),
            ),
            patch("src.retriever.embed_text", return_value=[1.0, 0.0]),
        ):
            rebuilt = self.client.post("/index/rebuild")
            found = self.client.post(
                "/retrieve",
                json={"question": "uploadable guide", "top_k": 1},
            )
        self.assertEqual(rebuilt.status_code, 200)
        self.assertNotEqual(rebuilt.json()["index_version"], "test-index-v1")
        self.assertEqual(found.status_code, 200)
        self.assertEqual(
            found.json()["passages"][0]["metadata"]["source"], stored_name
        )

        unsupported = self.client.post(
            "/documents",
            files={"file": ("payload.exe", b"not allowed", "application/octet-stream")},
        )
        self.assertEqual(unsupported.status_code, 400)

    def test_queries_continue_using_old_snapshot_while_rebuild_runs(self):
        build_started = threading.Event()
        allow_build_finish = threading.Event()
        replacement = make_snapshot("test-index-v2", ["Replacement is complete."])

        def delayed_build():
            build_started.set()
            if not allow_build_finish.wait(timeout=5):
                raise TimeoutError("test did not release rebuild")
            return replacement

        with (
            patch.object(self.runtime, "_build_snapshot_sync", side_effect=delayed_build),
            patch("src.retriever.embed_text", return_value=[1.0, 0.0]),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            rebuild_future = executor.submit(self.client.post, "/index/rebuild")
            self.assertTrue(build_started.wait(timeout=5))
            during_rebuild = self.client.post(
                "/retrieve", json={"question": "local guide", "top_k": 1}
            )
            self.assertEqual(during_rebuild.status_code, 200)
            self.assertEqual(during_rebuild.json()["index_version"], "test-index-v1")
            allow_build_finish.set()
            rebuilt = rebuild_future.result(timeout=5)
        self.assertEqual(rebuilt.status_code, 200)
        self.assertEqual(
            self.client.get("/ready").json()["index_version"], "test-index-v2"
        )

    def test_provider_failures_are_reported_without_leaking_provider_key(self):
        self.runtime.api_key = "private-test-token"
        with (
            patch("src.retriever.embed_text", return_value=[1.0, 0.0]),
            patch(
                "src.api.generate_answer",
                side_effect=RuntimeError("Could not reach the chat API: timeout"),
            ),
        ):
            response = self.client.post("/ask", json={"question": "topic"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private-test-token", response.text)
        self.assertIn("timeout", response.json()["detail"])

    def test_successful_answer_carries_citable_source_metadata(self):
        self.runtime.api_key = "private-test-token"
        with (
            patch("src.retriever.embed_text", return_value=[1.0, 0.0]),
            patch("src.api.generate_answer", return_value="The guide says so [1]."),
        ):
            response = self.client.post(
                "/ask",
                json={"question": "topic", "mode": "dense", "top_k": 1},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "The guide says so [1].")
        self.assertEqual(
            response.json()["citations"][0],
            {"source": "guide-0.txt", "page": None, "chunk_id": 0},
        )


if __name__ == "__main__":
    unittest.main()
