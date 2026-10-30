import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.generator import generate_answer


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class GenerateAnswerTests(unittest.TestCase):
    @patch("src.generator.urlopen")
    def test_sends_chat_completion_request_and_returns_answer(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": "Supported answer [1]."}}]}
        )

        answer = generate_answer(
            question="What is in the notes?",
            context="[1] Source: notes.txt\nResearch notes",
            api_key="test-key",
            base_url="https://example.test/v1/",
            model="test-model",
        )

        request = mock_urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(answer, "Supported answer [1].")
        self.assertEqual(
            request.full_url,
            "https://example.test/v1/chat/completions",
        )
        self.assertTrue(request.get_header("Authorization").endswith("test-key"))
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(body["model"], "test-model")
        self.assertIn("Research notes", body["messages"][1]["content"])
        self.assertNotIn("temperature", body)

    @patch("src.generator.urlopen")
    def test_includes_fixed_temperature_when_requested(self, mock_urlopen):
        mock_urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": "Answer."}}]}
        )
        generate_answer(
            question="Question",
            context="Evidence",
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            temperature=0,
        )
        body = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertEqual(body["temperature"], 0)

    @patch("src.generator.urlopen", side_effect=URLError("offline"))
    def test_reports_connection_errors(self, mock_urlopen):
        with self.assertRaisesRegex(RuntimeError, "Could not reach the chat API"):
            generate_answer(
                question="Question",
                context="Evidence",
                api_key="test-key",
                base_url="https://example.test/v1",
                model="test-model",
            )
        mock_urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
