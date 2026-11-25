import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.llm.llm_client import (
    FallbackLLMClient,
    StandardLLMClient,
    create_llm_client,
)


class TestStandardAndFactoryLLMClient(unittest.TestCase):
    def test_create_single_standard_client(self) -> None:
        fake_settings = SimpleNamespace(
            LLM_BASE_URL="https://api.example.com/v1",
            LLM_API_KEY="sk-test",
            LLM_MODEL="model-a",
            LLM_FALLBACKS="",
        )
        with patch("src.llm.llm_client.get_settings", return_value=fake_settings):
            client = create_llm_client()

        self.assertIsInstance(client, StandardLLMClient)
        self.assertEqual(client.endpoint, "https://api.example.com/v1/chat/completions")
        self.assertEqual(client.model, "model-a")

    def test_create_builds_fallback_chain_from_json(self) -> None:
        fake_settings = SimpleNamespace(
            LLM_BASE_URL="https://primary.example.com/v1",
            LLM_API_KEY="sk-1",
            LLM_MODEL="model-1",
            LLM_FALLBACKS=json.dumps(
                [
                    {
                        "base_url": "https://backup.example.com/v1/chat/completions",
                        "api_key": "sk-2",
                        "model": "model-2",
                    }
                ]
            ),
        )
        with patch("src.llm.llm_client.get_settings", return_value=fake_settings):
            client = create_llm_client()

        self.assertIsInstance(client, FallbackLLMClient)
        self.assertEqual(client.model, "model-1")

    def test_missing_base_url_or_model_raises(self) -> None:
        fake_settings = SimpleNamespace(
            LLM_BASE_URL="",
            LLM_API_KEY="",
            LLM_MODEL="",
            LLM_FALLBACKS="",
        )
        with (
            patch("src.llm.llm_client.get_settings", return_value=fake_settings),
            self.assertRaises(ValueError),
        ):
            create_llm_client()

    def test_standard_client_generate_and_stream(self) -> None:
        fake_resp = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "नमस्ते"}}]}).encode("utf-8")
        )
        fake_cm = MagicMock()
        fake_cm.__enter__.return_value = fake_resp
        fake_cm.__exit__.return_value = False

        with patch("src.llm.llm_client.urllib.request.urlopen", return_value=fake_cm):
            client = StandardLLMClient(
                base_url="https://api.example.com/v1",
                api_key="sk-test",
                model="model-a",
            )
            self.assertEqual(client.generate("sys", "user"), "नमस्ते")

        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"!"}}]}\n',
            b"data: [DONE]\n",
        ]
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = iter(sse_lines)
        stream_cm.__exit__.return_value = False

        with patch("src.llm.llm_client.urllib.request.urlopen", return_value=stream_cm):
            self.assertEqual("".join(client.stream("sys", "user")), "Hello!")


class TestFallbackLLMClient(unittest.TestCase):
    def test_failover_on_error_string_and_cooldown_skips_failed_endpoint(self) -> None:
        now = [0.0]
        p1 = MagicMock(model="model-1")
        p1.generate.return_value = "[Error: LLM generation failed — HTTP 429: Rate limit]"
        p2 = MagicMock(model="model-2")
        p2.generate.return_value = "जवाफ"

        router = FallbackLLMClient([p1, p2], cooldown_seconds=60.0, clock=lambda: now[0])

        self.assertEqual(router.generate("sys", "q1"), "जवाफ")
        self.assertEqual(p1.generate.call_count, 1)
        self.assertEqual(p2.generate.call_count, 1)
        self.assertEqual(router.model, "model-2")

        now[0] = 10.0
        self.assertEqual(router.generate("sys", "q2"), "जवाफ")
        self.assertEqual(p1.generate.call_count, 1)
        self.assertEqual(p2.generate.call_count, 2)

        now[0] = 61.0
        p1.generate.return_value = "Recovered"
        self.assertEqual(router.generate("sys", "q3"), "Recovered")
        self.assertEqual(p1.generate.call_count, 2)

    def test_stream_failover_when_first_endpoint_errors(self) -> None:
        p1 = MagicMock(model="model-1")
        p1.stream.return_value = iter(["\n[Error: streaming failed — 429]"])
        p2 = MagicMock(model="model-2")
        p2.stream.return_value = iter(["Hello", " Nepal"])

        router = FallbackLLMClient([p1, p2])
        self.assertEqual("".join(router.stream("sys", "hi")), "Hello Nepal")


class TestRateLimitBackoffAndFailover(unittest.TestCase):
    def _make_http_error(
        self,
        code: int,
        body: dict,
        headers: dict[str, str] | None = None,
    ):
        import email.message
        import urllib.error

        hdrs = email.message.Message()
        for k, v in (headers or {}).items():
            hdrs[k] = v
        return urllib.error.HTTPError(
            url="https://api.example.com/v1/chat/completions",
            code=code,
            msg="Too Many Requests",
            hdrs=hdrs,
            fp=io.BytesIO(json.dumps(body).encode("utf-8")),
        )

    def test_generate_uses_reset_header_and_jittered_exponential_backoff(self) -> None:
        sleeps: list[float] = []
        err1 = self._make_http_error(
            429,
            {"error": {"message": "Limit 8000, Used 3000, Requested 2000. Please try again in 4.5s"}},
            headers={"x-ratelimit-reset-tokens": "4.5s"},
        )
        err2 = self._make_http_error(
            429,
            {"error": {"message": "Rate limit reached"}},
            headers={},
        )
        ok_resp = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "सफल"}}]}).encode("utf-8")
        )
        ok_cm = MagicMock()
        ok_cm.__enter__.return_value = ok_resp
        ok_cm.__exit__.return_value = False

        client = StandardLLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="model-a",
            max_retries=3,
            retry_base_seconds=2.0,
            retry_max_wait_seconds=60.0,
            sleep_fn=sleeps.append,
            jitter_fn=lambda: 0.25,
        )
        with patch(
            "src.llm.llm_client.urllib.request.urlopen",
            side_effect=[err1, err2, ok_cm],
        ):
            out = client.generate("sys", "user")

        self.assertEqual(out, "सफल")
        # Attempt 1 uses header (4.5s + 0.25s jitter); Attempt 2 uses exponential 2.0 * 2^1 + 0.25s = 4.25s
        self.assertEqual(sleeps, [4.75, 4.25])

    def test_generate_fails_fast_when_single_request_exceeds_tpm_limit(self) -> None:
        sleeps: list[float] = []
        err = self._make_http_error(
            429,
            {
                "error": {
                    "message": (
                        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                        "Limit 8000, Used 2114, Requested 9500."
                    )
                }
            },
            headers={"retry-after": "30"},
        )
        client = StandardLLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="model-a",
            max_retries=4,
            sleep_fn=sleeps.append,
        )
        with patch("src.llm.llm_client.urllib.request.urlopen", side_effect=err) as mock_open:
            out = client.generate("sys", "user")

        self.assertIn("429", out)
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(sleeps, [])

    def test_stream_retries_pre_first_token_with_capped_header_wait(self) -> None:
        sleeps: list[float] = []
        err = self._make_http_error(
            429,
            {"error": {"message": "Rate limit reached"}},
            headers={"retry-after": "1m2s"},
        )
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"!"}}]}\n',
            b"data: [DONE]\n",
        ]
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = iter(sse_lines)
        stream_cm.__exit__.return_value = False

        client = StandardLLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="model-a",
            max_retries=3,
            retry_max_wait_seconds=10.0,
            sleep_fn=sleeps.append,
            jitter_fn=lambda: 0.0,
        )
        with patch(
            "src.llm.llm_client.urllib.request.urlopen",
            side_effect=[err, stream_cm],
        ):
            out = "".join(client.stream("sys", "user"))

        self.assertEqual(out, "Hi!")
        self.assertEqual(sleeps, [10.0])

    def test_create_fallback_chain_fails_over_immediately_without_per_client_sleep(self) -> None:
        sleeps: list[float] = []
        fake_settings = SimpleNamespace(
            LLM_BASE_URL="https://primary.example.com/v1",
            LLM_API_KEY="sk-1",
            LLM_MODEL="model-1",
            LLM_FALLBACKS=json.dumps(
                [
                    {
                        "base_url": "https://backup.example.com/v1",
                        "api_key": "sk-2",
                        "model": "model-2",
                    }
                ]
            ),
            LLM_MAX_RETRIES=4,
            LLM_RETRY_BASE_SECONDS=2.0,
            LLM_RETRY_MAX_WAIT_SECONDS=60.0,
        )
        with patch("src.llm.llm_client.get_settings", return_value=fake_settings):
            router = create_llm_client()

        self.assertIsInstance(router, FallbackLLMClient)
        assert isinstance(router, FallbackLLMClient)
        for c in router._clients:
            self.assertEqual(c.max_retries, 1)  # type: ignore[attr-defined]

        err1 = self._make_http_error(429, {"error": {"message": "Rate limit reached"}})
        ok_resp = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "Backup ok"}}]}).encode("utf-8")
        )
        ok_cm = MagicMock()
        ok_cm.__enter__.return_value = ok_resp
        ok_cm.__exit__.return_value = False

        with (
            patch("src.llm.llm_client.time.sleep", side_effect=sleeps.append),
            patch("src.llm.llm_client.urllib.request.urlopen", side_effect=[err1, ok_cm]),
        ):
            out = router.generate("sys", "user")

        self.assertEqual(out, "Backup ok")
        self.assertEqual(sleeps, [])

    def test_fallback_backs_off_only_after_all_endpoints_fail(self) -> None:
        sleeps: list[float] = []
        p1 = MagicMock(model="model-1")
        p1.generate.side_effect = [
            "[Error: LLM generation failed — HTTP 429: Rate limit]",
            "Recovered",
        ]
        p2 = MagicMock(model="model-2")
        p2.generate.return_value = "[Error: LLM generation failed — HTTP 429: Rate limit]"

        router = FallbackLLMClient(
            [p1, p2],
            max_exhausted_retries=2,
            retry_base_seconds=2.0,
            sleep_fn=sleeps.append,
        )
        out = router.generate("sys", "q")
        self.assertEqual(out, "Recovered")
        self.assertEqual(sleeps, [2.0])

    def test_create_helper_llm_client_respects_setting(self) -> None:
        from src.llm.llm_client import create_helper_llm_client

        empty_settings = SimpleNamespace(
            LLM_BASE_URL="https://api.example.com/v1",
            LLM_API_KEY="sk-1",
            LLM_MODEL="model-main",
            LLM_HELPER_MODEL="",
            LLM_FALLBACKS="",
        )
        with patch("src.llm.llm_client.get_settings", return_value=empty_settings):
            self.assertIsNone(create_helper_llm_client())

        helper_settings = SimpleNamespace(
            LLM_BASE_URL="https://api.example.com/v1",
            LLM_API_KEY="sk-1",
            LLM_MODEL="model-main",
            LLM_HELPER_MODEL="llama-3.1-8b-instant",
            LLM_FALLBACKS="",
        )
        with patch("src.llm.llm_client.get_settings", return_value=helper_settings):
            helper = create_helper_llm_client()
        self.assertIsNotNone(helper)
        assert helper is not None
        self.assertEqual(helper.model, "llama-3.1-8b-instant")

    def test_max_retries_zero_still_attempts_once(self) -> None:
        ok_resp = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
        )
        ok_cm = MagicMock()
        ok_cm.__enter__.return_value = ok_resp
        ok_cm.__exit__.return_value = False

        client = StandardLLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            model="model-a",
            max_retries=0,
        )
        self.assertEqual(client.max_retries, 1)
        with patch("src.llm.llm_client.urllib.request.urlopen", return_value=ok_cm):
            self.assertEqual(client.generate("sys", "user"), "ok")

    def test_fallback_stream_backs_off_only_after_all_endpoints_fail(self) -> None:
        sleeps: list[float] = []
        p1 = MagicMock(model="model-1")
        p1.stream.side_effect = [
            iter(["\n[Error: streaming failed — HTTP 429: Rate limit]"]),
            iter(["Recovered", " stream"]),
        ]
        p2 = MagicMock(model="model-2")
        p2.stream.return_value = iter(["\n[Error: streaming failed — HTTP 429: Rate limit]"])

        router = FallbackLLMClient(
            [p1, p2],
            max_exhausted_retries=2,
            retry_base_seconds=2.0,
            sleep_fn=sleeps.append,
        )
        self.assertEqual("".join(router.stream("sys", "hi")), "Recovered stream")
        self.assertEqual(sleeps, [2.0])


if __name__ == "__main__":
    unittest.main()
