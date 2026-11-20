import unittest
from unittest.mock import Mock, patch

import httpx
from openai import APITimeoutError, RateLimitError

from agent_mbpp import MBPPRunner
from structures import ModelResponse


class MBPPRequestMetricsTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock(model_name="test-model")
        self.agent.client.base_url = "https://example.test"
        self.runner = MBPPRunner(self.agent, None)
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(429, request=request)
        self.rate_limit = RateLimitError("Rate limited", response=response, body=None)
        self.timeout = APITimeoutError(request=request)

        tokenizer_loader = Mock()
        tokenizer_loader.return_value.encode_ordinary.return_value = [1]
        for target, value in [
            ("agent_mbpp.START", 0),
            ("agent_mbpp.tiktoken.get_encoding", tokenizer_loader),
        ]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_timeout_on_retry_keeps_metrics_for_both_calls(self):
        self.agent.run.side_effect = [self.rate_limit, self.timeout]

        with patch("agent_mbpp.perf_counter", side_effect=[115, 116, 116, 119]):
            result = self.runner.run_attempt(1, "prompt", None)

        self.assertTrue(result.stop)
        self.assertEqual(self.runner.total_requests, 2)
        self.assertEqual(self.runner.steps[0].retries, 1)
        self.assertEqual(self.runner.steps[0].request_time_ms, 4000)

    def test_retry_blocked_by_deadline_is_not_counted(self):
        self.agent.run.side_effect = [self.rate_limit, self.rate_limit]

        with patch("agent_mbpp.perf_counter", side_effect=[113, 114, 114, 115, 115]):
            result = self.runner.run_attempt(1, "prompt", None)

        self.assertTrue(result.stop)
        self.assertEqual(self.runner.total_requests, 2)
        self.assertEqual(self.runner.steps[0].retries, 1)
        self.assertEqual(self.runner.steps[0].request_time_ms, 2000)

    def test_latency_excludes_wait_between_requests(self):
        self.agent.run.side_effect = [
            self.rate_limit,
            self.rate_limit,
            ModelResponse(text="", input_tokens=3, output_tokens=7),
        ]

        with (
            patch("agent_mbpp.perf_counter", side_effect=[0, 1, 1, 2, 2, 32, 33]),
            patch("agent_mbpp.sleep") as sleep,
        ):
            self.runner.run_attempt(1, "prompt", None)

        sleep.assert_called_once_with(30)
        self.assertEqual(self.runner.total_requests, 3)
        self.assertEqual(self.runner.steps[0].retries, 2)
        self.assertEqual(self.runner.steps[0].request_time_ms, 3000)
