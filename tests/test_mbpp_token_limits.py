import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_mbpp import MBPPRunner, SYSTEM_PROMPT, TokenBudgetError
from src.APIs.groq import GroqAPI
from src.APIs.openrouter import OpenRouterAPI
from structures import MBPPTaskInput, ModelResponse


class MBPPTokenLimitsTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock(model_name="test-model")
        self.agent.client.base_url = "https://example.test"
        self.task = MBPPTaskInput(
            task_id=1,
            task_definition="Return one.",
            function_definition="def one():",
            test_list=["assert one() == 1"],
        )
        self.runner = MBPPRunner(self.agent, self.task)
        # Make the input boundaries deterministic without downloading a vocabulary.
        tokenizer = Mock()
        tokenizer.encode_ordinary.side_effect = (
            lambda text: [1] if text == SYSTEM_PROMPT else [2, 3]
        )
        encoding = patch("agent_mbpp.tiktoken.get_encoding", return_value=tokenizer)
        encoding.start()
        self.addCleanup(encoding.stop)

    def test_input_that_fits_exactly_is_sent(self):
        self.runner.total_input_tokens = 5997
        self.agent.run.return_value = ModelResponse(text="", input_tokens=3, output_tokens=1)

        self.runner.request_solution("prompt")
        self.assertEqual(self.runner.total_input_tokens, 6000)
        with self.assertRaisesRegex(TokenBudgetError, "input token budget exhausted"):
            self.runner.request_solution("prompt")
        self.assertEqual(self.agent.run.call_count, 1)

    def test_input_one_token_over_budget_is_blocked(self):
        self.runner.total_input_tokens = 5998

        with self.assertRaisesRegex(TokenBudgetError, "Estimated prompt size exceeds"):
            self.runner.request_solution("prompt")
        self.agent.run.assert_not_called()
        self.assertEqual(self.runner.total_requests, 0)

    def test_empty_response_still_consumes_output_budget(self):
        self.agent.run.side_effect = [
            ModelResponse(text="", input_tokens=3, output_tokens=420),
            ModelResponse(text="", input_tokens=3, output_tokens=1080),
        ]

        self.runner.request_solution("first prompt")
        self.runner.request_solution("second prompt")
        caps = [call.kwargs["max_output_tokens"] for call in self.agent.run.call_args_list]
        self.assertEqual(caps, [1500, 1080])
        self.assertEqual(self.runner.total_output_tokens, 1500)
        with self.assertRaisesRegex(TokenBudgetError, "output token budget exhausted"):
            self.runner.request_solution("third prompt")
        self.assertEqual(self.agent.run.call_count, 2)

    def test_solution_at_exact_output_limit_is_still_tested(self):
        self.agent.run.return_value = ModelResponse(
            text="def one(): return 1", input_tokens=3, output_tokens=1500,
        )
        sandbox = Mock(return_value=(True, "Tests passed."))

        with patch.object(self.runner, "load_docker_helper", return_value=sandbox):
            result = self.runner.agent_cycle()

        self.assertTrue(result.success)
        self.assertEqual(result.total_output_tokens, 1500)
        sandbox.assert_called_once()
        self.agent.run.assert_called_once()

    def test_exhausted_budget_preserves_candidate_in_saved_result(self):
        self.agent.run.return_value = ModelResponse(
            text="def one(): return 0", input_tokens=3, output_tokens=1500,
        )
        sandbox = Mock(return_value=(False, "AssertionError"))

        with patch.object(self.runner, "load_docker_helper", return_value=sandbox):
            result = self.runner.agent_cycle()
        with tempfile.TemporaryDirectory() as directory:
            output_file = Path(directory) / "solution.json"
            self.runner.save_solution(str(output_file), result)
            saved = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertFalse(saved["success"])
        self.assertEqual(saved["solution"], "def one(): return 0")
        self.assertEqual(saved["total_requests"], 1)
        self.assertEqual(saved["total_output_tokens"], 1500)
        self.assertIn("output token budget exhausted", saved["error"])
        request = self.agent.run.call_args
        self.assertEqual(saved["system_prompt"], request.kwargs["system_prompt"])
        self.assertIn(self.task.task_definition, request.args[0])
        self.assertNotIn("AssertionError", saved["system_prompt"])

    def test_invalid_python_is_reported_without_running_sandbox(self):
        self.agent.run.side_effect = [
            ModelResponse(text="You can solve this with a loop.", input_tokens=3, output_tokens=7),
            ModelResponse(text="def one(): return 1", input_tokens=3, output_tokens=7),
        ]
        sandbox = Mock(return_value=(True, "Tests passed."))

        with patch.object(self.runner, "load_docker_helper", return_value=sandbox):
            result = self.runner.agent_cycle()

        self.assertTrue(result.success)
        sandbox.assert_called_once()
        self.assertIn("No valid Python code found", self.agent.run.call_args.args[0])
        self.assertEqual(result.steps[0].sandbox_input, "")

    def test_recovered_markdown_is_executed_and_explained_to_model(self):
        self.agent.run.side_effect = [
            ModelResponse(text="```python\ndef one(): return 0", input_tokens=3, output_tokens=7),
            ModelResponse(text="```python\ndef one(): return 1\n```", input_tokens=3, output_tokens=7),
        ]
        sandbox = Mock(side_effect=[(False, "AssertionError"), (True, "Tests passed.")])

        with patch.object(self.runner, "load_docker_helper", return_value=sandbox):
            result = self.runner.agent_cycle()

        self.assertTrue(result.success)
        self.assertIn("def one(): return 0", sandbox.call_args_list[0].args[0])
        self.assertIn("no closing Markdown fence", self.agent.run.call_args.args[0])
        self.assertIn("AssertionError", self.agent.run.call_args.args[0])


class ProviderOutputLimitsTests(unittest.TestCase):
    def test_groq_forwards_output_limit(self):
        with patch.dict("os.environ", {"GROQ_API_KEY1": "test-key"}), patch("src.APIs.groq.OpenAI"):
            api = GroqAPI("test-model", "https://example.test")
        create = api.client.responses.create
        create.return_value = SimpleNamespace(
            output_text="code",
            usage=SimpleNamespace(input_tokens=3, output_tokens=7),
        )

        api.run("prompt", system_prompt=SYSTEM_PROMPT, max_output_tokens=1080)
        self.assertEqual(create.call_args.kwargs["max_output_tokens"], 1080)
        self.assertEqual(create.call_args.kwargs["instructions"], SYSTEM_PROMPT)
        self.assertEqual(create.call_args.kwargs["input"], "prompt")

    def test_openrouter_forwards_output_limit(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY1": "test-key"}), patch("src.APIs.openrouter.OpenAI"):
            api = OpenRouterAPI("test-model", "https://example.test")
        create = api.client.chat.completions.create
        create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="code"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=7),
        )

        api.run("prompt", system_prompt=SYSTEM_PROMPT, max_output_tokens=1080)
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 1080)
        self.assertEqual(create.call_args.kwargs["messages"], [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "prompt"},
        ])
