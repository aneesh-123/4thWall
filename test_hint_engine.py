"""
Tests for hint_engine.py — progressive 3-level hint generation.

All OpenAI calls are mocked so no API key is needed.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from hint_engine import (
    MAX_HINT_LEVEL,
    _fallback_hint,
    generate_all_hints,
    generate_hint,
    stream_hints,
    validate_hint_request,
)


def _mock_openai_response(content: str):
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestValidateHintRequest(unittest.TestCase):
    """Tests for validate_hint_request()."""

    def test_valid_request(self):
        ok, err = validate_hint_request({"question_text": "What is a BST?", "level": 1})
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_missing_question_text(self):
        ok, err = validate_hint_request({"level": 1})
        self.assertFalse(ok)
        self.assertIn("question_text", err)

    def test_empty_question_text(self):
        ok, err = validate_hint_request({"question_text": "  ", "level": 1})
        self.assertFalse(ok)
        self.assertIn("question_text", err)

    def test_invalid_level_string(self):
        ok, err = validate_hint_request({"question_text": "Q?", "level": "abc"})
        self.assertFalse(ok)
        self.assertIn("level", err)

    def test_level_too_high(self):
        ok, err = validate_hint_request({"question_text": "Q?", "level": 5})
        self.assertFalse(ok)
        self.assertIn("level", err)

    def test_level_too_low(self):
        ok, err = validate_hint_request({"question_text": "Q?", "level": 0})
        self.assertFalse(ok)
        self.assertIn("level", err)

    def test_default_level(self):
        """When level is missing, default 1 should be valid."""
        ok, err = validate_hint_request({"question_text": "Q?"})
        self.assertTrue(ok)


class TestFallbackHints(unittest.TestCase):
    """Tests for _fallback_hint() static responses."""

    def test_level_1_mentions_topic(self):
        text = _fallback_hint(1, "binary search")
        self.assertIn("binary search", text)

    def test_level_2_mentions_topic(self):
        text = _fallback_hint(2, "recursion")
        self.assertIn("recursion", text)

    def test_level_3_numbered_steps(self):
        text = _fallback_hint(3, "sorting")
        self.assertIn("1.", text)
        self.assertIn("sorting", text)

    def test_no_topic_uses_placeholder(self):
        text = _fallback_hint(1, "")
        self.assertIn("this concept", text)


class TestGenerateHint(unittest.TestCase):
    """Tests for generate_hint() with mocked OpenAI calls."""

    @patch("hint_engine.OpenAI")
    def test_level_1_returns_hint(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            "Have you considered what property makes a tree balanced?"
        )

        result = generate_hint(
            question_text="Define a balanced BST",
            level=1,
            topic="balanced_binary_tree",
        )

        self.assertEqual(result["level"], 1)
        self.assertIn("balanced", result["hint_text"])
        self.assertTrue(result["can_request_next"])
        self.assertEqual(result["topic"], "balanced_binary_tree")

    @patch("hint_engine.OpenAI")
    def test_level_2_with_student_answer(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            "A balanced tree ensures O(log n) operations. Check left and right subtree heights."
        )

        result = generate_hint(
            question_text="Why is balancing important?",
            level=2,
            topic="balanced_binary_tree",
            student_answer="It makes things faster I think",
        )

        self.assertEqual(result["level"], 2)
        self.assertTrue(result["can_request_next"])
        self.assertIn("O(log n)", result["hint_text"])

    @patch("hint_engine.OpenAI")
    def test_level_3_no_next(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            "1. Compare heights of left and right subtrees.\n2. Rotate if diff > 1."
        )

        result = generate_hint(
            question_text="How do you balance an AVL tree?",
            level=3,
            topic="avl_tree",
        )

        self.assertEqual(result["level"], 3)
        self.assertFalse(result["can_request_next"])

    @patch("hint_engine.OpenAI")
    def test_level_clamps_to_valid(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response("Hint text")

        # Level 10 should clamp to 3
        result = generate_hint(question_text="Q?", level=10)
        self.assertEqual(result["level"], 3)

        # Level -1 should clamp to 1
        result = generate_hint(question_text="Q?", level=-1)
        self.assertEqual(result["level"], 1)

    @patch("hint_engine.OpenAI")
    def test_empty_response_falls_back(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response("")

        result = generate_hint(question_text="Q?", level=1, topic="arrays")
        self.assertIn("arrays", result["hint_text"])  # fallback mentions topic

    @patch("hint_engine.OpenAI")
    def test_api_error_falls_back(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.side_effect = Exception("API down")

        result = generate_hint(question_text="Q?", level=2, topic="graphs")
        self.assertEqual(result["level"], 2)
        self.assertIn("graphs", result["hint_text"])  # fallback

    @patch("hint_engine.OpenAI")
    def test_previous_hints_passed(self, mock_cls):
        """Verify previous hints are included in the prompt."""
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response("Level 2 hint")

        generate_hint(
            question_text="Q?",
            level=2,
            topic="stacks",
            previous_hints=["Think about LIFO ordering."],
        )

        call_args = client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        self.assertIn("LIFO", user_msg)


class TestGenerateAllHints(unittest.TestCase):
    """Tests for generate_all_hints() which builds all 3 levels."""

    @patch("hint_engine.OpenAI")
    def test_returns_three_hints(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.side_effect = [
            _mock_openai_response("Level 1 nudge"),
            _mock_openai_response("Level 2 scaffold"),
            _mock_openai_response("Level 3 walkthrough"),
        ]

        hints = generate_all_hints(
            question_text="Explain quicksort",
            topic="sorting",
        )

        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0]["level"], 1)
        self.assertEqual(hints[1]["level"], 2)
        self.assertEqual(hints[2]["level"], 3)
        self.assertTrue(hints[0]["can_request_next"])
        self.assertTrue(hints[1]["can_request_next"])
        self.assertFalse(hints[2]["can_request_next"])

    @patch("hint_engine.OpenAI")
    def test_progressive_context(self, mock_cls):
        """Level 2 and 3 should receive previous hint texts in their prompts."""
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.side_effect = [
            _mock_openai_response("Think about partitioning."),
            _mock_openai_response("Choose a pivot element."),
            _mock_openai_response("1. Pick pivot. 2. Partition. 3. Recurse."),
        ]

        generate_all_hints(question_text="Explain quicksort", topic="sorting")

        # The third call should have both previous hints in its user message
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 3)

        # Level 2 prompt should mention level 1 hint
        l2_user_msg = calls[1].kwargs["messages"][1]["content"]
        self.assertIn("partitioning", l2_user_msg)

        # Level 3 prompt should mention both
        l3_user_msg = calls[2].kwargs["messages"][1]["content"]
        self.assertIn("partitioning", l3_user_msg)
        self.assertIn("pivot", l3_user_msg)


class TestStreamHints(unittest.TestCase):
    """Tests for stream_hints() SSE generator."""

    @patch("hint_engine.OpenAI")
    def test_yields_sse_format(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.side_effect = [
            _mock_openai_response("Nudge"),
            _mock_openai_response("Scaffold"),
            _mock_openai_response("Walkthrough"),
        ]

        events = list(stream_hints(question_text="Q?", topic="test"))

        # 3 hint events + 1 done event
        self.assertEqual(len(events), 4)

        # Each event should start with "data: " and end with "\n\n"
        for event in events:
            self.assertTrue(event.startswith("data: "))
            self.assertTrue(event.endswith("\n\n"))

        # Parse hint events
        for i, event in enumerate(events[:3]):
            payload = json.loads(event[len("data: "):].strip())
            self.assertEqual(payload["level"], i + 1)
            self.assertIn("hint_text", payload)

        # Final done event
        done = json.loads(events[3][len("data: "):].strip())
        self.assertTrue(done["done"])

    @patch("hint_engine.OpenAI")
    def test_handles_partial_failure(self, mock_cls):
        """If one level fails, fallback hint is used and stream continues."""
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.side_effect = [
            _mock_openai_response("Good nudge"),
            Exception("API error"),
            _mock_openai_response("Walkthrough"),
        ]

        events = list(stream_hints(question_text="Q?", topic="trees"))

        # Should still get 4 events (3 hints + done)
        self.assertEqual(len(events), 4)

        # Level 2 should have fallback content
        l2 = json.loads(events[1][len("data: "):].strip())
        self.assertEqual(l2["level"], 2)
        self.assertIn("trees", l2["hint_text"])  # fallback mentions topic


class TestHintLevelProgression(unittest.TestCase):
    """Tests verifying hint levels don't give away too much."""

    @patch("hint_engine.OpenAI")
    def test_level1_system_prompt_no_spoilers(self, mock_cls):
        """Level 1 system prompt should explicitly forbid revealing answers."""
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response("Hint")

        generate_hint(question_text="Q?", level=1)

        call_args = client.chat.completions.create.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        self.assertIn("Do NOT", system_msg)
        self.assertIn("answer", system_msg.lower())

    @patch("hint_engine.OpenAI")
    def test_level3_system_prompt_has_steps(self, mock_cls):
        """Level 3 system prompt should mention breaking into steps."""
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response("Steps")

        generate_hint(question_text="Q?", level=3)

        call_args = client.chat.completions.create.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        self.assertIn("step", system_msg.lower())


if __name__ == "__main__":
    unittest.main()
