"""
Tests for test_engine.py — adaptive test generation and grading.

All OpenAI calls are mocked so no API key is needed.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from test_engine import (
    DIFFICULTY_QUESTION_TYPES,
    OBJECTIVE_TYPES,
    SUBJECTIVE_TYPES,
    _fallback_question,
    _grade_objective,
    _grade_subjective,
    generate_question,
    generate_test,
    get_difficulty_level,
    grade_test,
)


def _mock_openai_response(content: str):
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Difficulty level tests ───────────────────────────────────────────────────

class TestGetDifficultyLevel(unittest.TestCase):
    def test_zero_mastery_is_easy(self):
        self.assertEqual(get_difficulty_level(0.0), "EASY")

    def test_low_mastery_is_easy(self):
        self.assertEqual(get_difficulty_level(0.2), "EASY")

    def test_boundary_03_is_medium(self):
        self.assertEqual(get_difficulty_level(0.3), "MEDIUM")

    def test_mid_mastery_is_medium(self):
        self.assertEqual(get_difficulty_level(0.5), "MEDIUM")

    def test_boundary_06_is_hard(self):
        self.assertEqual(get_difficulty_level(0.6), "HARD")

    def test_high_mastery_is_hard(self):
        self.assertEqual(get_difficulty_level(0.8), "HARD")

    def test_boundary_085_is_extension(self):
        self.assertEqual(get_difficulty_level(0.85), "EXTENSION")

    def test_full_mastery_is_extension(self):
        self.assertEqual(get_difficulty_level(1.0), "EXTENSION")


class TestDifficultyQuestionTypes(unittest.TestCase):
    def test_easy_has_objective_types(self):
        for qt in DIFFICULTY_QUESTION_TYPES["EASY"]:
            self.assertIn(qt, OBJECTIVE_TYPES)

    def test_medium_has_mixed_types(self):
        types = DIFFICULTY_QUESTION_TYPES["MEDIUM"]
        self.assertTrue(any(t in OBJECTIVE_TYPES for t in types))
        self.assertTrue(any(t in SUBJECTIVE_TYPES for t in types))

    def test_hard_has_subjective_types(self):
        for qt in DIFFICULTY_QUESTION_TYPES["HARD"]:
            self.assertIn(qt, SUBJECTIVE_TYPES)


# ── Fallback question tests ──────────────────────────────────────────────────

class TestFallbackQuestion(unittest.TestCase):
    def test_mcq_fallback_has_options(self):
        q = _fallback_question("sorting", "multiple_choice")
        self.assertEqual(q["type"], "multiple_choice")
        self.assertEqual(len(q["options"]), 4)
        self.assertIsInstance(q["correct_index"], int)

    def test_true_false_fallback(self):
        q = _fallback_question("graphs", "true_false")
        self.assertEqual(q["type"], "true_false")
        self.assertIn("correct_answer", q)

    def test_fill_blank_fallback(self):
        q = _fallback_question("trees", "fill_blank")
        self.assertEqual(q["type"], "fill_blank")
        self.assertIn("_____", q["text"])

    def test_subjective_fallback(self):
        q = _fallback_question("recursion", "explain")
        self.assertEqual(q["type"], "explain")
        self.assertIn("rubric_hint", q)


# ── Objective grading tests ──────────────────────────────────────────────────

class TestGradeObjective(unittest.TestCase):
    def test_mcq_correct(self):
        q = {"type": "multiple_choice", "correct_index": 2, "explanation": "Because C"}
        result = _grade_objective(q, "2")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["correct"])

    def test_mcq_incorrect(self):
        q = {"type": "multiple_choice", "correct_index": 2, "explanation": "Because C"}
        result = _grade_objective(q, "0")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["correct"])

    def test_mcq_invalid_answer(self):
        q = {"type": "multiple_choice", "correct_index": 1, "explanation": ""}
        result = _grade_objective(q, "not_a_number")
        self.assertEqual(result["score"], 0.0)

    def test_true_false_correct(self):
        q = {"type": "true_false", "correct_answer": True, "explanation": ""}
        result = _grade_objective(q, "true")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["correct"])

    def test_true_false_incorrect(self):
        q = {"type": "true_false", "correct_answer": True, "explanation": ""}
        result = _grade_objective(q, "false")
        self.assertEqual(result["score"], 0.0)

    def test_fill_blank_exact_match(self):
        q = {
            "type": "fill_blank", "correct_answer": "HashMap",
            "accept_variations": ["hash map", "hashmap"],
            "explanation": "",
        }
        result = _grade_objective(q, "HashMap")
        self.assertEqual(result["score"], 1.0)

    def test_fill_blank_variation_match(self):
        q = {
            "type": "fill_blank", "correct_answer": "HashMap",
            "accept_variations": ["hash map", "hashmap"],
            "explanation": "",
        }
        result = _grade_objective(q, "hash map")
        self.assertEqual(result["score"], 1.0)

    def test_fill_blank_wrong(self):
        q = {
            "type": "fill_blank", "correct_answer": "HashMap",
            "accept_variations": [],
            "explanation": "",
        }
        result = _grade_objective(q, "LinkedList")
        self.assertEqual(result["score"], 0.0)


# ── Generate question tests ──────────────────────────────────────────────────

class TestGenerateQuestion(unittest.TestCase):
    @patch("test_engine.OpenAI")
    def test_returns_question_with_id(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({
                "type": "multiple_choice",
                "text": "What is O(1)?",
                "options": ["Constant", "Linear", "Quad", "Log"],
                "correct_index": 0,
                "explanation": "O(1) means constant time.",
                "concept_id": "time_complexity",
            })
        )

        q = generate_question("time complexity", "EASY", "multiple_choice")
        self.assertIn("id", q)
        self.assertEqual(q["difficulty"], "EASY")
        self.assertEqual(q["type"], "multiple_choice")

    @patch("test_engine.OpenAI")
    def test_invalid_difficulty_defaults_medium(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"type": "short_answer", "text": "Q?", "concept_id": "x"})
        )

        q = generate_question("test", "INVALID", "short_answer")
        self.assertEqual(q["difficulty"], "MEDIUM")

    @patch("test_engine.OpenAI")
    def test_api_failure_returns_fallback(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.side_effect = Exception("API down")

        q = generate_question("sorting", "EASY", "multiple_choice")
        self.assertIn("id", q)
        self.assertEqual(q["type"], "multiple_choice")
        self.assertIn("options", q)


# ── Generate test tests ──────────────────────────────────────────────────────

class TestGenerateTest(unittest.TestCase):
    @patch("test_engine.OpenAI")
    def test_generates_correct_count(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({
                "type": "multiple_choice",
                "text": "Q?", "options": ["A", "B", "C", "D"],
                "correct_index": 0, "explanation": "X", "concept_id": "y",
            })
        )

        test = generate_test("sorting", num_questions=3, mastery=0.1)
        self.assertEqual(test["num_questions"], 3)
        self.assertEqual(len(test["questions"]), 3)
        self.assertIn("test_id", test)

    @patch("test_engine.OpenAI")
    def test_low_mastery_gets_easy(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({
                "type": "multiple_choice",
                "text": "Q?", "options": ["A", "B", "C", "D"],
                "correct_index": 0, "explanation": "X", "concept_id": "y",
            })
        )

        test = generate_test("arrays", num_questions=2, mastery=0.1)
        self.assertEqual(test["difficulty"], "EASY")

    @patch("test_engine.OpenAI")
    def test_high_mastery_gets_hard(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({
                "type": "explain",
                "text": "Explain BFS vs DFS.",
                "rubric_hint": "Compare traversal strategies",
                "concept_id": "graph_traversal",
            })
        )

        test = generate_test("graphs", num_questions=2, mastery=0.7)
        self.assertEqual(test["difficulty"], "HARD")

    @patch("test_engine.OpenAI")
    def test_num_questions_clamped(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"type": "true_false", "text": "T?", "correct_answer": True, "explanation": "", "concept_id": "x"})
        )

        test = generate_test("test", num_questions=50, mastery=0.0)
        self.assertEqual(test["num_questions"], 10)  # clamped to max 10

    @patch("test_engine.OpenAI")
    def test_has_instructions(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"type": "multiple_choice", "text": "Q?", "options": ["A","B","C","D"], "correct_index": 0, "explanation": "", "concept_id": "x"})
        )

        test = generate_test("test", num_questions=1, mastery=0.0)
        self.assertIn("instructions", test)
        self.assertTrue(len(test["instructions"]) > 0)


# ── Grade test tests ─────────────────────────────────────────────────────────

class TestGradeTest(unittest.TestCase):
    def test_all_correct_objective(self):
        test = {
            "test_id": "abc",
            "questions": [
                {"id": "q1", "type": "multiple_choice", "correct_index": 2, "explanation": ""},
                {"id": "q2", "type": "true_false", "correct_answer": True, "explanation": ""},
            ],
        }
        responses = [
            {"question_id": "q1", "answer": "2"},
            {"question_id": "q2", "answer": "true"},
        ]
        result = grade_test(test, responses)
        self.assertEqual(result["score"], 2.0)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["score_pct"], 100.0)
        self.assertEqual(result["outcome"], "correct")

    def test_all_wrong_objective(self):
        test = {
            "test_id": "abc",
            "questions": [
                {"id": "q1", "type": "multiple_choice", "correct_index": 2, "explanation": ""},
                {"id": "q2", "type": "true_false", "correct_answer": True, "explanation": ""},
            ],
        }
        responses = [
            {"question_id": "q1", "answer": "0"},
            {"question_id": "q2", "answer": "false"},
        ]
        result = grade_test(test, responses)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["outcome"], "incorrect")

    def test_mixed_scores(self):
        test = {
            "test_id": "abc",
            "questions": [
                {"id": "q1", "type": "multiple_choice", "correct_index": 1, "explanation": ""},
                {"id": "q2", "type": "multiple_choice", "correct_index": 3, "explanation": ""},
            ],
        }
        responses = [
            {"question_id": "q1", "answer": "1"},
            {"question_id": "q2", "answer": "0"},
        ]
        result = grade_test(test, responses)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["score_pct"], 50.0)
        self.assertEqual(result["outcome"], "partial")

    def test_missing_response_scores_zero(self):
        test = {
            "test_id": "abc",
            "questions": [
                {"id": "q1", "type": "multiple_choice", "correct_index": 0, "explanation": ""},
            ],
        }
        # No response for q1
        result = grade_test(test, [])
        self.assertEqual(result["score"], 0.0)

    @patch("grading_engine.grade")
    def test_subjective_grading_integration(self, mock_grade):
        """Test that subjective questions call grading_engine."""
        mock_grade.return_value = {
            "overall_score": 7.5,
            "outcome": "partial",
            "strength": "Good reasoning",
            "improvement": "Add more detail",
            "criteria_scores": [],
        }

        test = {
            "test_id": "abc",
            "questions": [
                {"id": "q1", "type": "explain", "text": "Explain BFS", "concept_id": "bfs", "rubric_hint": ""},
            ],
        }
        responses = [
            {"question_id": "q1", "answer": "BFS explores level by level using a queue."},
        ]
        result = grade_test(test, responses, topic="graph_traversal")
        self.assertEqual(result["total"], 1)
        # 7.5/10 = 0.75
        self.assertAlmostEqual(result["question_scores"][0]["score"], 0.75, places=1)

    def test_empty_subjective_answer(self):
        test = {
            "test_id": "abc",
            "questions": [
                {"id": "q1", "type": "short_answer", "text": "Explain", "concept_id": "x"},
            ],
        }
        responses = [
            {"question_id": "q1", "answer": ""},
        ]
        result = grade_test(test, responses)
        self.assertEqual(result["question_scores"][0]["score"], 0.0)


class TestOutcomeMapping(unittest.TestCase):
    """Test that score_pct maps to correct outcome labels."""

    def _make_result(self, n_correct: int, n_total: int) -> dict:
        test = {
            "test_id": "t",
            "questions": [
                {"id": f"q{i}", "type": "true_false", "correct_answer": True, "explanation": ""}
                for i in range(n_total)
            ],
        }
        responses = [
            {"question_id": f"q{i}", "answer": "true" if i < n_correct else "false"}
            for i in range(n_total)
        ]
        return grade_test(test, responses)

    def test_80pct_is_correct(self):
        result = self._make_result(4, 5)
        self.assertEqual(result["outcome"], "correct")

    def test_60pct_is_partial(self):
        result = self._make_result(3, 5)
        self.assertEqual(result["outcome"], "partial")

    def test_20pct_is_incorrect(self):
        result = self._make_result(1, 5)
        self.assertEqual(result["outcome"], "incorrect")


if __name__ == "__main__":
    unittest.main()
