"""
test_grading_engine.py — Tests for the transparent dual-mode grading engine.

Tests cover:
  - Rubric validation and normalization
  - Grade result validation and normalization
  - Default/fallback rubric generation
  - Fallback grading when LLM fails
  - Outcome inference from scores
  - Weight normalization
  - Input sanitization (truncation, clamping)
  - grade() router function (mocked LLM)
  - grade_informal() and grade_formal() (mocked LLM)
  - generate_rubric() (mocked LLM)

All tests mock OpenAI — no API key needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from grading_engine import (
    OUTCOME_SCORES,
    _default_rubric,
    _fallback_grade_result,
    _validate_grade_result,
    _validate_rubric,
    generate_rubric,
    grade,
    grade_formal,
    grade_informal,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    return mock_resp


def _sample_rubric() -> dict:
    return {
        "criteria": [
            {
                "name": "Core Definition",
                "weight": 0.4,
                "formal_requirement": "Precise definition with terminology",
                "informal_requirement": "Shows general understanding",
            },
            {
                "name": "Reasoning",
                "weight": 0.35,
                "formal_requirement": "Logical chain with citations",
                "informal_requirement": "Reasoning in right direction",
            },
            {
                "name": "Examples",
                "weight": 0.25,
                "formal_requirement": "Names specific examples",
                "informal_requirement": "Gives any example",
            },
        ],
        "sources": [
            {"type": "concept_graph", "reference": "From PDF page 47"},
            {"type": "authoritative", "reference": "CLRS Chapter 13"},
        ],
        "total_points": 10,
    }


def _sample_grade_result() -> dict:
    return {
        "criteria_scores": [
            {"name": "Core Definition", "score": 3, "max_score": 4, "met": True,
             "feedback": "Good understanding shown"},
            {"name": "Reasoning", "score": 2, "max_score": 4, "met": True,
             "feedback": "Partial reasoning provided"},
            {"name": "Examples", "score": 1, "max_score": 4, "met": False,
             "feedback": "No examples given"},
        ],
        "overall_score": 6.5,
        "outcome": "partial",
        "strength": "Good grasp of core concept",
        "improvement": "Add specific examples",
        "misconceptions_detected": ["height_vs_depth"],
        "sources_used": ["From PDF page 47", "CLRS Chapter 13"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Outcome scores match memory_update.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutcomeScores:
    def test_correct_score(self):
        assert OUTCOME_SCORES["correct"] == 1.0

    def test_partial_score(self):
        assert OUTCOME_SCORES["partial"] == 0.5

    def test_confused_score(self):
        assert OUTCOME_SCORES["confused"] == 0.25

    def test_incorrect_score(self):
        assert OUTCOME_SCORES["incorrect"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Rubric validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateRubric:
    def test_valid_rubric_passes_through(self):
        rubric = _sample_rubric()
        result = _validate_rubric(rubric)
        assert len(result["criteria"]) == 3
        assert result["total_points"] == 10
        assert len(result["sources"]) == 2

    def test_weights_normalized_to_one(self):
        rubric = {
            "criteria": [
                {"name": "A", "weight": 2.0, "formal_requirement": "x", "informal_requirement": "y"},
                {"name": "B", "weight": 3.0, "formal_requirement": "x", "informal_requirement": "y"},
            ],
            "sources": [],
            "total_points": 10,
        }
        result = _validate_rubric(rubric)
        total = sum(c["weight"] for c in result["criteria"])
        assert abs(total - 1.0) < 0.05  # rounding tolerance

    def test_empty_criteria_returns_default(self):
        result = _validate_rubric({"criteria": [], "sources": []})
        assert len(result["criteria"]) == 3  # default rubric has 3 criteria

    def test_max_five_criteria(self):
        rubric = {
            "criteria": [
                {"name": f"C{i}", "weight": 0.1, "formal_requirement": "x", "informal_requirement": "y"}
                for i in range(10)
            ],
            "sources": [],
            "total_points": 10,
        }
        result = _validate_rubric(rubric)
        assert len(result["criteria"]) <= 5

    def test_weight_clamped_to_zero_one(self):
        rubric = {
            "criteria": [
                {"name": "A", "weight": -5.0, "formal_requirement": "x", "informal_requirement": "y"},
                {"name": "B", "weight": 100.0, "formal_requirement": "x", "informal_requirement": "y"},
            ],
            "sources": [],
            "total_points": 10,
        }
        result = _validate_rubric(rubric)
        for c in result["criteria"]:
            assert 0.0 <= c["weight"] <= 1.0

    def test_name_truncated(self):
        rubric = {
            "criteria": [
                {"name": "A" * 200, "weight": 1.0, "formal_requirement": "x", "informal_requirement": "y"},
            ],
            "sources": [],
            "total_points": 10,
        }
        result = _validate_rubric(rubric)
        assert len(result["criteria"][0]["name"]) <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Grade result validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateGradeResult:
    def test_valid_result_passes(self):
        result = _validate_grade_result(_sample_grade_result())
        assert result["outcome"] == "partial"
        assert result["overall_score"] == 6.5
        assert len(result["criteria_scores"]) == 3

    def test_score_clamped_to_range(self):
        result = _validate_grade_result({
            "criteria_scores": [{"name": "A", "score": 99, "met": True, "feedback": "ok"}],
            "overall_score": 15.0,
            "outcome": "correct",
        })
        assert result["overall_score"] == 10.0
        assert result["criteria_scores"][0]["score"] == 4  # max per-criterion

    def test_negative_score_clamped(self):
        result = _validate_grade_result({
            "criteria_scores": [{"name": "A", "score": -5, "met": False, "feedback": ""}],
            "overall_score": -3.0,
            "outcome": "incorrect",
        })
        assert result["overall_score"] == 0.0
        assert result["criteria_scores"][0]["score"] == 0

    def test_invalid_outcome_inferred_from_score_correct(self):
        result = _validate_grade_result({
            "overall_score": 9.0,
            "outcome": "INVALID",
        })
        assert result["outcome"] == "correct"

    def test_invalid_outcome_inferred_from_score_partial(self):
        result = _validate_grade_result({
            "overall_score": 6.0,
            "outcome": "BOGUS",
        })
        assert result["outcome"] == "partial"

    def test_invalid_outcome_inferred_from_score_confused(self):
        result = _validate_grade_result({
            "overall_score": 3.5,
            "outcome": "???",
        })
        assert result["outcome"] == "confused"

    def test_invalid_outcome_inferred_from_score_incorrect(self):
        result = _validate_grade_result({
            "overall_score": 1.0,
            "outcome": "nope",
        })
        assert result["outcome"] == "incorrect"

    def test_misconceptions_truncated(self):
        result = _validate_grade_result({
            "overall_score": 5.0,
            "outcome": "partial",
            "misconceptions_detected": [f"m{i}" for i in range(20)],
        })
        assert len(result["misconceptions_detected"]) <= 5

    def test_empty_misconceptions_filtered(self):
        result = _validate_grade_result({
            "overall_score": 5.0,
            "outcome": "partial",
            "misconceptions_detected": ["valid", "", None, "also_valid"],
        })
        assert "" not in result["misconceptions_detected"]
        assert len(result["misconceptions_detected"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Default rubric
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefaultRubric:
    def test_has_three_criteria(self):
        rubric = _default_rubric("recursion", "What is recursion?")
        assert len(rubric["criteria"]) == 3

    def test_weights_sum_to_one(self):
        rubric = _default_rubric("sorting", "Compare quicksort and mergesort")
        total = sum(c["weight"] for c in rubric["criteria"])
        assert total == 1.0

    def test_has_sources(self):
        rubric = _default_rubric("trees", "What is a BST?")
        assert len(rubric["sources"]) >= 1

    def test_topic_in_requirements(self):
        rubric = _default_rubric("linked lists", "What is a linked list?")
        core = rubric["criteria"][0]
        assert "linked lists" in core["formal_requirement"]
        assert "linked lists" in core["informal_requirement"]


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Fallback grade result
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackGradeResult:
    def test_nonempty_submission_gets_partial(self):
        result = _fallback_grade_result("This is my answer about binary trees")
        assert result["outcome"] == "partial"
        assert result["overall_score"] == 5.0

    def test_empty_submission_gets_incorrect(self):
        result = _fallback_grade_result("")
        assert result["outcome"] == "incorrect"
        assert result["overall_score"] == 0.0

    def test_very_short_submission_gets_incorrect(self):
        result = _fallback_grade_result("hi")
        assert result["outcome"] == "incorrect"

    def test_has_required_fields(self):
        result = _fallback_grade_result("some answer")
        assert "criteria_scores" in result
        assert "overall_score" in result
        assert "outcome" in result
        assert "misconceptions_detected" in result
        assert "sources_used" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Test: generate_rubric (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateRubric:
    @patch("grading_engine.OpenAI")
    def test_returns_validated_rubric(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps(_sample_rubric())
        )

        rubric = generate_rubric("binary trees", "What is a balanced BST?")
        assert len(rubric["criteria"]) == 3
        assert rubric["total_points"] == 10

    @patch("grading_engine.OpenAI")
    def test_falls_back_on_llm_error(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        rubric = generate_rubric("trees", "What is a tree?")
        assert len(rubric["criteria"]) == 3  # default rubric


# ═══════════════════════════════════════════════════════════════════════════════
# Test: grade_informal (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradeInformal:
    @patch("grading_engine.OpenAI")
    def test_returns_validated_result(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps(_sample_grade_result())
        )

        result = grade_informal(
            submission="A balanced tree has equal sides",
            topic="balanced binary trees",
            question_text="What is a balanced BST?",
            rubric=_sample_rubric(),
        )
        assert result["outcome"] == "partial"
        assert 0 <= result["overall_score"] <= 10

    @patch("grading_engine.OpenAI")
    def test_falls_back_on_error(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("timeout")

        result = grade_informal(
            submission="some answer here that is long enough",
            topic="trees",
            question_text="What is a tree?",
            rubric=_sample_rubric(),
        )
        assert result["outcome"] == "partial"  # fallback for non-empty


# ═══════════════════════════════════════════════════════════════════════════════
# Test: grade_formal (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradeFormal:
    @patch("grading_engine.OpenAI")
    def test_returns_validated_result(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        formal_result = _sample_grade_result()
        formal_result["overall_score"] = 4.0
        formal_result["outcome"] = "confused"

        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps(formal_result)
        )

        result = grade_formal(
            submission="A tree where both sides are the same",
            topic="balanced binary trees",
            question_text="Define a balanced binary search tree.",
            rubric=_sample_rubric(),
        )
        assert result["outcome"] == "confused"
        assert result["overall_score"] == 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: grade() router (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradeRouter:
    @patch("grading_engine.OpenAI")
    def test_informal_mode(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: rubric gen, second call: grading
        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(json.dumps(_sample_rubric())),
            _mock_openai_response(json.dumps(_sample_grade_result())),
        ]

        result = grade(
            submission="Balanced means equal height on both sides",
            topic="balanced binary trees",
            question_text="What is a balanced BST?",
            mode="informal",
        )
        assert result["mode"] == "informal"
        assert "rubric" in result
        assert result["outcome"] == "partial"

    @patch("grading_engine.OpenAI")
    def test_formal_mode(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        formal_result = _sample_grade_result()
        formal_result["outcome"] = "correct"
        formal_result["overall_score"] = 9.0

        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(json.dumps(_sample_rubric())),
            _mock_openai_response(json.dumps(formal_result)),
        ]

        result = grade(
            submission="A balanced binary tree is defined as a tree where the height difference between left and right subtrees of every node is at most 1, guaranteeing O(log n) operations. Examples include AVL and Red-Black trees.",
            topic="balanced binary trees",
            question_text="Define a balanced binary search tree.",
            mode="formal",
        )
        assert result["mode"] == "formal"
        assert result["outcome"] == "correct"
        assert result["overall_score"] == 9.0

    @patch("grading_engine.OpenAI")
    def test_result_includes_rubric(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(json.dumps(_sample_rubric())),
            _mock_openai_response(json.dumps(_sample_grade_result())),
        ]

        result = grade(
            submission="test answer",
            topic="trees",
            question_text="What is a tree?",
        )
        assert "rubric" in result
        assert "criteria" in result["rubric"]
        assert "sources" in result["rubric"]

    @patch("grading_engine.OpenAI")
    def test_misconceptions_propagated(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        grade_result = _sample_grade_result()
        grade_result["misconceptions_detected"] = ["height_vs_depth", "bst_vs_bt"]

        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response(json.dumps(_sample_rubric())),
            _mock_openai_response(json.dumps(grade_result)),
        ]

        result = grade(
            submission="a tree that is balanced",
            topic="binary trees",
            question_text="What is a BST?",
        )
        assert "height_vs_depth" in result["misconceptions_detected"]
        assert "bst_vs_bt" in result["misconceptions_detected"]
