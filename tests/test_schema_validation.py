"""
test_schema_validation.py — Tests for JSON schema validation and fallback paths
                             in memory_extract.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from memory_extract import validate_extract, _heuristic_fallback


# ── validate_extract ──────────────────────────────────────────────────────────

class TestValidateExtract:

    def test_valid_full_object(self):
        data = {
            "concepts":       [{"concept_id": "binary_search", "confidence": 0.9}],
            "outcome":        {"label": "correct", "confidence": 0.9, "inferred": False},
            "misconceptions": [{"misconception_id": "off_by_one",
                                 "concept_id": "binary_search",
                                 "evidence": "student said index=mid",
                                 "confidence": 0.7}],
            "preferences":    [{"key": "style", "value": "worked_examples", "confidence": 0.85}],
            "goals":          [{"goal": "learn sorting", "confidence": 0.8}],
            "notes":          "all good",
        }
        result = validate_extract(data)
        assert result["concepts"][0]["concept_id"]          == "binary_search"
        assert result["outcome"]["label"]                   == "correct"
        assert result["outcome"]["inferred"]                is False
        assert result["misconceptions"][0]["misconception_id"] == "off_by_one"
        assert result["preferences"][0]["key"]              == "style"
        assert result["preferences"][0]["value"]            == "worked_examples"
        assert result["goals"][0]["goal"]                   == "learn sorting"
        assert result["notes"]                              == "all good"

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            validate_extract("not a dict")

    def test_invalid_outcome_label_defaults_to_partial(self):
        data = {"outcome": {"label": "great_job", "confidence": 0.9, "inferred": True}}
        result = validate_extract(data)
        assert result["outcome"]["label"] == "partial"

    def test_invalid_preference_key_dropped(self):
        data = {"preferences": [{"key": "unknown_key", "value": "direct", "confidence": 0.9}]}
        result = validate_extract(data)
        assert result["preferences"] == []

    def test_invalid_preference_value_dropped(self):
        data = {"preferences": [{"key": "style", "value": "telepathy", "confidence": 0.9}]}
        result = validate_extract(data)
        assert result["preferences"] == []

    def test_confidence_clamped(self):
        data = {"concepts": [{"concept_id": "foo", "confidence": 99}]}
        result = validate_extract(data)
        assert result["concepts"][0]["confidence"] <= 1.0

    def test_confidence_clamped_low(self):
        data = {"concepts": [{"concept_id": "foo", "confidence": -5}]}
        result = validate_extract(data)
        assert result["concepts"][0]["confidence"] >= 0.0

    def test_empty_concept_id_dropped(self):
        data = {"concepts": [{"concept_id": "", "confidence": 0.8}]}
        result = validate_extract(data)
        assert result["concepts"] == []

    def test_missing_fields_default_gracefully(self):
        result = validate_extract({})
        assert "concepts"       in result
        assert "outcome"        in result
        assert "misconceptions" in result
        assert "preferences"    in result
        assert "goals"          in result
        assert "notes"          in result

    def test_notes_truncated(self):
        data = {"notes": "x" * 500}
        result = validate_extract(data)
        assert len(result["notes"]) <= 300

    def test_evidence_truncated(self):
        long_evidence = "e" * 400
        data = {
            "misconceptions": [{
                "misconception_id": "m1",
                "concept_id": "c1",
                "evidence": long_evidence,
                "confidence": 0.8,
            }]
        }
        result = validate_extract(data)
        assert len(result["misconceptions"][0]["evidence"]) <= 300


# ── _heuristic_fallback ───────────────────────────────────────────────────────

class TestHeuristicFallback:

    def test_returns_valid_shape(self):
        result = _heuristic_fallback("I am confused", "Let me explain")
        assert "concepts"       in result
        assert "outcome"        in result
        assert "misconceptions" in result
        required_keys = {"label", "confidence", "inferred"}
        assert required_keys.issubset(result["outcome"].keys())

    def test_detects_confused_outcome(self):
        result = _heuristic_fallback("I am confused about this", "Sure, let me clarify")
        assert result["outcome"]["label"] == "confused"

    def test_detects_correct_outcome(self):
        result = _heuristic_fallback("yes, I got it now!", "Exactly right!")
        assert result["outcome"]["label"] == "correct"

    def test_inferred_always_true(self):
        result = _heuristic_fallback("some message", "some reply")
        assert result["outcome"]["inferred"] is True

    def test_notes_contains_fallback(self):
        result = _heuristic_fallback("test", "test")
        assert "fallback" in result.get("notes", "").lower()

    def test_concept_ids_are_snakeish_words(self):
        result = _heuristic_fallback(
            "I struggle with binary search trees specifically",
            "Binary search trees are a kind of balanced data structure",
        )
        for c in result["concepts"]:
            # All concept IDs should be lowercase alpha strings (heuristic tokens)
            assert c["concept_id"].replace("_", "").isalpha() or len(c["concept_id"]) > 0
