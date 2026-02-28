"""
test_mastery_update.py — Unit tests for mastery EMA logic and misconception
                          evidence truncation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from memory_update import update_mastery, apply_extract_to_profile, build_memory_preview

# ── update_mastery ────────────────────────────────────────────────────────────

class TestUpdateMastery:
    def test_correct_raises_mastery(self):
        new = update_mastery(0.0, "correct", alpha=0.2)
        assert abs(new - 0.2) < 1e-9

    def test_incorrect_lowers_mastery(self):
        new = update_mastery(0.8, "incorrect", alpha=0.2)
        assert abs(new - 0.64) < 1e-9

    def test_partial_partial_convergence(self):
        mastery = 0.0
        for _ in range(50):
            mastery = update_mastery(mastery, "partial", alpha=0.2)
        # Should converge toward 0.5
        assert 0.45 < mastery < 0.55

    def test_clamped_at_1(self):
        new = update_mastery(1.0, "correct", alpha=0.5)
        assert new <= 1.0

    def test_clamped_at_0(self):
        new = update_mastery(0.0, "incorrect", alpha=0.5)
        assert new >= 0.0

    def test_unknown_outcome_defaults_to_partial_score(self):
        """Unknown outcome should use partial score (0.5 via fallback)."""
        base = 0.3
        valid_partial = update_mastery(base, "partial", alpha=0.2)
        unknown       = update_mastery(base, "UNKNOWN", alpha=0.2)
        assert unknown == valid_partial

    def test_alpha_configurable(self):
        new_fast = update_mastery(0.0, "correct", alpha=0.5)
        new_slow  = update_mastery(0.0, "correct", alpha=0.1)
        assert new_fast > new_slow


# ── apply_extract_to_profile ──────────────────────────────────────────────────

def _base_profile(learner_id="test") -> dict:
    return {
        "learner_id": learner_id,
        "version": 1,
        "preferences": {"style": "direct", "verbosity": "medium", "pace": "medium"},
        "goals": [],
        "concepts": {},
        "misconceptions": {},
        "recent_summary": "",
        "updated_at": "",
    }

def _minimal_extract(outcome="correct", concepts=None, misc=None, prefs=None, goals=None):
    return {
        "concepts":       concepts or [{"concept_id": "binary_search", "confidence": 0.9}],
        "outcome":        {"label": outcome, "confidence": 0.9, "inferred": False},
        "misconceptions": misc or [],
        "preferences":    prefs or [],
        "goals":          goals or [],
        "notes":          "",
    }


class TestApplyExtract:
    def test_new_concept_created(self):
        profile = _base_profile()
        apply_extract_to_profile(profile, _minimal_extract("correct"))
        assert "binary_search" in profile["concepts"]

    def test_mastery_rises_on_correct(self):
        profile = _base_profile()
        apply_extract_to_profile(profile, _minimal_extract("correct"))
        assert profile["concepts"]["binary_search"]["mastery"] > 0.0

    def test_attempts_incremented(self):
        profile = _base_profile()
        apply_extract_to_profile(profile, _minimal_extract("correct"))
        apply_extract_to_profile(profile, _minimal_extract("partial"))
        assert profile["concepts"]["binary_search"]["attempts"] == 2

    def test_correct_attempts_only_on_correct(self):
        profile = _base_profile()
        apply_extract_to_profile(profile, _minimal_extract("correct"))
        apply_extract_to_profile(profile, _minimal_extract("incorrect"))
        ca = profile["concepts"]["binary_search"]["correct_attempts"]
        assert ca == 1

    def test_mastery_falls_on_incorrect(self):
        profile = _base_profile()
        # Prime with some mastery
        apply_extract_to_profile(profile, _minimal_extract("correct"))
        m_before = profile["concepts"]["binary_search"]["mastery"]
        apply_extract_to_profile(profile, _minimal_extract("incorrect"))
        m_after  = profile["concepts"]["binary_search"]["mastery"]
        assert m_after < m_before

    def test_misconception_evidence_truncated_to_5(self):
        profile = _base_profile()
        for i in range(8):
            extract = _minimal_extract(
                misc=[{
                    "misconception_id": "off_by_one",
                    "concept_id": "binary_search",
                    "evidence": f"evidence_{i}",
                    "confidence": 0.8,
                }]
            )
            apply_extract_to_profile(profile, extract)

        evidence = profile["misconceptions"]["off_by_one"]["evidence"]
        assert len(evidence) <= 5

    def test_misconception_count_increments(self):
        profile = _base_profile()
        misc = [{"misconception_id": "off_by_one", "concept_id": "binary_search",
                 "evidence": "test", "confidence": 0.8}]
        for _ in range(3):
            apply_extract_to_profile(profile, _minimal_extract(misc=misc))
        assert profile["misconceptions"]["off_by_one"]["count"] == 3

    def test_preference_updated_above_threshold(self):
        profile = _base_profile()
        extract = _minimal_extract(
            prefs=[{"key": "style", "value": "worked_examples", "confidence": 0.8}]
        )
        apply_extract_to_profile(profile, extract)
        assert profile["preferences"]["style"] == "worked_examples"

    def test_preference_NOT_updated_below_threshold(self):
        profile = _base_profile()
        extract = _minimal_extract(
            prefs=[{"key": "style", "value": "socratic", "confidence": 0.5}]
        )
        apply_extract_to_profile(profile, extract)
        assert profile["preferences"]["style"] == "direct"   # unchanged

    def test_goals_deduplicated(self):
        profile = _base_profile()
        g = [{"goal": "master sorting algorithms", "confidence": 0.9}]
        apply_extract_to_profile(profile, _minimal_extract(goals=g))
        apply_extract_to_profile(profile, _minimal_extract(goals=g))
        assert profile["goals"].count("master sorting algorithms") == 1

    def test_goals_capped_at_5(self):
        profile = _base_profile()
        for i in range(8):
            g = [{"goal": f"goal number {i}", "confidence": 0.9}]
            apply_extract_to_profile(profile, _minimal_extract(goals=g))
        assert len(profile["goals"]) <= 5

    def test_low_confidence_concept_skipped(self):
        profile = _base_profile()
        extract = _minimal_extract(
            concepts=[{"concept_id": "vague_topic", "confidence": 0.1}]
        )
        apply_extract_to_profile(profile, extract)
        assert "vague_topic" not in profile["concepts"]

    def test_deltas_returned(self):
        profile = _base_profile()
        _, deltas = apply_extract_to_profile(profile, _minimal_extract("correct"))
        assert "binary_search" in deltas
        assert deltas["binary_search"].startswith("+")

    def test_updated_at_set(self):
        profile = _base_profile()
        apply_extract_to_profile(profile, _minimal_extract())
        assert profile["updated_at"] != ""


# ── build_memory_preview ──────────────────────────────────────────────────────

class TestBuildMemoryPreview:
    def test_weak_concepts_only_below_05(self):
        profile = _base_profile()
        profile["concepts"] = {
            "strong_concept": {"mastery": 0.9, "attempts": 5, "correct_attempts": 4,
                               "last_seen": "", "last_outcome": "correct", "error_pattern": ""},
            "weak_concept":   {"mastery": 0.2, "attempts": 3, "correct_attempts": 1,
                               "last_seen": "", "last_outcome": "confused", "error_pattern": ""},
        }
        preview = build_memory_preview(profile)
        ids = [c["concept_id"] for c in preview["weak_concepts"]]
        assert "weak_concept"   in ids
        assert "strong_concept" not in ids

    def test_empty_profile_returns_empty_lists(self):
        profile = _base_profile()
        preview = build_memory_preview(profile)
        assert preview["weak_concepts"]  == []
        assert preview["misconceptions"] == []
