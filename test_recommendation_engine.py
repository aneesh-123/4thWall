"""
Tests for recommendation_engine.py — the cubic power law recommendation layer.

Verifies:
  1. Cubic weight formula correctness
  2. Ranking: lower mastery → higher priority
  3. Misconception boosting
  4. Goal alignment boosting
  5. Recency penalty
  6. Mastered-threshold exclusion
  7. Focus percentages sum to ~100%
  8. Integration with memory_update.py profile schema
  9. Edge cases: empty profile, all mastered, single concept
"""

from __future__ import annotations

import time
import pytest
from recommendation_engine import (
    cubic_focus_weight,
    mastery_to_difficulty,
    get_recommendations,
    get_recommendations_dict,
    recommend_next_for_chat_turn,
    EPSILON,
    MISCONCEPTION_BOOST,
    GOAL_BOOST,
    RECENCY_PENALTY,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_profile(
    concepts: dict | None = None,
    misconceptions: dict | None = None,
    goals: list | None = None,
    preferences: dict | None = None,
) -> dict:
    """Build a minimal learner profile matching memory_update.py schema."""
    return {
        "concepts": concepts or {},
        "misconceptions": misconceptions or {},
        "goals": goals or [],
        "preferences": preferences or {"style": "direct", "verbosity": "medium", "pace": "medium"},
    }


def _concept_entry(mastery: float, last_seen: str = "", attempts: int = 5) -> dict:
    return {
        "mastery": mastery,
        "attempts": attempts,
        "correct_attempts": int(attempts * mastery),
        "last_seen": last_seen,
        "last_outcome": "correct" if mastery > 0.5 else "partial",
        "error_pattern": "",
    }


# ── 1. Cubic weight formula ──────────────────────────────────────────────────

class TestCubicFocusWeight:

    def test_zero_mastery(self):
        w = cubic_focus_weight(0.0)
        assert w == pytest.approx(1.0 + EPSILON)

    def test_full_mastery(self):
        w = cubic_focus_weight(1.0)
        assert w == pytest.approx(EPSILON)

    def test_half_mastery(self):
        w = cubic_focus_weight(0.5)
        assert w == pytest.approx(0.125 + EPSILON)

    def test_monotonically_decreasing(self):
        """Higher mastery → lower weight."""
        weights = [cubic_focus_weight(m / 10) for m in range(11)]
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_low_mastery_much_higher_than_mid(self):
        """mastery=0.2 should get ~3.4× more weight than mastery=0.5."""
        w_low = cubic_focus_weight(0.2)
        w_mid = cubic_focus_weight(0.5)
        ratio = w_low / w_mid
        assert ratio > 3.0
        assert ratio < 4.0

    def test_custom_epsilon(self):
        w = cubic_focus_weight(1.0, epsilon=0.01)
        assert w == pytest.approx(0.01)


# ── 2. Difficulty mapping ─────────────────────────────────────────────────────

class TestMasteryToDifficulty:

    def test_low_mastery_easy(self):
        assert mastery_to_difficulty(0.0) == "easy"
        assert mastery_to_difficulty(0.29) == "easy"

    def test_mid_mastery_medium(self):
        assert mastery_to_difficulty(0.3) == "medium"
        assert mastery_to_difficulty(0.59) == "medium"

    def test_high_mastery_hard(self):
        assert mastery_to_difficulty(0.6) == "hard"
        assert mastery_to_difficulty(0.99) == "hard"


# ── 3. Basic recommendations ─────────────────────────────────────────────────

class TestGetRecommendations:

    def test_empty_profile(self):
        recs = get_recommendations(_make_profile())
        assert recs == []

    def test_single_concept(self):
        profile = _make_profile(concepts={
            "trees.binary": _concept_entry(0.3),
        })
        recs = get_recommendations(profile)
        assert len(recs) == 1
        assert recs[0].concept_id == "trees.binary"
        assert recs[0].focus_pct == pytest.approx(100.0, abs=0.1)

    def test_lower_mastery_ranked_first(self):
        profile = _make_profile(concepts={
            "graphs.bfs": _concept_entry(0.7),
            "trees.binary": _concept_entry(0.2),
            "sorting.merge": _concept_entry(0.5),
        })
        recs = get_recommendations(profile)
        assert recs[0].concept_id == "trees.binary"
        assert recs[1].concept_id == "sorting.merge"
        assert recs[2].concept_id == "graphs.bfs"

    def test_focus_percentages_sum_to_100(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.1),
            "b": _concept_entry(0.3),
            "c": _concept_entry(0.5),
            "d": _concept_entry(0.7),
        })
        recs = get_recommendations(profile)
        total = sum(r.focus_pct for r in recs)
        assert total == pytest.approx(100.0, abs=0.5)

    def test_mastered_excluded(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.9),
            "b": _concept_entry(0.3),
        })
        recs = get_recommendations(profile, mastered_threshold=0.85)
        assert len(recs) == 1
        assert recs[0].concept_id == "b"

    def test_top_n_limits_results(self):
        profile = _make_profile(concepts={
            f"concept_{i}": _concept_entry(i * 0.1) for i in range(8)
        })
        recs = get_recommendations(profile, top_n=3)
        assert len(recs) == 3

    def test_concept_ids_filter(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.2),
            "b": _concept_entry(0.3),
            "c": _concept_entry(0.4),
        })
        recs = get_recommendations(profile, concept_ids=["a", "c"])
        ids = [r.concept_id for r in recs]
        assert "b" not in ids
        assert "a" in ids
        assert "c" in ids

    def test_exclude_filter(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.2),
            "b": _concept_entry(0.3),
        })
        recs = get_recommendations(profile, exclude=["a"])
        assert len(recs) == 1
        assert recs[0].concept_id == "b"


# ── 4. Misconception boosting ─────────────────────────────────────────────────

class TestMisconceptionBoost:

    def test_misconception_raises_priority(self):
        """A concept with misconceptions should outrank a lower-mastery concept without."""
        profile = _make_profile(
            concepts={
                "a": _concept_entry(0.4),  # higher mastery but has misconception
                "b": _concept_entry(0.2),  # lower mastery, no misconception
            },
            misconceptions={
                "sign-error": {
                    "concept_id": "a",
                    "count": 3,
                    "last_seen": "2026-02-28T12:00:00+00:00",
                    "evidence": ["forgot negative sign"],
                },
            },
        )
        recs = get_recommendations(profile)
        # 'a' should be first despite higher mastery, due to misconception boost
        assert recs[0].concept_id == "a"

    def test_misconception_hints_populated(self):
        profile = _make_profile(
            concepts={"a": _concept_entry(0.3)},
            misconceptions={
                "causality-reversed": {
                    "concept_id": "a",
                    "count": 2,
                    "last_seen": "2026-02-28T12:00:00+00:00",
                    "evidence": ["outer function changes first"],
                },
            },
        )
        recs = get_recommendations(profile)
        assert len(recs[0].misconception_hints) > 0
        assert "outer function changes first" in recs[0].misconception_hints[0]


# ── 5. Goal alignment ────────────────────────────────────────────────────────

class TestGoalBoost:

    def test_goal_aligned_concept_boosted(self):
        """Goal-aligned concept should rank higher than non-aligned at same mastery."""
        profile = _make_profile(
            concepts={
                "calculus.chain_rule": _concept_entry(0.4),
                "calculus.product_rule": _concept_entry(0.4),
            },
            goals=[{"concept": "calculus.chain_rule", "target_mastery": 0.8}],
        )
        recs = get_recommendations(profile)
        assert recs[0].concept_id == "calculus.chain_rule"
        assert "aligned to learning goal" in recs[0].reason

    def test_string_goals_matched(self):
        """String-type goals should match concept IDs by substring."""
        profile = _make_profile(
            concepts={
                "sorting.merge": _concept_entry(0.3),
                "sorting.quick": _concept_entry(0.3),
            },
            goals=["Master sorting.merge by Friday"],
        )
        recs = get_recommendations(profile)
        assert recs[0].concept_id == "sorting.merge"


# ── 6. Recency penalty ───────────────────────────────────────────────────────

class TestRecencyPenalty:

    def test_recently_seen_deprioritized(self):
        """Concept seen 1 minute ago should rank below one seen 1 hour ago."""
        from datetime import datetime, timezone, timedelta

        now = time.time()
        recent = datetime.fromtimestamp(now - 60, tz=timezone.utc).isoformat()
        old = datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat()

        profile = _make_profile(concepts={
            "a": {**_concept_entry(0.3), "last_seen": recent},
            "b": {**_concept_entry(0.3), "last_seen": old},
        })
        recs = get_recommendations(profile, now=now)
        # 'b' should rank first (not recently seen)
        assert recs[0].concept_id == "b"
        assert "recently seen" in recs[1].reason


# ── 7. All mastered ──────────────────────────────────────────────────────────

class TestAllMastered:

    def test_all_mastered_returns_empty(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.95),
            "b": _concept_entry(0.90),
        })
        recs = get_recommendations(profile, mastered_threshold=0.85)
        assert recs == []

    def test_all_mastered_dict_flag(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.95),
            "b": _concept_entry(0.90),
        })
        result = get_recommendations_dict(profile, mastered_threshold=0.85)
        assert result["all_mastered"] is True
        assert result["recommendations"] == []


# ── 8. Chat turn integration ─────────────────────────────────────────────────

class TestRecommendNextForChatTurn:

    def test_excludes_current_concept(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.2),
            "b": _concept_entry(0.3),
        })
        result = recommend_next_for_chat_turn(profile, current_concept="a")
        ids = [r["concept_id"] for r in result["next_concepts"]]
        assert "a" not in ids
        assert "b" in ids

    def test_hint_template_for_misconceptions(self):
        profile = _make_profile(
            concepts={"a": _concept_entry(0.3)},
            misconceptions={
                "sign-error": {
                    "concept_id": "a",
                    "count": 2,
                    "last_seen": "2026-02-28T12:00:00+00:00",
                    "evidence": ["forgot negative"],
                },
            },
        )
        result = recommend_next_for_chat_turn(profile)
        assert result["next_template"] == "hint"

    def test_lesson_template_for_low_mastery(self):
        profile = _make_profile(concepts={
            "a": _concept_entry(0.1),
        })
        result = recommend_next_for_chat_turn(profile)
        assert result["next_template"] == "lesson"

    def test_empty_profile_returns_reflection(self):
        result = recommend_next_for_chat_turn(_make_profile())
        assert result["next_template"] == "reflection"
        assert result["next_concepts"] == []


# ── 9. Difficulty in recommendations ──────────────────────────────────────────

class TestDifficultyInRecommendations:

    def test_low_mastery_gets_easy(self):
        profile = _make_profile(concepts={"a": _concept_entry(0.1)})
        recs = get_recommendations(profile)
        assert recs[0].difficulty == "easy"

    def test_mid_mastery_gets_medium(self):
        profile = _make_profile(concepts={"a": _concept_entry(0.45)})
        recs = get_recommendations(profile)
        assert recs[0].difficulty == "medium"

    def test_high_mastery_gets_hard(self):
        profile = _make_profile(concepts={"a": _concept_entry(0.7)})
        recs = get_recommendations(profile)
        assert recs[0].difficulty == "hard"


# ── 10. Serialization ─────────────────────────────────────────────────────────

class TestSerialization:

    def test_to_dict_is_json_serializable(self):
        import json
        profile = _make_profile(concepts={
            "a": _concept_entry(0.3),
            "b": _concept_entry(0.5),
        })
        result = get_recommendations_dict(profile)
        serialized = json.dumps(result)
        assert '"concept_id"' in serialized
        assert '"focus_pct"' in serialized
