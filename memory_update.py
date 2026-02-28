"""
memory_update.py — Deterministic learner profile updates from extractor output.

All state transitions are pure functions (no I/O) so they are trivially testable.

Mastery EMA update rule:
  score = 1.0 (correct) | 0.5 (partial) | 0.25 (confused) | 0.0 (incorrect)
  alpha = MASTERY_ALPHA env var (default 0.2)
  mastery_new = mastery_old + alpha * (score - mastery_old)
  clamped to [0.0, 1.0]
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MASTERY_ALPHA: float = float(os.getenv("MASTERY_ALPHA", "0.2"))

_OUTCOME_SCORES: dict[str, float] = {
    "correct":   1.0,
    "partial":   0.5,
    "confused":  0.25,
    "incorrect": 0.0,
}

_MAX_MISCONCEPTION_EVIDENCE = 5
_MAX_GOALS                  = 5
_PREFERENCE_CONFIDENCE_THRESHOLD = 0.75


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Mastery update (exported for unit tests) ──────────────────────────────────

def update_mastery(
    current_mastery: float,
    outcome_label: str,
    alpha: float = MASTERY_ALPHA,
) -> float:
    """
    EMA-based mastery update.

    Parameters
    ----------
    current_mastery : float in [0.0, 1.0]
    outcome_label   : one of correct | partial | confused | incorrect
    alpha           : EMA learning rate (default MASTERY_ALPHA)

    Returns
    -------
    Updated mastery clamped to [0.0, 1.0].
    """
    score = _OUTCOME_SCORES.get(outcome_label, 0.5)
    new_mastery = current_mastery + alpha * (score - current_mastery)
    return max(0.0, min(1.0, new_mastery))


# ── Profile updater ───────────────────────────────────────────────────────────

def apply_extract_to_profile(
    profile: dict[str, Any],
    extract:  dict[str, Any],
    alpha:    float = MASTERY_ALPHA,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Apply validated extractor output to a profile. Returns:
      - updated_profile  (new dict, profile argument is mutated in-place AND returned)
      - deltas           (dict concept_id → "+X.XX" mastery delta strings for logging)

    Side-effect: profile["updated_at"] is refreshed.
    """
    now = _now_iso()
    outcome = extract.get("outcome", {})
    outcome_label = outcome.get("label", "partial")
    deltas: dict[str, str] = {}

    # ── Concepts ──────────────────────────────────────────────────────────────
    concepts_node = profile.setdefault("concepts", {})
    for c in extract.get("concepts", []):
        cid        = c["concept_id"]
        confidence = c.get("confidence", 0.5)
        if confidence < 0.2:           # too uncertain to record
            continue

        entry = concepts_node.get(cid)
        if entry is None:
            entry = {
                "mastery":          0.0,
                "attempts":         0,
                "correct_attempts": 0,
                "last_seen":        now,
                "last_outcome":     outcome_label,
                "error_pattern":    "",
            }
            concepts_node[cid] = entry

        old_mastery = entry["mastery"]
        new_mastery = update_mastery(old_mastery, outcome_label, alpha)

        entry["attempts"]        += 1
        entry["correct_attempts"] += (1 if outcome_label == "correct" else 0)
        entry["mastery"]         = new_mastery
        entry["last_seen"]       = now
        entry["last_outcome"]    = outcome_label

        delta = new_mastery - old_mastery
        deltas[cid] = f"{delta:+.3f}"
        logger.info(
            "concept %-30s mastery %.3f → %.3f (%s%s)",
            cid, old_mastery, new_mastery,
            "↑" if delta > 0 else "↓" if delta < 0 else "=",
            f"{abs(delta):.3f}",
        )

    # ── Concept edges (co-occurrence knowledge graph) ─────────────────────────
    # Every pair of concepts appearing in the same turn gets a bidirectional
    # co-occurrence edge, weighted by how many turns they share.
    edges_node = profile.setdefault("concept_edges", {})
    concept_ids_this_turn = [
        c["concept_id"]
        for c in extract.get("concepts", [])
        if c.get("confidence", 0) >= 0.2
    ]
    for i, cid_a in enumerate(concept_ids_this_turn):
        for cid_b in concept_ids_this_turn[i + 1:]:
            edges_node.setdefault(cid_a, {})
            edges_node.setdefault(cid_b, {})
            edges_node[cid_a][cid_b] = edges_node[cid_a].get(cid_b, 0) + 1
            edges_node[cid_b][cid_a] = edges_node[cid_b].get(cid_a, 0) + 1

    # ── Misconceptions ────────────────────────────────────────────────────────
    misc_node = profile.setdefault("misconceptions", {})
    for m in extract.get("misconceptions", []):
        mid      = m["misconception_id"]
        cid      = m["concept_id"]
        evidence = m.get("evidence", "")[:200]
        conf     = m.get("confidence", 0.0)
        if conf < 0.3:
            continue

        entry = misc_node.get(mid)
        if entry is None:
            entry = {
                "concept_id": cid,
                "count":      0,
                "last_seen":  now,
                "evidence":   [],
            }
            misc_node[mid] = entry

        entry["count"]    += 1
        entry["last_seen"] = now
        if evidence and evidence not in entry["evidence"]:
            entry["evidence"].append(evidence)
            if len(entry["evidence"]) > _MAX_MISCONCEPTION_EVIDENCE:
                entry["evidence"] = entry["evidence"][-_MAX_MISCONCEPTION_EVIDENCE:]

        logger.info("misconception %-30s count=%d", mid, entry["count"])

    # ── Preferences ───────────────────────────────────────────────────────────
    prefs_node = profile.setdefault("preferences", {
        "style": "direct", "verbosity": "medium", "pace": "medium"
    })
    for p in extract.get("preferences", []):
        key  = p.get("key", "")
        val  = p.get("value", "")
        conf = p.get("confidence", 0.0)
        if key and val and conf >= _PREFERENCE_CONFIDENCE_THRESHOLD:
            old_val = prefs_node.get(key, "?")
            prefs_node[key] = val
            if old_val != val:
                logger.info("preference %s: %s → %s (conf=%.2f)", key, old_val, val, conf)

    # ── Goals ─────────────────────────────────────────────────────────────────
    goals_list: list[str] = profile.setdefault("goals", [])
    for g in extract.get("goals", []):
        goal_text = g.get("goal", "").strip()
        conf      = g.get("confidence", 0.0)
        if goal_text and conf >= 0.5 and goal_text not in goals_list:
            goals_list.append(goal_text)
    # Keep top 5
    profile["goals"] = goals_list[:_MAX_GOALS]

    # ── Timestamps ────────────────────────────────────────────────────────────
    profile["updated_at"] = now

    return profile, deltas


# ── Memory-preview builder ────────────────────────────────────────────────────

def build_memory_preview(profile: dict[str, Any]) -> dict[str, Any]:
    """
    Build the compact memory_preview dict returned in /chat/turn responses.
    Includes full graph data (nodes + edges) for the knowledge-graph UI.
    """
    concepts = profile.get("concepts", {})
    misconceptions = profile.get("misconceptions", {})
    edges_raw = profile.get("concept_edges", {})

    weak = sorted(
        [{"concept_id": cid, "mastery": round(v["mastery"], 3)}
         for cid, v in concepts.items() if v["mastery"] < 0.5],
        key=lambda x: x["mastery"],
    )[:5]

    misconceptions_preview = sorted(
        [{"misconception_id": mid, "count": v["count"]}
         for mid, v in misconceptions.items()],
        key=lambda x: -x["count"],
    )[:3]

    # ── Knowledge graph: nodes ────────────────────────────────────────────────
    misc_by_concept: set[str] = {
        v["concept_id"] for v in misconceptions.values()
    }
    nodes = [
        {
            "id":              cid,
            "mastery":         round(v["mastery"], 3),
            "attempts":        v.get("attempts", 0),
            "last_outcome":    v.get("last_outcome", ""),
            "has_misconception": cid in misc_by_concept,
        }
        for cid, v in concepts.items()
    ]

    # ── Knowledge graph: edges (deduplicated) ─────────────────────────────────
    seen: set[tuple[str, str]] = set()
    links: list[dict] = []
    for cid_a, targets in edges_raw.items():
        if cid_a not in concepts:
            continue
        for cid_b, weight in targets.items():
            if cid_b not in concepts:
                continue
            key = tuple(sorted([cid_a, cid_b]))
            if key in seen:
                continue
            seen.add(key)
            links.append({"source": cid_a, "target": cid_b, "weight": weight})

    return {
        "weak_concepts":   weak,
        "misconceptions":  misconceptions_preview,
        "preferences":     profile.get("preferences", {}),
        "graph":           {"nodes": nodes, "links": links},
    }
