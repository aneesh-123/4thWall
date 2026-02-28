#!/usr/bin/env python3
"""
demo_memory.py — Simulates 5 chat turns against a running server and prints
                 the evolving learner memory state.

Usage:
  1) Start the server:    python 4thWall/server.py
  2) Run this demo:       python demo_memory.py [--base-url http://localhost:5000]

The demo shows:
  - A learner starting with zero mastery on binary search
  - Getting it wrong first (mastery drops / stays low)
  - Improving over turns (mastery rises)
  - Explicitly asking for a worked example (preference updated)
  - Tutor adapting its style based on learner memory
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any

import requests

# ── Colour helpers (ANSI; works on most terminals) ────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"


def _c(col: str, text: str) -> str:
    return f"{col}{text}{_RESET}"


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.3f}"


# ── Turn definitions ──────────────────────────────────────────────────────────
# Each turn: (user_message, outcome_signal or None, description)
TURNS: list[tuple[str, str | None, str]] = [
    (
        "Can you explain how binary search works? I've never seen it before.",
        None,
        "Turn 1 — First encounter with binary search (no outcome yet)",
    ),
    (
        "So binary search always starts from the first element and scans forward?",
        "incorrect",
        "Turn 2 — INCORRECT: learner confuses binary with linear search",
    ),
    (
        "Oh wait, I think binary search keeps halving the search space? "
        "But I'm not totally sure how the midpoint is calculated.",
        "partial",
        "Turn 3 — PARTIAL: learner has the right idea but is unsure of midpoint formula",
    ),
    (
        "Can you show me a concrete worked example of binary search on an array? "
        "Like, step by step please.",
        "partial",
        "Turn 4 — PARTIAL + explicit request for worked example (preference signal)",
    ),
    (
        "Got it! The midpoint is (low + high) // 2, we compare target to arr[mid], "
        "and recurse left if target < arr[mid], right if target > arr[mid]. "
        "I understand it now.",
        "correct",
        "Turn 5 — CORRECT: learner demonstrates clear understanding",
    ),
]


# ── Printing helpers ──────────────────────────────────────────────────────────

def _print_turn_header(n: int, description: str) -> None:
    print()
    print(_c(_BOLD + _CYAN, f"{'━'*70}"))
    print(_c(_BOLD + _CYAN, f"  {description}"))
    print(_c(_BOLD + _CYAN, f"{'━'*70}"))


def _print_request(message: str, outcome_signal: str | None) -> None:
    print(_c(_DIM, f"  ▶ user: {message[:120]}{'…' if len(message) > 120 else ''}"))
    if outcome_signal:
        label_col = {
            "correct": _GREEN, "partial": _YELLOW,
            "confused": _YELLOW, "incorrect": _RED
        }.get(outcome_signal, _DIM)
        print(_c(label_col, f"  ▶ outcome_signal: {outcome_signal}"))


def _print_response(data: dict[str, Any]) -> None:
    assistant = data.get("assistant", "")
    print()
    print(_c(_BOLD, "  Tutor:"))
    # Wrap at 80 chars
    words = assistant.split()
    line = "    "
    for word in words:
        candidate = (line + " " + word) if line != "    " else ("    " + word)
        if len(candidate) > 82:
            print(line)
            line = "    " + word
        else:
            line = candidate
    if line.strip():
        print(line)


def _print_memory(data: dict[str, Any], prev_concepts: dict[str, float]) -> dict[str, float]:
    preview = data.get("memory_preview", {})
    print()
    print(_c(_BOLD, "  📦 Memory State:"))

    # Concepts that have been seen across all turns so far
    all_concepts_now = {
        c["concept_id"]: c["mastery"]
        for c in preview.get("weak_concepts", [])
    }
    # Merge with previously known strong concepts (not returned in weak list)
    # We track them externally via prev_concepts
    merged: dict[str, float] = {**prev_concepts, **all_concepts_now}

    if merged:
        print(_c(_BOLD, "  Concepts:"))
        for cid, mastery in sorted(merged.items(), key=lambda x: x[0]):
            prev = prev_concepts.get(cid, None)
            bar  = _bar(mastery)
            if prev is None:
                change = _c(_CYAN, " (new)")
            else:
                delta = mastery - prev
                if   delta > 0.005: change = _c(_GREEN, f" ▲{delta:+.3f}")
                elif delta < -0.005: change = _c(_RED,   f" ▼{delta:+.3f}")
                else:               change = _c(_DIM,    " (unchanged)")
            print(f"    {cid:<30} {bar}{change}")
    else:
        print(_c(_DIM, "    (no concept data yet)"))

    # Misconceptions
    misconceptions = preview.get("misconceptions", [])
    if misconceptions:
        print(_c(_BOLD, "  Misconceptions:"))
        for m in misconceptions:
            print(_c(_YELLOW, f"    {m['misconception_id']:<35} count={m['count']}"))

    # Preferences
    prefs = preview.get("preferences", {})
    if prefs:
        print(_c(_BOLD, "  Preferences:"))
        for k, v in prefs.items():
            print(f"    {k:<15} {_c(_CYAN, v)}")

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def run_demo(base_url: str) -> None:
    endpoint = f"{base_url.rstrip('/')}/chat/turn"
    learner_id = f"demo-learner-{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())

    print(_c(_BOLD + _GREEN, "\n  4thWall Persistent Learner Memory — Demo Run"))
    print(_c(_DIM, f"  endpoint  : {endpoint}"))
    print(_c(_DIM, f"  learner   : {learner_id}"))
    print(_c(_DIM, f"  session   : {session_id}"))

    prev_concepts: dict[str, float] = {}

    for turn_num, (message, outcome_signal, description) in enumerate(TURNS, start=1):
        _print_turn_header(turn_num, description)
        _print_request(message, outcome_signal)

        payload: dict[str, Any] = {
            "learner_id": learner_id,
            "session_id": session_id,
            "message":    message,
        }
        if outcome_signal:
            payload["outcome_signal"] = outcome_signal

        try:
            resp = requests.post(endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            print(_c(_RED, f"\n  ERROR: Cannot connect to {endpoint}"))
            print(_c(_RED, "  Make sure the server is running:  python 4thWall/server.py"))
            sys.exit(1)
        except requests.exceptions.HTTPError as exc:
            print(_c(_RED, f"\n  HTTP ERROR: {exc}"))
            print(_c(_RED, f"  Response: {resp.text[:400]}"))
            sys.exit(1)
        except Exception as exc:
            print(_c(_RED, f"\n  Unexpected error: {exc}"))
            sys.exit(1)

        _print_response(data)
        prev_concepts = _print_memory(data, prev_concepts)

        if turn_num < len(TURNS):
            print(_c(_DIM, "\n  (sleeping 1s between turns to respect rate limits…)"))
            time.sleep(1)

    print()
    print(_c(_BOLD + _GREEN, f"{'━'*70}"))
    print(_c(_BOLD + _GREEN, "  Demo complete.  5 turns processed."))
    print(_c(_BOLD + _GREEN, f"{'━'*70}"))
    print(_c(_DIM, "\nKey observations to verify:"))
    print("  1. Mastery for 'binary_search' starts at 0, dips on incorrect, rises on correct")
    print("  2. After Turn 4, preferences.style should be 'worked_examples'")
    print("  3. Tutor response in Turn 4 should contain a step-by-step example")
    print("  4. Tutor response in Turn 2 should address the linear-search misconception")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo 5 turns of persistent learner memory")
    parser.add_argument(
        "--base-url",
        default="http://localhost:5000",
        help="Base URL of the running Flask server (default: http://localhost:5000)",
    )
    args = parser.parse_args()
    run_demo(args.base_url)
