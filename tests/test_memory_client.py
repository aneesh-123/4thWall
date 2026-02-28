"""
test_memory_client.py — Tests for MemoryClient with a fully-mocked
                         Supermemory SDK.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import unittest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_search_result(content: str) -> MagicMock:
    doc = MagicMock()
    doc.content = content
    doc.id      = "mock-id-123"
    return doc


def _make_search_response(docs: list) -> MagicMock:
    resp = MagicMock()
    resp.results = docs
    return resp


DEFAULT_PROFILE = {
    "learner_id": "alice",
    "version": 1,
    "preferences": {"style": "direct", "verbosity": "medium", "pace": "medium"},
    "goals": [],
    "concepts": {},
    "misconceptions": {},
    "recent_summary": "",
    "updated_at": "",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMemoryClientGetProfile(unittest.TestCase):

    @patch("supermemory_client.Supermemory")
    def test_returns_default_on_empty_results(self, MockSupermemory):
        sdk = MockSupermemory.return_value
        sdk.search.execute.return_value = _make_search_response([])

        from supermemory_client import MemoryClient
        client = MemoryClient()
        profile = client.get_profile("alice")

        assert profile["learner_id"] == "alice"
        assert profile["concepts"] == {}

    @patch("supermemory_client.Supermemory")
    def test_returns_stored_profile(self, MockSupermemory):
        stored = {**DEFAULT_PROFILE, "recent_summary": "stored value"}
        sdk = MockSupermemory.return_value
        sdk.search.execute.return_value = _make_search_response(
            [_make_search_result(json.dumps(stored))]
        )

        from supermemory_client import MemoryClient
        client = MemoryClient()
        profile = client.get_profile("alice")

        assert profile["recent_summary"] == "stored value"

    @patch("supermemory_client.Supermemory")
    def test_returns_default_on_search_exception(self, MockSupermemory):
        sdk = MockSupermemory.return_value
        sdk.search.execute.side_effect = ConnectionError("network error")

        from supermemory_client import MemoryClient
        client = MemoryClient()
        profile = client.get_profile("alice")

        assert profile["learner_id"] == "alice"


class TestMemoryClientPutProfile(unittest.TestCase):

    @patch("supermemory_client.Supermemory")
    def test_deletes_old_then_adds_new(self, MockSupermemory):
        sdk = MockSupermemory.return_value
        old_doc = _make_search_result(json.dumps(DEFAULT_PROFILE))
        sdk.search.execute.return_value = _make_search_response([old_doc])
        sdk.memories.delete.return_value = None
        sdk.add.return_value = None

        from supermemory_client import MemoryClient
        client = MemoryClient()
        client.put_profile("alice", {**DEFAULT_PROFILE, "recent_summary": "updated"})

        sdk.memories.delete.assert_called_once_with("mock-id-123")
        sdk.add.assert_called_once()
        call_kwargs = sdk.add.call_args
        content = call_kwargs[1].get("content") or call_kwargs[0][0]
        parsed = json.loads(content)
        assert parsed["recent_summary"] == "updated"

    @patch("supermemory_client.Supermemory")
    def test_add_called_with_correct_tags(self, MockSupermemory):
        sdk = MockSupermemory.return_value
        sdk.search.execute.return_value = _make_search_response([])
        sdk.add.return_value = None

        from supermemory_client import MemoryClient
        client = MemoryClient()
        client.put_profile("alice", DEFAULT_PROFILE)

        tags = sdk.add.call_args[1].get("container_tags") or sdk.add.call_args[0][1]
        assert "profile:alice" in tags
        assert "learner:alice" in tags


class TestMemoryClientAddEvent(unittest.TestCase):

    @patch("supermemory_client.Supermemory")
    def test_event_stored(self, MockSupermemory):
        sdk = MockSupermemory.return_value
        sdk.add.return_value = None

        from supermemory_client import MemoryClient
        client = MemoryClient()
        event = {"ts": "2026-02-28T00:00:00Z", "session_id": "s1",
                 "outcome": "correct", "concepts": ["binary_search"],
                 "inferred_outcome": True,
                 "user_message_summary": "hi", "tutor_message_summary": "hello"}
        client.add_event("alice", event)

        sdk.add.assert_called_once()
        tags = sdk.add.call_args[1].get("container_tags") or sdk.add.call_args[0][1]
        assert "event:alice" in tags
        assert "learner:alice" in tags


class TestMemoryClientQueryRecentEvents(unittest.TestCase):

    @patch("supermemory_client.Supermemory")
    def test_returns_parsed_events(self, MockSupermemory):
        event1 = {"ts": "2026-02-27", "outcome": "correct"}
        event2 = {"ts": "2026-02-28", "outcome": "partial"}
        sdk = MockSupermemory.return_value
        sdk.search.execute.return_value = _make_search_response([
            _make_search_result(json.dumps(event1)),
            _make_search_result(json.dumps(event2)),
        ])

        from supermemory_client import MemoryClient
        client = MemoryClient()
        events = client.query_recent_events("alice", limit=5)

        assert len(events) == 2
        assert events[0]["outcome"] == "correct"

    @patch("supermemory_client.Supermemory")
    def test_returns_empty_list_on_error(self, MockSupermemory):
        sdk = MockSupermemory.return_value
        sdk.search.execute.side_effect = RuntimeError("fail")

        from supermemory_client import MemoryClient
        client = MemoryClient()
        events = client.query_recent_events("alice")

        assert events == []


if __name__ == "__main__":
    unittest.main()
