"""
supermemory_client.py — Thin wrapper around the Supermemory Python SDK.

Environment variables:
  SUPERMEMORY_API_KEY     Required. Your Supermemory API key.
  SUPER_MEMORY_BASE_URL   Optional. Override base URL (for self-hosted).

Naming conventions for tags used as pseudo-keys:
  profile:{learner_id}   — the single profile document per learner
  event:{learner_id}     — event/turn documents (many per learner)
  learner:{learner_id}   — broad container tag on ALL learner docs
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from supermemory import Supermemory

logger = logging.getLogger(__name__)

# ── Default profile skeleton ──────────────────────────────────────────────────

def _default_profile(learner_id: str) -> dict:
    return {
        "learner_id": learner_id,
        "version": 1,
        "preferences": {
            "style": "direct",
            "verbosity": "medium",
            "pace": "medium",
        },
        "goals": [],
        "concepts": {},
        "misconceptions": {},
        "recent_summary": "",
        "updated_at": "",
    }


# ── Client ────────────────────────────────────────────────────────────────────

class MemoryClient:
    """
    A learner-scoped wrapper around Supermemory.

    All documents for a learner share the container tag ``learner:{learner_id}``.
    The profile document also carries the tag ``profile:{learner_id}`` so it
    can be retrieved and replaced individually.
    """

    def __init__(self) -> None:
        self._client = Supermemory()   # reads SUPERMEMORY_API_KEY from env

    # ── Profile ───────────────────────────────────────────────────────────────

    @staticmethod
    def _doc_content(doc: Any) -> str:
        """Extract raw text from a Supermemory search result document."""
        # Try direct content/document field first
        for attr in ("content", "document"):
            val = getattr(doc, attr, None)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        # Fall back to chunks (list of chunk objects or strings)
        chunks = getattr(doc, "chunks", None) or []
        parts: list[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                parts.append(chunk)
            else:
                for ca in ("content", "text", "document"):
                    cv = getattr(chunk, ca, None)
                    if cv and isinstance(cv, str):
                        parts.append(cv)
                        break
        return " ".join(parts).strip()

    def get_profile(self, learner_id: str) -> dict[str, Any]:
        """
        Fetch the stored learner profile. Returns a default profile dict if
        nothing is found (first visit).
        """
        tag = f"profile:{learner_id}"
        try:
            results = self._client.search.documents(
                q=f"learner profile {learner_id}",
                container_tags=[tag],
                limit=1,
            )
            docs = getattr(results, "results", None) or []
            if docs:
                raw = self._doc_content(docs[0])
                if raw:
                    profile = json.loads(raw)
                    logger.debug("get_profile hit for %s", learner_id)
                    return profile
                else:
                    logger.debug("get_profile: result found but content empty for %s", learner_id)
        except Exception as exc:
            logger.warning("get_profile search failed for %s: %s", learner_id, exc)

        logger.debug("get_profile miss for %s — initialising default", learner_id)
        return _default_profile(learner_id)

    def put_profile(self, learner_id: str, profile: dict[str, Any]) -> None:
        """
        Overwrite the stored profile. 
        Strategy: delete any existing profile docs, then insert fresh.
        """
        tag = f"profile:{learner_id}"
        learner_tag = f"learner:{learner_id}"

        # 1) Delete old profile docs
        try:
            results = self._client.search.documents(
                q=f"learner profile {learner_id}",
                container_tags=[tag],
                limit=10,
            )
            docs = getattr(results, "results", None) or []
            for doc in docs:
                doc_id = getattr(doc, "id", None) or getattr(doc, "document_id", None)
                if doc_id:
                    try:
                        self._client.documents.delete(doc_id)
                        logger.debug("Deleted old profile doc %s for %s", doc_id, learner_id)
                    except Exception as del_exc:
                        logger.warning("Could not delete old profile doc %s: %s", doc_id, del_exc)
        except Exception as exc:
            logger.warning("put_profile pre-delete search failed for %s: %s", learner_id, exc)

        # 2) Insert fresh profile
        try:
            self._client.add(
                content=json.dumps(profile, ensure_ascii=False),
                container_tags=[tag, learner_tag],
            )
            logger.debug("put_profile stored for %s", learner_id)
        except Exception as exc:
            logger.error("put_profile insert failed for %s: %s", learner_id, exc)
            raise

    # ── Events ────────────────────────────────────────────────────────────────

    def add_event(self, learner_id: str, event: dict[str, Any]) -> None:
        """
        Append a turn-event document. Events accumulate; old ones are never
        deleted automatically.
        """
        event_tag = f"event:{learner_id}"
        learner_tag = f"learner:{learner_id}"
        try:
            self._client.add(
                content=json.dumps(event, ensure_ascii=False),
                container_tags=[event_tag, learner_tag],
            )
            logger.debug("add_event stored for %s", learner_id)
        except Exception as exc:
            logger.error("add_event failed for %s: %s", learner_id, exc)
            raise

    def query_recent_events(
        self, learner_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Return up to `limit` recent event dicts, newest-first where possible.
        Returns empty list on any error.
        """
        tag = f"event:{learner_id}"
        try:
            results = self._client.search.documents(
                q=f"session turn event {learner_id}",
                container_tags=[tag],
                limit=limit,
            )
            docs = getattr(results, "results", None) or []
            events = []
            for doc in docs:
                raw = self._doc_content(doc)
                if raw:
                    try:
                        events.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
            return events
        except Exception as exc:
            logger.warning("query_recent_events failed for %s: %s", learner_id, exc)
            return []


# ── Module-level singleton (lazy) ─────────────────────────────────────────────

_client_instance: MemoryClient | None = None


def get_memory_client() -> MemoryClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = MemoryClient()
    return _client_instance
