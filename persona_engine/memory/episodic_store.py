"""
Episodic Store — compressed conversation summaries.

Stores semantic summaries of conversation segments, NOT verbatim
transcripts. This prevents persona drift from exact memory replay.
"""

from __future__ import annotations

from persona_engine.memory.models import Episode, MemorySource, MemoryType


class EpisodicStore:
    """
    Stores compressed conversation episode summaries.

    Design principles:
    - Never store verbatim user text (privacy + drift prevention)
    - Compress to semantic summary: topic, outcome, key points
    - Episodes are per-topic segments, not per-turn
    - Recency-weighted retrieval
    """

    def __init__(self) -> None:
        self._episodes: list[Episode] = []
        self._by_topic: dict[str, list[Episode]] = {}
        self._by_conversation: dict[str, list[Episode]] = {}

    def store(self, episode: Episode) -> None:
        """Store a new episode summary."""
        self._episodes.append(episode)

        topic = episode.topic.lower()
        if topic not in self._by_topic:
            self._by_topic[topic] = []
        self._by_topic[topic].append(episode)

        conv_id = episode.conversation_id
        if conv_id not in self._by_conversation:
            self._by_conversation[conv_id] = []
        self._by_conversation[conv_id].append(episode)

    def get_by_topic(self, topic: str, limit: int = 5) -> list[Episode]:
        """
        Get episodes about a topic, most recent first.

        Args:
            topic: Topic to search for
            limit: Max episodes to return

        Returns:
            Matching episodes, most recent first
        """
        candidates = self._by_topic.get(topic.lower(), [])
        return sorted(candidates, key=lambda e: e.turn_end, reverse=True)[:limit]

    def get_by_conversation(self, conversation_id: str) -> list[Episode]:
        """Get all episodes from a specific conversation."""
        return self._by_conversation.get(conversation_id, [])

    def search(self, query: str, limit: int = 5) -> list[Episode]:
        """
        Search episodes by keyword in content, topic, or outcome.

        Args:
            query: Keyword to search for
            limit: Max results

        Returns:
            Matching episodes, most recent first
        """
        query_lower = query.lower()
        results = []
        for episode in self._episodes:
            if (
                query_lower in episode.content.lower()
                or query_lower in episode.topic.lower()
                or query_lower in episode.outcome.lower()
            ):
                results.append(episode)
        return sorted(results, key=lambda e: e.turn_end, reverse=True)[:limit]

    def get_recent(self, limit: int = 5) -> list[Episode]:
        """Get the N most recent episodes across all conversations."""
        return sorted(self._episodes, key=lambda e: e.turn_end, reverse=True)[:limit]

    def has_discussed(self, topic: str) -> bool:
        """Check if a topic has been discussed before."""
        return topic.lower() in self._by_topic

    @property
    def count(self) -> int:
        return len(self._episodes)

    @property
    def topics(self) -> list[str]:
        return list(self._by_topic.keys())
