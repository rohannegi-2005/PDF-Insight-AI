"""
graph/state.py
----------------
The "notebook" that travels through the LangGraph workflow.

Each node reads what it needs and returns ONLY the fields it wants to
change; LangGraph merges that into the state before the next node runs.
"""

from typing import List, Optional, TypedDict

from langchain_core.documents import Document

from agents.critic_agent import EvidenceVerdict


class ResearchState(TypedDict):
    question: str
    """The question being researched. Set once, never changed."""

    search_query: str
    """The query used for the CURRENT search (changes on every retry)."""

    documents: List[Document]
    """What the latest search returned (kept for debugging)."""

    evidence: List[Document]
    """ALL evidence gathered so far, across retries, without duplicates.
    The Critic judges this, and the final answer is written from this."""

    new_evidence_count: int
    """How many chunks the latest search added that we did not have yet.
    If a retry adds 0, searching again would not help, so we stop."""

    verdict: Optional[EvidenceVerdict]
    """The Critic's latest judgment on `evidence`."""

    retry_count: int
    """How many retries have happened. Starts at 0."""

    max_retries: int
    """The retry budget (config.MAX_RETRIES)."""

    attempts: List[dict]
    """Log of every attempt (query + verdict), for transparency."""

    final_answer: Optional[str]
    """The answer. Filled in by the last node."""