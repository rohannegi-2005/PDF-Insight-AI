"""
graph/state.py
----------------
Shared state schema for the LangGraph research workflow.

What is "state" in LangGraph?
--------------------------------
Every node in the graph (Step by step in graph/workflow.py) receives
the current state, does its job, and returns a dict of just the fields
it wants to CHANGE. LangGraph merges that returned dict into the
overall state before handing it to the next node. Nodes never mutate
a shared object directly and never need to know about fields they
don't use -- they just read what they need and return updates.

Using a TypedDict (rather than a plain dict) means editors and type
checkers can catch typos in field names ("documnets" instead of
"documents") before you ever run the code.
"""

from typing import List, Optional, TypedDict

from langchain_core.documents import Document

from agents.critic_agent import EvidenceVerdict


class ResearchState(TypedDict):
    """
    The state object passed between every node in the research graph.
    """

    question: str
    """The original user question. Set once at the start, never changed."""

    search_query: str
    """The search query currently being used. Starts as the rewritten
    version of `question`, and changes each time the Critic asks for a
    retry."""

    documents: List[Document]
    """The most recent retrieval+rerank result for `search_query`."""

    verdict: Optional[EvidenceVerdict]
    """The Critic's most recent judgment on `documents`."""

    retry_count: int
    """How many retries have happened so far. Starts at 0."""

    max_retries: int
    """The retry budget -- see config.MAX_RETRIES."""

    attempts: List[dict]
    """A log of every attempt (search query tried + verdict), purely for
    transparency/debugging -- Step 10's UI will show this to the user so
    they can see the system's reasoning, not just its final answer."""

    final_answer: Optional[str]
    """Populated once the graph decides to answer (either because
    evidence was sufficient, or the retry budget ran out)."""