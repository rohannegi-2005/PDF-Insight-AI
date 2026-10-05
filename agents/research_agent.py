"""
agents/research_agent.py
--------------------------
The Research Agent: responsible for turning a raw user question into a
better search query before retrieval runs.

Two jobs in this file:

1. rewrite_query() -- clean up a user's raw, conversational question
   into a clear, keyword-rich search query. People type questions the
   way they'd ask a friend ("hey can you tell me a bit about..."),
   which includes filler words that don't help a retriever match
   relevant passages. This strips that down to the actual information
   need.

2. rewrite_query_for_retry() -- used when a FIRST search attempt didn't
   find enough evidence (this is what Step 6's Critic agent will flag,
   and Step 7's LangGraph loop will call). Instead of just tidying up
   the same question, this generates a genuinely DIFFERENT search
   angle -- since re-running the exact same search would just retrieve
   the exact same (already-insufficient) results again.

Why is this its own "agent" file and not just a retrieval helper?
---------------------------------------------------------------------
Everything through Step 4 (chunking, dense/sparse retrieval, fusion,
reranking) is DETERMINISTIC -- the same input always produces the same
output via fixed algorithms, no judgment involved. This file is the
first place an LLM makes a DECISION that changes what happens next
(what to search for), rather than just writing a grounded answer from
already-retrieved text. That distinction -- deciding vs. describing --
is what actually makes something an "agent" in this project, not just
a bigger prompt.

Note: get_llm() is reused from core/rag_pipeline.py rather than
creating a second Groq client here -- one shared LLM factory function,
used by both the answer-generation chain and the agent layer.
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from agents.prompts import QUERY_REWRITE_PROMPT, QUERY_RETRY_PROMPT
from core.rag_pipeline import get_llm


def rewrite_query(question: str, llm: Optional[BaseChatModel] = None) -> str:
    """
    Clean up a user's raw question into a clear, retrieval-friendly
    search query.

    Example:
        "hey can you tell me a bit about how RAGFusion-7 works?"
        -> "RAGFusion-7 methodology hybrid retrieval"
    """
    if llm is None:
        llm = get_llm()

    chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question})
    return rewritten.strip()


def rewrite_query_for_retry(
    question: str,
    previous_query: str,
    reason: str,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """
    Generate a genuinely different search query after a first retrieval
    attempt didn't find enough evidence to answer the question.

    Parameters
    ----------
    question : str
        The original user question (unchanged throughout retries).
    previous_query : str
        The search query that was already tried and came up short.
    reason : str
        A short explanation of WHY the evidence was insufficient --
        this will come from the Critic agent in Step 6 (e.g. "the
        retrieved chunks discuss the method in general but don't
        mention this specific comparison"). Giving the LLM this reason
        produces a much more useful retry than just saying "try again."
    """
    if llm is None:
        llm = get_llm()

    chain = QUERY_RETRY_PROMPT | llm | StrOutputParser()
    new_query = chain.invoke({
        "question": question,
        "previous_query": previous_query,
        "reason": reason,
    })
    return new_query.strip()