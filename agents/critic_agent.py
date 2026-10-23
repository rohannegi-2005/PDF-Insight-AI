"""
agents/critic_agent.py
------------------------
The Critic Agent: decides whether retrieved evidence is actually
sufficient to answer the question, or whether the system should
rewrite the query (agents/research_agent.py's rewrite_query_for_retry)
and search again.

Why structured output instead of free text?
----------------------------------------------
If we just asked the LLM "is this evidence enough? answer yes or no"
and read back a plain string, calling code would have to parse that
string to decide what to do next ("if 'yes' in response.lower(): ...").
That's fragile -- the LLM might say "Yes, this seems sufficient" or
"Sufficient: yes" or any number of phrasings, and a brittle string
check can silently misread the verdict.

Instead, `invoke_structured` forces the LLM's response into a
fixed Pydantic schema (EvidenceVerdict below). The calling code can
then just check `verdict.sufficient` (a real Python bool), and
`verdict.reason` always exists in a predictable place. This is the
current, non-deprecated LangChain pattern for getting reliable,
machine-readable output from an LLM, rather than regex-parsing prose.

This file only builds and tests the judgment itself. Step 7 wires this
into an actual retry LOOP using LangGraph -- this step keeps it as a
plain, directly-callable function so it can be tested and understood
in isolation first.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from agents.prompts import CRITIC_PROMPT
from agents.structured import invoke_structured


class EvidenceVerdict(BaseModel):
    """Structured judgment produced by the Critic agent."""

    sufficient: bool = Field(
        description=(
            "True if the evidence contains enough specific information to "
            "directly and completely answer the question. False otherwise."
        )
    )
    reason: str = Field(
        description=(
            "A short (1-2 sentence) explanation of the verdict: what the "
            "evidence does or does not cover. If a paper is missing, name it."
        )
    )


def format_evidence(documents: List[Document]) -> str:
    """
    Turn a list of retrieved chunks into a single numbered, labeled
    block of text for the Critic's prompt. Numbering + page/paper labels make
    it possible to later extend this (e.g. "insufficient, missing
    passage 3's claim") without changing the data structure.
    """
    if not documents:
        return "(no evidence retrieved)"

    blocks = []
    for i, doc in enumerate(documents, start=1):
        paper = doc.metadata.get("paper_id") or doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Passage {i} | {paper} | page {page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def evaluate_evidence(
    question: str,
    documents: List[Document],
    papers: Optional[List[str]] = None,
    llm: Optional[BaseChatModel] = None,
) -> EvidenceVerdict:
    """
    Judge whether `documents` are enough to answer `question`.
    
    `papers` is the list of paper names in the collection; with it the
    Critic can notice that a paper is missing from the evidence.
    
    Returns a structured EvidenceVerdict (not a plain string), so
    calling code (the LangGraph retry loop in Step 7) can branch on
    `verdict.sufficient` directly instead of parsing free text.
    """
    papers_text = "\n".join(f"- {name}" for name in papers) if papers else "(not provided)"
    
    return invoke_structured(
        CRITIC_PROMPT,
        EvidenceVerdict,
        {
            "question": question,
            "evidence": format_evidence(documents),
            "papers": papers_text,
        },
        llm=llm,
    )