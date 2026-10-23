"""
agents/research_agent.py
--------------------------
The Research Agent: the part that PLANS the search.

Three jobs in this file:

1. rewrite_query()
     Clean a casual question into a short, keyword-style search query.

2. rewrite_query_for_retry()
     When the first search did not find enough evidence, write a
     DIFFERENT search query (re-running the same one gives the same
     weak results).

3. decompose_question()
     Break one big question ("compare these papers...") into small
     sub-questions. Each sub-question is then researched on its own.
     For a simple question, the LLM returns just ONE sub-question, so
     simple questions are not made more complicated than needed.

Note: get_llm() is reused from core/rag_pipeline.py, so the whole
project shares one LLM setup.
"""

import logging
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

import config
from agents.prompts import DECOMPOSE_PROMPT, QUERY_REWRITE_PROMPT, QUERY_RETRY_PROMPT
from agents.structured import invoke_structured
from core.rag_pipeline import get_llm

logger = logging.getLogger(__name__)


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
    rewritten = chain.invoke({"question": question}).strip()
    
    # If the LLM returns nothing, searching for the original question is
    # still better than searching for an empty string.
    return rewritten or question


def rewrite_query_for_retry(
    question: str,
    previous_query: str,
    reason: str,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """
    Generate a different search query after a first attempt did not
    find enough evidence. `reason` comes from the Critic agent and tells
    the LLM WHAT was missing, which gives a much better retry than just
    saying "try again".
    """
    if llm is None:
        llm = get_llm()

    chain = QUERY_RETRY_PROMPT | llm | StrOutputParser()
    new_query = chain.invoke({
        "question": question,
        "previous_query": previous_query,
        "reason": reason,
    }).strip()
    
    # Fallback to the previous query if the LLM output is empty
    return new_query or previous_query


class SubQuestionPlan(BaseModel):
    """The shape of the planner's answer, as a fixed form."""

    sub_questions: List[str] = Field(
        description=(
            "Focused, self-contained research sub-questions. Exactly one item "
            "if the question is simple; several if it is a comparison or "
            "multi-part question. Never a question addressed to the user."
        )
    )


def decompose_question(
    question: str,
    papers: Optional[List[str]] = None,
    max_sub_questions: Optional[int] = None,
    llm: Optional[BaseChatModel] = None,
) -> List[str]:
    """
    Split a big question into smaller sub-questions.

    Example:
        "Compare the retrieval techniques in these papers"
        -> ["What retrieval technique does each paper use?",
            "Which techniques improved retrieval, and by how much?",
            "What limitations does each paper report?"]

    `papers` = names of the papers in the collection. The planner needs
    them to understand "these papers"; without them it may ask the user a
    question back instead of planning.

    Safety: always returns at least one question and at most
    max_sub_questions. If the planner fails even after its retry, we fall
    back to the ORIGINAL question as one sub-question (the research still
    runs, just without splitting) and log a warning.
    """
    if max_sub_questions is None:
        max_sub_questions = config.MAX_SUB_QUESTIONS

    papers_text = "\n".join(f"- {name}" for name in papers) if papers else "(not provided)"

    try:
        plan = invoke_structured(
            DECOMPOSE_PROMPT,
            SubQuestionPlan,
            {
                "question": question,
                "papers": papers_text,
                "max_sub_questions": max_sub_questions,
            },
            llm=llm,
        )
    except Exception as error:
        logger.warning("Question planner failed (%s); using the original question.", error)
        return [question]

    sub_questions = [q.strip() for q in plan.sub_questions if q.strip()]
    if not sub_questions:
        return [question]
        
    return sub_questions[:max_sub_questions]