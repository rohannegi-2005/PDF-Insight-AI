"""
graph/workflow.py
--------------------
Two layers of research.

INNER layer: a LangGraph state machine for ONE question
---------------------------------------------------------
    rewrite --> retrieve --+-- (retry found nothing new) -----------------> answer --> END
                           |
                           +--> critic --+-- (sufficient OR out of retries) -> answer
                                         |
                                         +-- (insufficient) --> retry --> retrieve (loop)

OUTER layer: run_multi_research() for a BIG question
------------------------------------------------------
    big question -> split into sub-questions -> run the inner graph for
    each one -> combine the findings into one final answer.

(A manager splits the job, workers keep trying until the job is done or
their retry budget is over, then the manager writes the report.)

Behaviour worth knowing
-----------------------
* Evidence is COLLECTED across retries (without duplicates), so a retry
  adds to what we have instead of replacing it.
* If a retry brings no new chunk, we stop early (saves LLM calls).
* No evidence at all -> an honest fixed answer, no LLM call (so it can
  not invent anything).
* If one sub-question fails (rate limit, network), the others still run.
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

import config
from agents.critic_agent import evaluate_evidence
from agents.prompts import ANSWER_PROMPT, SYNTHESIS_PROMPT
from agents.research_agent import (
    decompose_question,
    rewrite_query,
    rewrite_query_for_retry,
)
from core.rag_pipeline import get_llm
from core.retrieval import chunk_key, hybrid_search_with_rerank
from graph.state import ResearchState

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = (
    "No relevant evidence was found in the documents, so this question "
    "cannot be answered from them."
)


def merge_evidence(
    old: List[Document],
    new: List[Document],
    cap: int = config.MAX_EVIDENCE_CHUNKS,
) -> Tuple[List[Document], int]:
    """
    Add the chunks of a new search to the evidence we already have.

    - a chunk we already have is not added again
    - newest chunks go first (a retry query was written to fill a gap, so
      its results matter most)
    - at most `cap` chunks are kept, so the answer prompt stays small

    Returns (merged evidence, how many chunks were really new).
    """
    old_keys = {chunk_key(doc) for doc in old}

    fresh: List[Document] = []
    fresh_keys = set()
    for doc in new:
        key = chunk_key(doc)
        if key not in old_keys and key not in fresh_keys:
            fresh.append(doc)
            fresh_keys.add(key)

    return (fresh + list(old))[:cap], len(fresh)


def _format_context(documents: List[Document]) -> str:
    """Chunks as one text block, each labelled with its paper and page."""
    blocks = []
    for doc in documents:
        paper = doc.metadata.get("paper_id") or doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{paper}, page {page}] {doc.page_content}")
    return "\n\n".join(blocks)


def _initial_state(question: str, max_retries: int) -> ResearchState:
    """An empty notebook for one run of the inner graph."""
    return {
        "question": question,
        "search_query": "",
        "documents": [],
        "evidence": [],
        "new_evidence_count": 0,
        "verdict": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "attempts": [],
        "final_answer": None,
    }


def build_research_graph(vector_store, sparse_store):
    """
    Compile the INNER graph for one document collection.

    The node functions are defined INSIDE this function (closures) so they
    can use vector_store / sparse_store without global variables.
    """
    papers = sparse_store.paper_ids()

    def rewrite_node(state: ResearchState) -> dict:
        return {"search_query": rewrite_query(state["question"])}

    def retrieve_node(state: ResearchState) -> dict:
        documents = hybrid_search_with_rerank(
            state["search_query"], vector_store, sparse_store
        )
        evidence, new_count = merge_evidence(state["evidence"], documents)
        update = {
            "documents": documents,
            "evidence": evidence,
            "new_evidence_count": new_count,
        }
        if state["retry_count"] > 0 and new_count == 0:
            update["attempts"] = state["attempts"] + [{
                "search_query": state["search_query"],
                "sufficient": False,
                "reason": "This retry found no new evidence, so the search stopped.",
            }]
        return update

    def critic_node(state: ResearchState) -> dict:
        verdict = evaluate_evidence(state["question"], state["evidence"], papers=papers)
        attempt = {
            "search_query": state["search_query"],
            "sufficient": verdict.sufficient,
            "reason": verdict.reason,
        }
        return {"verdict": verdict, "attempts": state["attempts"] + [attempt]}

    def retry_node(state: ResearchState) -> dict:
        new_query = rewrite_query_for_retry(
            question=state["question"],
            previous_query=state["search_query"],
            reason=state["verdict"].reason,
        )
        return {"search_query": new_query, "retry_count": state["retry_count"] + 1}

    def answer_node(state: ResearchState) -> dict:
        if not state["evidence"]:
            return {"final_answer": NO_EVIDENCE_ANSWER}
        chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
        answer = chain.invoke({
            "question": state["question"],
            "context": _format_context(state["evidence"]),
        })
        return {"final_answer": answer}

    def after_retrieve(state: ResearchState) -> str:
        """A retry that found nothing new cannot help: go straight to the answer."""
        if state["retry_count"] > 0 and state["new_evidence_count"] == 0:
            return "answer"
        return "critic"

    def after_critic(state: ResearchState) -> str:
        """Retry only if the evidence is not enough AND the retry budget is not used up."""
        if state["verdict"].sufficient:
            return "answer"
        if state["retry_count"] >= state["max_retries"]:
            return "answer"
        return "retry"

    graph = StateGraph(ResearchState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("critic", critic_node)
    graph.add_node("retry", retry_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, {"critic": "critic", "answer": "answer"})
    graph.add_conditional_edges("critic", after_critic, {"retry": "retry", "answer": "answer"})
    graph.add_edge("retry", "retrieve")
    graph.add_edge("answer", END)

    return graph.compile()


def run_research(
    question: str,
    vector_store,
    sparse_store,
    max_retries: Optional[int] = None,
) -> dict:
    """Run the INNER graph for ONE question. Returns the final state."""
    if max_retries is None:
        max_retries = config.MAX_RETRIES

    app = build_research_graph(vector_store, sparse_store)
    return app.invoke(_initial_state(question, max_retries))


def run_multi_research(
    question: str,
    vector_store,
    sparse_store,
    max_retries: Optional[int] = None,
) -> dict:
    """
    OUTER layer: research a BIG question.

    1. split it into sub-questions (one sub-question for a simple question)
    2. run the inner graph once per sub-question; a sub-question that
       fails is recorded (error=...) and the others continue
    3. combine the findings (skipped when there is only one sub-question)

    Raises RuntimeError only if EVERY sub-question failed.
    """
    if max_retries is None:
        max_retries = config.MAX_RETRIES

    sub_questions = decompose_question(question, papers=sparse_store.paper_ids())
    app = build_research_graph(vector_store, sparse_store)  # compiled once, reused

    sub_results = []
    for sub_question in sub_questions:
        try:
            state = app.invoke(_initial_state(sub_question, max_retries))
            sub_results.append({
                "sub_question": sub_question,
                "answer": state["final_answer"],
                "evidence": state["evidence"],
                "attempts": state["attempts"],
                "retries_used": state["retry_count"],
                "error": None,
            })
        except Exception as error:
            logger.warning("Sub-question failed (%s): %s", sub_question, error)
            sub_results.append({
                "sub_question": sub_question,
                "answer": None,
                "evidence": [],
                "attempts": [],
                "retries_used": 0,
                "error": str(error),
            })

    succeeded = [r for r in sub_results if r["error"] is None]
    if not succeeded:
        raise RuntimeError(
            f"Every sub-question failed. Last error: {sub_results[-1]['error']}"
        )

    if len(sub_results) == 1:
        final_answer = sub_results[0]["answer"]
    else:
        findings = "\n\n".join(
            f"Sub-question {i}: {r['sub_question']}\nFinding: "
            + (r["answer"] if r["error"] is None
               else "(This sub-question could not be researched because of a technical error.)")
            for i, r in enumerate(sub_results, start=1)
        )
        chain = SYNTHESIS_PROMPT | get_llm() | StrOutputParser()
        final_answer = chain.invoke({"question": question, "findings": findings})

    return {
        "question": question,
        "sub_questions": sub_questions,
        "sub_results": sub_results,
        "final_answer": final_answer,
    }