"""
graph/workflow.py
--------------------
The agentic research loop, built with LangGraph's StateGraph.

Flow
-----
    rewrite --> retrieve --> critic --+-- (sufficient OR out of retries) --> answer --> END
                   ^                  |
                   |                  +-- (insufficient AND retries left) --> retry --+
                   |                                                                  |
                   +----------------------------------------------------------------+

- rewrite:  turn the raw question into a clean first search query
            (agents/research_agent.py: rewrite_query)
- retrieve: dense+sparse hybrid search + cross-encoder reranking
            (core/retrieval.py: hybrid_search_with_rerank)
- critic:   judge whether the retrieved evidence is sufficient
            (agents/critic_agent.py: evaluate_evidence)
- retry:    generate a genuinely different search query and loop back
            (agents/research_agent.py: rewrite_query_for_retry)
- answer:   generate the final answer from whichever evidence we ended
            up with

Why closures instead of global variables for vector_store/sparse_store?
---------------------------------------------------------------------------
Each node function needs access to the vector store and sparse store to
run retrieval, but LangGraph node functions only receive `state` as an
argument -- there's no built-in way to pass extra arguments through.
The common bad fix is a module-level global variable, which is fragile
(not safe if you ever build more than one graph in the same process,
e.g. for two different document collections at once). Instead,
`build_research_graph()` defines the node functions AS CLOSURES inside
itself, so each node function can see `vector_store` and `sparse_store`
from its enclosing scope without any global state at all.
"""

from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

import config
from agents.critic_agent import evaluate_evidence
from agents.prompts import ANSWER_PROMPT
from agents.research_agent import rewrite_query, rewrite_query_for_retry
from core.rag_pipeline import get_llm
from core.retrieval import hybrid_search_with_rerank
from graph.state import ResearchState


def _format_context(documents) -> str:
    """Flatten retrieved chunks into one text block for the answer prompt."""
    return "\n\n".join(
        f"[page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in documents
    )


def build_research_graph(vector_store, sparse_store):
    """
    Compile the LangGraph workflow for a specific vector store + sparse
    store pair (i.e. a specific indexed document collection).
    """

    def rewrite_node(state: ResearchState) -> dict:
        """Entry point: turn the raw question into a clean first search query."""
        query = rewrite_query(state["question"])
        return {"search_query": query}

    def retrieve_node(state: ResearchState) -> dict:
        """Run hybrid search + reranking for the CURRENT search query."""
        documents = hybrid_search_with_rerank(
            state["search_query"], vector_store, sparse_store
        )
        return {"documents": documents}

    def critic_node(state: ResearchState) -> dict:
        """Judge whether the retrieved evidence is sufficient, and log the attempt."""
        verdict = evaluate_evidence(state["question"], state["documents"])

        attempt_record = {
            "search_query": state["search_query"],
            "sufficient": verdict.sufficient,
            "reason": verdict.reason,
        }
        attempts = state.get("attempts", []) + [attempt_record]

        return {"verdict": verdict, "attempts": attempts}

    def retry_node(state: ResearchState) -> dict:
        """After insufficient evidence: generate a different search query and bump the retry count."""
        new_query = rewrite_query_for_retry(
            question=state["question"],
            previous_query=state["search_query"],
            reason=state["verdict"].reason,
        )
        return {
            "search_query": new_query,
            "retry_count": state["retry_count"] + 1,
        }

    def answer_node(state: ResearchState) -> dict:
        """Generate the final answer from whichever evidence we ended up with."""
        llm = get_llm()
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "question": state["question"],
            "context": _format_context(state["documents"]),
        })
        return {"final_answer": answer}

    def should_retry(state: ResearchState) -> str:
        """
        Conditional edge after the Critic: decide whether to loop back
        and retry, or move on to answering.

        Stops retrying once EITHER evidence is sufficient, OR the retry
        budget is used up. The second condition is what guarantees the
        graph always terminates -- without it, a genuinely unanswerable
        question (not covered by the document at all) would loop forever.
        """
        if state["verdict"].sufficient:
            return "answer"
        if state["retry_count"] >= state["max_retries"]:
            return "answer"  # give up gracefully, answer with best-effort evidence
        return "retry"

    graph = StateGraph(ResearchState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("critic", critic_node)
    graph.add_node("retry", retry_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "retry", "answer": "answer"},
    )
    graph.add_edge("retry", "retrieve")
    graph.add_edge("answer", END)

    return graph.compile()


def run_research(
    question: str,
    vector_store,
    sparse_store,
    max_retries: Optional[int] = None,
) -> dict:
    """
    Convenience entry point: build the graph and run it end-to-end for
    one question. Returns the final state dict, which includes
    `final_answer`, `documents` (the evidence actually used), and
    `attempts` (the full retry history, for transparency).
    """
    if max_retries is None:
        max_retries = config.MAX_RETRIES

    app = build_research_graph(vector_store, sparse_store)

    initial_state: ResearchState = {
        "question": question,
        "search_query": "",
        "documents": [],
        "verdict": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "attempts": [],
        "final_answer": None,
    }

    return app.invoke(initial_state)