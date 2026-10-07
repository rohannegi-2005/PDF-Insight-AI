"""
agents/prompts.py
-------------------
Shared prompt templates for the agent layer.

Keeping prompts here (rather than inlining prompt strings inside each
agent file) makes them easy to find and tune in one place -- e.g. the
Critic agent added in Step 6 lives in this same "agents" package and
may build on wording defined here.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Research Agent: query rewriting
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a search query optimizer for a research paper retrieval system.

Rewrite the user's question into a clear, keyword-rich search query optimized
for finding relevant passages in academic papers. Remove conversational
filler ("hey", "can you tell me", "I was wondering"), expand obvious
abbreviations, and keep it concise -- a few keywords/phrases is often better
for retrieval than a full grammatical sentence.

Return ONLY the rewritten search query. No explanation, no quotes, nothing else.

Original question: {question}

Rewritten search query:"""
)

QUERY_RETRY_PROMPT = ChatPromptTemplate.from_template(
    """A search for information to answer a question did not return sufficient
evidence. Generate a DIFFERENT search query that approaches the same
question from a new angle, so it has a chance of finding evidence the
first search missed.

Original question: {question}
Previous search query tried: {previous_query}
Why the evidence was insufficient: {reason}

Write a genuinely different search query -- different phrasing, different
keywords, or a narrower/more specific sub-aspect of the question. Do not
just repeat or lightly reword the previous query.

Return ONLY the new search query. No explanation, no quotes, nothing else.

New search query:"""
)

# ---------------------------------------------------------------------------
# Critic Agent: evidence sufficiency check
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """You are an evidence-sufficiency critic for a research question-answering system.

You will be given a QUESTION and a set of retrieved EVIDENCE passages. Decide
whether the evidence is sufficient to write a complete, directly-supported
answer to the question.

Evidence is INSUFFICIENT if:
- it only discusses the topic in general terms, without the specific fact,
  number, or comparison the question actually asks for
- it is relevant to a related but different sub-topic than what was asked
- a key part of the question is not addressed by any passage

Evidence is SUFFICIENT if the passages, taken together, contain enough
specific information to answer the question directly.

Be strict: do not mark evidence sufficient just because it is topically
related to the question."""

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CRITIC_SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nEvidence:\n{evidence}"),
])