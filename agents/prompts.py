"""
agents/prompts.py
-------------------
Every prompt the agents use, in one place.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Research Agent: clean up a casual question into a search query
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

# ---------------------------------------------------------------------------
# Research Agent: a DIFFERENT query after a failed search
# ---------------------------------------------------------------------------

QUERY_RETRY_PROMPT = ChatPromptTemplate.from_template(
    """A search for information to answer a question did not return sufficient
evidence. Generate a DIFFERENT search query that approaches the same
question from a new angle, so it has a chance of finding evidence the
first search missed.

Original question: {question}
Previous search query tried: {previous_query}
Why the evidence was insufficient: {reason}

Write a genuinely different search query -- different phrasing, different
keywords, or a narrower/more specific sub-aspect of the question. If the
reason names a paper that is missing, aim the new query at that paper. Do not
just repeat or lightly reword the previous query.

Return ONLY the new search query. No explanation, no quotes, nothing else.

New search query:"""
)

# ---------------------------------------------------------------------------
# Critic Agent: is the evidence enough?
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """You are an evidence-sufficiency critic for a research question-answering system.

You will be given the list of papers in the collection, a QUESTION, and a set of
retrieved EVIDENCE passages. Decide whether the evidence is sufficient to write a
complete, directly-supported answer to the question.

Evidence is INSUFFICIENT if:
- it only discusses the topic in general terms, without the specific fact,
  number, or comparison the question actually asks for
- it is relevant to a related but different sub-topic than what was asked
- a key part of the question is not addressed by any passage
- the question is about SEVERAL papers (for example "each paper" or "compare
  these papers") and the evidence does not cover every paper in the collection.
  Name the missing paper(s) in your reason.

Evidence is SUFFICIENT if the passages, taken together, contain enough
specific information to answer the question directly. A question about one
specific paper only needs evidence from that paper.

Be strict: do not mark evidence sufficient just because it is topically
related to the question."""

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CRITIC_SYSTEM_PROMPT),
    (
        "human",
        "Papers in the collection:\n{papers}\n\nQuestion: {question}\n\nEvidence:\n{evidence}",
    ),
])

# ---------------------------------------------------------------------------
# Final answer for ONE question
# ---------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Use the following retrieved evidence to answer the question. Cite
specific facts from the evidence rather than general statements, and say which
paper each fact comes from. If the evidence is incomplete, answer with what it
does support and note what remains uncertain -- do not fabricate anything that
is not in the evidence.

Evidence:
{context}

Question: {question}

Answer:"""
)

# ---------------------------------------------------------------------------
# Planner: split a big question into sub-questions
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research planner for a collection of research papers.
Break the user's question into the smallest set of focused sub-questions that,
answered together, would fully answer it.

The collection contains these papers:
{papers}

When the user says "these papers", "the documents" or similar, they mean the
papers listed above.

Rules:
- NEVER ask the user a question or ask for clarification. Every sub-question
  must be a research question that can be answered by searching the papers.
- If the question is simple and can be answered directly, return exactly ONE
  sub-question (the question itself, cleaned up).
- For comparison or multi-part questions, return between 2 and {max_sub_questions}
  sub-questions. For comparisons, each sub-question should cover ALL the papers
  at once (for example "What retrieval technique does each paper use?"), not
  one paper at a time.
- Each sub-question must make sense on its own and be searchable on its own.
- Do not add sub-questions that are not needed."""),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Combine the sub-question answers into the final answer
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """You are writing the final answer to a research question. You are given
the original question and a finding for each sub-question. Each finding
already has its own evidence-based answer.

Write one clear, well-organized answer to the ORIGINAL question using ONLY
the findings below. Mention which paper each point comes from when the findings name it. If a finding says the evidence was missing or could not be
researched, say so honestly instead of filling the gap.

Original question: {question}

Findings:
{findings}

Final answer:"""
)