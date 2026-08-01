"""
core/rag_pipeline.py
----------------------
Prompt templates, the Groq LLM, and the conversational LCEL chain.

Architecture
------------
This upgrades the original single-turn LCEL chain into a **conversational,
citation-returning** chain using LangChain's standard "conversational RAG"
pattern (retriever + history-aware rewriting + RunnableWithMessageHistory):

    1. history_aware_retriever
         Rewrites a follow-up question ("what about her?") into a
         standalone question ("what about Kathy?") using the chat
         history, THEN retrieves documents for that standalone question.
         Without this step, follow-up questions retrieve irrelevant
         chunks because the retriever itself has no memory.

    2. question_answer_chain  (create_stuff_documents_chain)
         "Stuffs" the retrieved documents into the prompt as context and
         asks the LLM to answer, grounded only in that context.

    3. rag_chain  (create_retrieval_chain)
         Wires the two together. Its output is a **dict**, not a plain
         string:
             {"input": ..., "chat_history": ..., "context": [...], "answer": "..."}
         The "context" key (the retrieved Document objects) is what
         powers Source Citations in the UI, instead of only getting back
         a plain answer string like the original prototype did.

    4. RunnableWithMessageHistory
         Wraps step 3 so chat history is automatically loaded before each
         call and appended after -- this is the Conversational Memory
         feature. History is keyed by `session_id`, so multiple users /
         browser tabs never see each other's conversations.

Streaming
---------
Because the chain is a LangChain Runnable, `.stream()` works out of the
box. Chunks arrive as partial dicts: "context" arrives once (as soon as
retrieval finishes), then "answer" arrives incrementally, token-by-token,
as the LLM generates. `stream_answer()` below turns that into a plain
string generator (for Streamlit's `st.write_stream`) while separately
capturing the source documents as a side effect.
"""

import os
from typing import Dict, Generator, List

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_MODEL_NAME = "llama-3.1-8b-instant"  # Groq's fast Llama-3.1 8B model


def get_llm() -> ChatGroq:
    """
    Build the Groq chat model.

    temperature=0 keeps answers deterministic and tightly grounded in the
    retrieved context, which matters for a RAG system where hallucination
    is the main failure mode you're trying to avoid.

    streaming=True enables token-by-token output, which is what powers
    the word-by-word rendering in the Streamlit UI.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing GROQ_API_KEY. Make sure it is set in your .env file."
        )
    return ChatGroq(model=LLM_MODEL_NAME, temperature=0, groq_api_key=api_key, streaming=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Step 1's prompt: turns a follow-up question into a standalone one using
# chat history. It does NOT answer the question -- it only reformulates it.
_CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might "
    "reference context in the chat history, formulate a standalone "
    "question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and "
    "otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", _CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Step 2's prompt: the actual answer-generation prompt, grounded in the
# retrieved context. Kept close to the original prototype's wording, with
# a `chat_history` placeholder added so the LLM can also see prior turns
# (e.g. to avoid re-introducing itself or repeating earlier answers).
_QA_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a document. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer based on the context, say you don't "
    "know -- do not make anything up. Provide specific details from the "
    "context to support your answer.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ---------------------------------------------------------------------------
# Chain assembly
# ---------------------------------------------------------------------------

def build_rag_chain(retriever, llm=None):
    """
    Assemble the citation-returning conversational RAG chain (without the
    memory wrapper yet -- see `build_conversational_rag_chain` for that).
    """
    if llm is None:
        llm = get_llm()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # create_retrieval_chain returns a dict with "input", "chat_history",
    # "context" (List[Document]) and "answer" (str) -- the "context" key
    # is exactly what the Streamlit UI needs for the citations expander.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# ---------------------------------------------------------------------------
# Conversational memory
# ---------------------------------------------------------------------------

# In-memory chat history store, keyed by session_id.
# NOTE: this resets whenever the Python process restarts. For a real
# production deployment you'd swap this for a persistent store (Redis, a
# DB table, etc.) -- the RunnableWithMessageHistory interface doesn't
# change either way, only what `get_session_history` returns.
_session_store: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (creating if necessary) the chat history for a session."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def build_conversational_rag_chain(retriever, llm=None) -> RunnableWithMessageHistory:
    """
    Build the full conversational, citation-returning RAG chain.

    This is the object the Streamlit app actually calls. Every invocation
    needs `config={"configurable": {"session_id": <id>}}` so
    RunnableWithMessageHistory knows which conversation's history to load
    before the call and append to after.
    """
    rag_chain = build_rag_chain(retriever, llm=llm)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def clear_session_history(session_id: str) -> None:
    """Reset a session's conversational memory (used by a 'New chat' button)."""
    _session_store.pop(session_id, None)


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def stream_answer(
    conversational_rag_chain: RunnableWithMessageHistory,
    user_input: str,
    session_id: str,
    sources_out: List[Document],
) -> Generator[str, None, None]:
    """
    Stream the answer token-by-token (e.g. for `st.write_stream(...)`).

    A single `.stream()` pass yields BOTH the retrieved context and the
    streaming answer tokens, so this does both jobs in one LLM call:
      - yields each answer token as it's generated, for the UI to render
        word-by-word
      - appends the retrieved Document objects to `sources_out` (a list
        passed in by reference) so the caller can read them back *after*
        the generator is exhausted, e.g. to populate a "Sources" expander

    Passing a mutable list in rather than returning a tuple is a
    deliberate trade-off: generators can only `yield`, so this is the
    simplest way to surface a second piece of data (sources) alongside
    the streamed tokens without buffering the whole answer first.
    """
    for chunk in conversational_rag_chain.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        if "context" in chunk:
            sources_out.extend(chunk["context"])
        if "answer" in chunk:
            yield chunk["answer"]
