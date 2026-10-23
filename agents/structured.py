"""
agents/structured.py
----------------------
One safe helper for "ask the LLM to fill a fixed form" (a Pydantic model).
Used by the Critic (EvidenceVerdict) and the Planner (SubQuestionPlan).

A real bug we hit
-----------------
with_structured_output() in its default mode ("function_calling") FORCES
the model to call a tool. Sometimes the model just wrote text instead, and
Groq answered: "Tool choice is required, but model did not call a tool".
It was random: the same code worked on the next run.

What this helper does
---------------------
1. method="json_schema": ask for a JSON answer in our format (documented
   by Groq for openai/gpt-oss-20b) instead of forcing a tool call.
2. A NON-streaming LLM: Groq's docs say streaming and Structured Outputs
   do not work together.
3. A retry: Groq says this mode "may occasionally error", so try again
   before giving up.

If this mode ever misbehaves on your account, change the one line marked
(*) to `llm.with_structured_output(schema)` -- the retry keeps working.
"""

from typing import Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

import config
from core.rag_pipeline import get_llm


def invoke_structured(
    prompt: ChatPromptTemplate,
    schema: Type[BaseModel],
    inputs: dict,
    attempts: int = config.STRUCTURED_OUTPUT_ATTEMPTS,
    llm: Optional[BaseChatModel] = None,
):
    """
    Run `prompt` and return the answer as an instance of `schema`.
    Tries up to `attempts` times; if every try fails, the LAST error is raised.
    """
    if llm is None:
        llm = get_llm(streaming=False)

    chain = prompt | llm.with_structured_output(schema, method="json_schema")  # (*)

    last_error: Optional[Exception] = None
    for _ in range(attempts):
        try:
            return chain.invoke(inputs)
        except Exception as error:  # retry on any API / parsing hiccup
            last_error = error

    raise last_error