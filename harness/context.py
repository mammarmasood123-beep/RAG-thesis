"""SIM-RAG Reasoner context, reproduced from ucscirkm/SIM-RAG@3d5f13a inference/inference.py.

  initial user content (Llama path):  "Question: {q}\\nContext:\\n"
  per-round append (update_task_with_retrieved_text):
      "Query: " + query + "\\nRetrieved Document: " + "\\n".join("Title: {t} Content: {x}") + "\\n"

Only the query and retrieved documents enter the context; previous answers and rationales
never do. Experiment A deviations (logged in the run manifest):
  * the query call is skipped, so the query slot is the empty string;
  * a round that injects zero paragraphs appends nothing (SIM-RAG would append an empty
    block), so N=0 prompts are byte-identical from round G onward.
"""
from __future__ import annotations

BASE_TEMPLATE = "Question: {question}\nContext:\n"
EMPTY_QUERY = ""


def round_block(paragraphs: list[tuple[str, str]], query: str = EMPTY_QUERY) -> str:
    docs = "\n".join(f"Title: {t} Content: {x}" for t, x in paragraphs)
    return "Query: " + query + "\nRetrieved Document: " + docs + "\n"


def build_context(question: str, rounds: list[list[tuple[str, str]]], upto: int) -> str:
    """User content after `upto` rounds. rounds[i] = [(title, text)] in position order."""
    out = BASE_TEMPLATE.format(question=question)
    for paragraphs in rounds[:upto]:
        if paragraphs:
            out += round_block(paragraphs)
    return out


def messages(system_prompt: str, user_content: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
