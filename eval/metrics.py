"""Answer parsing (SIM-RAG's regex) and scoring (HotpotQA's official EM/F1)."""
from __future__ import annotations

import re

from eval.hotpot_official import exact_match_score, f1_score, normalize_answer

# utils/text_processing.py::extract_final_answer_and_rationale @ SIM-RAG 3d5f13a (no DOTALL)
_ANSWER_RATIONALE = re.compile(r"Answer:\s*(.*?)\s*Rationale:\s*(.*)")


def parse_reasoner_output(text: str) -> tuple[str, str, bool]:
    """Returns (answer, rationale, parse_ok).

    Mirrors SIM-RAG: blank lines are collapsed first (extract_assistant_output), then the
    Answer/Rationale regex is applied. Deviation: on failure SIM-RAG returns the literal
    "Answer not found", which can earn F1 > 0 against some golds; we return "".
    """
    cleaned = re.sub(r"\n\s*\n", "\n", text.strip())
    m = _ANSWER_RATIONALE.search(cleaned)
    if not m:
        return "", "", False
    return m.group(1).strip(), m.group(2).strip(), True


def em(pred: str, gold: str) -> int:
    return int(exact_match_score(pred, gold))


def f1(pred: str, gold: str) -> float:
    return float(f1_score(pred, gold)[0])


def is_unsure(answer: str) -> bool:
    return normalize_answer(answer) == "unsure"


def answer_in_text(gold_answer: str, text: str) -> bool | None:
    """Answer-leakage flag (Task 6b). Does the normalised gold answer occur in the normalised
    text as a whole-token sequence? Both sides use HotpotQA's normalize_answer.
    Returns None (not applicable) when the answer is yes/no or normalises to empty."""
    ans = normalize_answer(gold_answer)
    if ans in ("", "yes", "no"):
        return None
    return f" {ans} " in f" {normalize_answer(text)} "
