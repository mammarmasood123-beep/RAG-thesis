"""Reasoner prompt, copied verbatim from SIM-RAG.

Source: https://github.com/ucscirkm/SIM-RAG  utils/prompt_template.py
Commit: 3d5f13a8b53764549d1d5b013bbd25c5e17b9c8e (2025-10-08)
  force_answer_prompt = (<instruction>) + formatting2
The repository has no LICENSE file; reproduced for research replication with attribution
to Yang et al., SIGIR 2025. The missing space in "reasoning.Respond" is in the original.

Experiment A uses FORCE_ANSWER_NOABSTAIN: the one sentence offering the 'unsure' option
is deleted (Design00 VII.5: abstention disabled in fixed-round runs). The preceding
"do not try to make it up" sentence is kept verbatim; it offers no abstention token.
"""
import hashlib

SOURCE_REPO = "https://github.com/ucscirkm/SIM-RAG"
SOURCE_COMMIT = "3d5f13a8b53764549d1d5b013bbd25c5e17b9c8e"

FORMATTING2 = '''Respond only with the following format, nothing else:
Answer: [Provide the answer here]
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''

FORCE_ANSWER_VERBATIM = (
    "You are a knowledgeable question-answering assistant. Based on the context provided (if any), answer the following "
    "multihop question. Provide a brief multihop explanation. If you do not know part of what is referenced in the question, "
    "do not try to make it up. If unsure, respond with 'unsure.' Provide your answer along with a brief explanation of your reasoning."
) + FORMATTING2

ABSTAIN_SENTENCE = "If unsure, respond with 'unsure.' "
FORCE_ANSWER_NOABSTAIN = FORCE_ANSWER_VERBATIM.replace(ABSTAIN_SENTENCE, "", 1)


def _entry(text: str) -> dict:
    return {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}


PROMPTS = {
    "simrag_force_answer_verbatim": _entry(FORCE_ANSWER_VERBATIM),
    "simrag_force_answer_noabstain": _entry(FORCE_ANSWER_NOABSTAIN),
}
