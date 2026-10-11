"""The Reasoner prompt must be SIM-RAG's verbatim, minus exactly one sentence."""
import hashlib

from prompts.simrag import (
    ABSTAIN_SENTENCE,
    FORCE_ANSWER_NOABSTAIN,
    FORCE_ANSWER_VERBATIM,
    PROMPTS,
    SOURCE_COMMIT,
)

# sha256 of utils/prompt_template.py::force_answer_prompt at ucscirkm/SIM-RAG@3d5f13a
VERBATIM_SHA256 = "e18030fac24f030babc2aa5d266a997118ac2d197279c57ad8e9794a1009d2d3"


def test_verbatim_matches_simrag_source():
    assert hashlib.sha256(FORCE_ANSWER_VERBATIM.encode()).hexdigest() == VERBATIM_SHA256
    assert SOURCE_COMMIT.startswith("3d5f13a")


def test_noabstain_removes_exactly_one_sentence():
    assert FORCE_ANSWER_VERBATIM.count(ABSTAIN_SENTENCE) == 1
    assert FORCE_ANSWER_NOABSTAIN == FORCE_ANSWER_VERBATIM.replace(ABSTAIN_SENTENCE, "", 1)
    assert "unsure" not in FORCE_ANSWER_NOABSTAIN.lower()


def test_registry_has_ids_and_hashes():
    assert PROMPTS["simrag_force_answer_noabstain"]["text"] == FORCE_ANSWER_NOABSTAIN
    for entry in PROMPTS.values():
        assert entry["sha256"] == hashlib.sha256(entry["text"].encode()).hexdigest()
