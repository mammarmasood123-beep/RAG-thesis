"""Tests for harness.context — the SIM-RAG Reasoner context builder.

SIM-RAG (inference/inference.py @ 3d5f13a) builds the Reasoner's user message as
    "Question: {q}\\nContext:\\n"
and after each round appends (update_task_with_retrieved_text)
    "Query: " + query + "\\nRetrieved Document: " + "\\n".join("Title: {t} Content: {x}") + "\\n"
Only the query and retrieved documents are appended. Previous answers and rationales never
enter the context. Experiment A skips the query call, so the query slot is "".

Deviation under test: a round that injects zero paragraphs appends NOTHING. SIM-RAG would
append an empty block, which would make N=0 rounds 3..6 differ by whitespace only.
"""
from harness.context import build_context, round_block
from harness.schedule import PoolEntry, build_plan

Q = "Which band did the drummer of X join?"
A = ("Gold A", "Alpha text.")
B = ("Gold B", "Beta text.")
D = ("Dist 1", "Delta text.")


def _simrag_update(task_content, search_query, retrieved_text):
    # verbatim from SIM-RAG inference/inference.py
    return task_content + "Query: " + search_query + "\nRetrieved Document: " + retrieved_text + "\n"


def test_round0_is_bare_question():
    assert build_context(Q, [[A], [B]], upto=0) == f"Question: {Q}\nContext:\n"


def test_round_block_matches_simrag_template_with_empty_query():
    docs = "\n".join(f"Title: {t} Content: {x}" for t, x in [A, D])
    assert round_block([A, D]) == _simrag_update("", "", docs)


def test_contexts_accumulate_as_prefixes():
    rounds = [[A, D], [D, B], [D]]
    ctx = [build_context(Q, rounds, upto=r) for r in range(4)]
    for r in range(3):
        assert ctx[r + 1].startswith(ctx[r]) and len(ctx[r + 1]) > len(ctx[r])


def test_empty_round_appends_nothing():
    rounds = [[A], [B], [], [], [], []]
    assert build_context(Q, rounds, upto=2) == build_context(Q, rounds, upto=6)


def test_n0_prompts_byte_identical_from_round_g_onward_via_plan():
    pool = [PoolEntry(f"P{i}", i) for i in range(1, 51)]
    texts = {"Gold A": "Alpha.", "Gold B": "Beta.", **{f"P{i}": f"text {i}." for i in range(1, 51)}}
    plan = build_plan("q1", ["Gold A", "Gold B"], pool, n=0, n_max=4, t_max=6, master=1)
    rounds = [[(it.title, texts[it.title]) for it in rp.items] for rp in plan]
    prompts = [build_context(Q, rounds, upto=r) for r in range(1, 7)]
    assert len(set(prompts[1:])) == 1          # rounds 2..6 identical
    assert prompts[0] != prompts[1]            # round 1 has only one gold


def test_n4_prompts_strictly_grow_every_round():
    pool = [PoolEntry(f"P{i}", i) for i in range(1, 51)]
    texts = {"Gold A": "Alpha.", "Gold B": "Beta.", **{f"P{i}": f"text {i}." for i in range(1, 51)}}
    plan = build_plan("q1", ["Gold A", "Gold B"], pool, n=4, n_max=4, t_max=6, master=1)
    rounds = [[(it.title, texts[it.title]) for it in rp.items] for rp in plan]
    lens = [len(build_context(Q, rounds, upto=r)) for r in range(7)]
    assert lens == sorted(set(lens))


def test_paragraph_order_follows_position():
    assert round_block([D, A]).index("Dist 1") < round_block([D, A]).index("Gold A")
