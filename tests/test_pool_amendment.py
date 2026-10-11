"""Design00 III.5 amendment (accepted 2026-10-08; matcher decided 2026-10-08 23:01):
answer-bearing candidates are excluded from the distractor pool.
Matcher: whole-token on TITLE + TEXT after normalize_answer; raw-lowercase whole-token
fallback when normalize_answer(answer) is empty; skipped for yes/no; gold never screened.

Unit tests run everywhere. The `artifact` tests run only when artifacts/ and the dataset
file exist (they read the real pool), and are skipped otherwise.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from data.build_pool import (MAX_BM25_RANK, answer_key, candidate_strings, contains_answer,
                             filter_candidates, main as build_pool_main)
from harness.schedule import PoolEntry, build_plan

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "artifacts" / "pool.jsonl"
PARAS = ROOT / "artifacts" / "paragraphs.jsonl"
SAMPLE = ROOT / "artifacts" / "sample.jsonl"
HOTPOT = ROOT / "hotpot_dev_distractor_v1.json"


def _filter(titles, texts, answer, k, exclude=frozenset(), scores=None):
    norm, raw = candidate_strings(titles, texts)
    scores = np.linspace(len(titles), 1, len(titles)) if scores is None else scores
    return filter_candidates(scores, titles, norm, raw, set(exclude), answer, k=k)


def _corpus(n=80, leak_every=3):
    titles = [f"T{i:03d}" for i in range(n)]
    texts = [(f"A genus of flowering plants in the family {i}." if i % leak_every == 0
              else f"An unrelated article number {i}.") for i in range(n)]
    return titles, texts


def _leaks(title, text, answer):
    mode, key = answer_key(answer)
    norm, raw = candidate_strings([title], [text])
    return contains_answer(mode, key, norm[0], raw[0])


# ---------------------------------------------------------------- (a) no answer in any candidate
def test_a_survivors_never_contain_answer():
    titles, texts = _corpus()
    out, stats = _filter(titles, texts, "flowering plants", k=20)
    assert len(out) == 20
    assert not any(_leaks(c["title"], texts[titles.index(c["title"])], "flowering plants") for c in out)
    assert stats["n_removed"] > 0 and stats["applied"] is True and stats["match_mode"] == "normalized"


def test_a_match_is_whole_token_not_substring():
    titles, texts = ["X", "Y", "Z"], ["He moved to Annapolis.", "Met ANNA there.", "Nothing here."]
    out, _ = _filter(titles, texts, "Anna", k=3)
    assert [c["title"] for c in out] == ["X", "Z"]          # Annapolis kept, ANNA removed


def test_a_title_is_matched_not_only_text():
    titles, texts = ["Gillian Anderson", "Other"], ["An actress.", "Unrelated."]
    out, _ = _filter(titles, texts, "Gillian Anderson", k=2)
    assert [c["title"] for c in out] == ["Other"]


@pytest.mark.parametrize("answer,leaky_title,leaky_text", [
    ("The The", "The The", "an English band"),        # normalize_answer -> "" ; raw "the the"
    ("!!!", "!!!", "an American dance-punk band"),     # normalize_answer -> "" ; raw "!!!"
])
def test_a_raw_lowercase_fallback_when_normalised_answer_is_empty(answer, leaky_title, leaky_text):
    mode, key = answer_key(answer)
    assert mode == "raw_fallback" and key == answer.lower()
    titles = [leaky_title, "Band", "Other"]
    texts = [leaky_text, "they played the band the next day", "nothing"]
    out, stats = _filter(titles, texts, answer, k=3)
    assert [c["title"] for c in out] == ["Band", "Other"] and stats["n_removed"] == 1


# ---------------------------------------------------------------- (b) gold never filtered
def test_b_gold_is_excluded_by_title_not_screened_and_still_scheduled():
    titles, texts = _corpus()
    gold = {"T000", "T003"}                   # both gold paragraphs CONTAIN the answer
    out, stats = _filter(titles, texts, "flowering plants", k=30, exclude=gold)
    assert len(out) == 30 and gold.isdisjoint(c["title"] for c in out)
    last = titles.index(out[-1]["title"])
    assert stats["n_removed"] == sum(1 for i, t in enumerate(titles)
                                     if t not in gold and i <= last and i % 3 == 0)
    pool = [PoolEntry(c["title"], c["pool_rank"], c["bm25_rank"]) for c in out]
    plan = build_plan("q", ["T000", "T003"], pool, n=0, n_max=4, t_max=6, master=1, rank_band=(1, 30))
    assert [it.title for rp in plan for it in rp.items if it.source == "gold"] == ["T000", "T003"]


# ---------------------------------------------------------------- (c) yes/no unfiltered
@pytest.mark.parametrize("ans", ["yes", "No", "YES."])
def test_c_yes_no_answers_are_not_filtered(ans):
    out, stats = _filter(["A", "B", "C"], ["yes yes", "no no", "neither"], ans, k=3)
    assert [c["title"] for c in out] == ["A", "B", "C"]
    assert stats["applied"] is False and stats["n_removed"] == 0 and stats["match_mode"] == "skip_yes_no"
    assert [c["pool_rank"] for c in out] == [c["bm25_rank"] for c in out] == [1, 2, 3]


# ---------------------------------------------------------------- (d) ranks logged + consistent
def test_d_pool_rank_and_bm25_rank_consistent():
    titles, texts = _corpus(leak_every=2)
    out, stats = _filter(titles, texts, "flowering plants", k=20)
    assert [c["pool_rank"] for c in out] == list(range(1, 21))
    b = [c["bm25_rank"] for c in out]
    assert b == sorted(set(b)) and all(c["bm25_rank"] >= c["pool_rank"] for c in out)
    for c in out:   # bm25_rank - pool_rank = answer-bearing candidates ranked above it
        i = titles.index(c["title"])
        assert c["bm25_rank"] - c["pool_rank"] == sum(1 for j in range(i) if j % 2 == 0)
    assert stats["max_bm25_rank"] == b[-1]


def test_d_bm25_rank_reaches_injected_items_in_plan():
    pool = [PoolEntry(f"P{i}", i, 2 * i) for i in range(1, 51)]
    plan = build_plan("q", ["G1", "G2"], pool, n=4, n_max=4, t_max=6, master=1)
    for rp in plan:
        for it in rp.items:
            if it.source == "distractor":
                assert it.bm25_rank == 2 * it.pool_rank
            else:
                assert it.bm25_rank is None and it.pool_rank is None


# ---------------------------------------------------------------- cap and STOP
def test_cap_stops_scan_at_rank_300():
    n = 400
    titles = [f"T{i:03d}" for i in range(n)]
    texts = ["the answer here" if i < 280 else "clean" for i in range(n)]
    out, stats = _filter(titles, texts, "answer", k=50)
    assert len(out) == MAX_BM25_RANK - 280 == 20       # only ranks 281..300 are scanned clean
    assert stats["max_bm25_rank"] == MAX_BM25_RANK


def test_build_pool_stops_when_a_question_cannot_reach_50(tmp_path):
    # every paragraph contains the answer "zebra" -> no survivors -> exit 2, nothing written
    ctx = [[f"Doc {i}", [f"The zebra number {i}."]] for i in range(60)]
    data = [{"_id": "q1", "question": "zebra number?", "answer": "zebra", "type": "bridge",
             "level": "hard", "supporting_facts": [["Doc 0", 0], ["Doc 1", 0]], "context": ctx}]
    f = tmp_path / "h.json"
    f.write_text(json.dumps(data))
    out = tmp_path / "out"
    assert build_pool_main(["--hotpot", str(f), "--out", str(out)]) == 2
    assert not (out / "pool.jsonl").exists()


# ---------------------------------------------------------------- artifact-level checks
needs_artifacts = pytest.mark.skipif(
    not (POOL.exists() and PARAS.exists() and SAMPLE.exists() and HOTPOT.exists()),
    reason="real artifacts or dataset not present")


@pytest.fixture(scope="module")
def real():
    from data.hotpot_io import load_hotpot
    items = {it.qid: it for it in load_hotpot(HOTPOT)}
    paras = {p["title"]: p["text"] for p in map(json.loads, open(PARAS, encoding="utf-8"))}
    pools = [json.loads(line) for line in open(POOL, encoding="utf-8")]
    sample = [json.loads(line) for line in open(SAMPLE, encoding="utf-8")]
    return items, paras, pools, sample


@needs_artifacts
def test_artifact_a_no_candidate_contains_answer(real):
    items, paras, pools, _ = real
    bad = []
    for p in pools:
        ans = items[p["qid"]].answer
        if answer_key(ans)[0] == "skip_yes_no":
            continue
        bad += [(p["qid"], c["title"]) for c in p["candidates"] if _leaks(c["title"], paras[c["title"]], ans)]
    assert bad == []


@needs_artifacts
def test_artifact_b_gold_untouched(real):
    items, _, pools, sample = real
    for r in sample:
        it = items[r["qid"]]
        assert r["gold_titles_canonical"] == it.sf_titles
        assert r["gold_text"] == {t: it.paragraph(t) for t in it.sf_titles}
    for p in pools:
        assert set(items[p["qid"]].sf_titles).isdisjoint(c["title"] for c in p["candidates"])


@needs_artifacts
def test_artifact_c_yes_no_unfiltered(real):
    items, _, pools, _ = real
    yn = [p for p in pools if answer_key(items[p["qid"]].answer)[0] == "skip_yes_no"]
    assert yn
    for p in yn:
        assert p["answer_filter"]["applied"] is False and p["answer_filter"]["n_removed"] == 0
        assert [c["pool_rank"] for c in p["candidates"]] == [c["bm25_rank"] for c in p["candidates"]]


@needs_artifacts
def test_artifact_d_ranks_consistent_and_50_each(real):
    _, _, pools, _ = real
    for p in pools:
        c = p["candidates"]
        assert len(c) == 50
        assert [x["pool_rank"] for x in c] == list(range(1, 51))
        b = [x["bm25_rank"] for x in c]
        assert b == sorted(set(b)) and b[-1] <= MAX_BM25_RANK
        assert all(x["bm25_rank"] >= x["pool_rank"] for x in c)
        assert b[-1] == p["answer_filter"]["max_bm25_rank"]
        assert b[-1] - 50 == p["answer_filter"]["n_removed"]
