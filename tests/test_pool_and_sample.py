"""Pool construction and sample selection."""
from collections import Counter

import numpy as np

from data.bm25 import BM25Index
from data.build_pool import build_corpus, top_k_excluding
from data.build_sample import select_sample
from data.hotpot_io import HotpotItem


def _item(qid, q_type="bridge", ctx=None, sf=("A", "B")):
    ctx = ctx or [("A", "a text"), ("B", "b text")] + [(f"{qid}-d{i}", "noise") for i in range(8)]
    return HotpotItem(qid, f"question {qid}", "ans", q_type, "hard", ctx, list(sf))


def test_corpus_dedups_titles_and_counts_slots():
    items = [_item("q1"), _item("q2")]          # A and B shared across both
    titles, texts, n_slots, conflicts = build_corpus(items)
    assert n_slots == 20
    assert len(titles) == 18 and titles == sorted(titles)
    assert conflicts == {}


def test_corpus_reports_title_text_conflicts():
    items = [_item("q1"), _item("q2", ctx=[("A", "DIFFERENT"), ("B", "b text")])]
    assert build_corpus(items)[3] == {"A": 2}


def test_fast_bm25_equals_rank_bm25():
    docs = ["the cat sat on the mat", "dogs and cats", "a mat for the dog", "unrelated words here",
            "cat cat cat", "the the the"]
    idx = BM25Index(docs)
    for q in ["cat on mat", "the dog", "zebra", "cat cat"]:
        assert np.allclose(idx.scores(q), idx.reference_scores(q), atol=1e-12)


def test_top_k_excludes_gold_and_zero_scores_and_ranks_from_one():
    titles = ["A", "B", "C", "D", "E"]
    scores = np.array([5.0, 4.0, 3.0, 0.0, 3.0])
    out = top_k_excluding(scores, titles, exclude={"A"}, k=10)
    assert [c["title"] for c in out] == ["B", "C", "E"]     # D has score 0; C/E tie -> title order
    assert [c["rank"] for c in out] == [1, 2, 3]


def test_select_sample_stratified_pilot_prefix_and_counterbalanced():
    items = [_item(f"b{i}", "bridge") for i in range(400)] + [_item(f"c{i}", "comparison") for i in range(100)]
    rows = select_sample(items, n_total=250, n_pilot=100, seed=7)
    assert len(rows) == 250
    types = Counter(r["q_type"] for r in rows)
    assert types == {"bridge": 200, "comparison": 50}        # proportional (80/20)
    pilot = [r for r in rows if r["pilot"]]
    assert len(pilot) == 100 and Counter(r["q_type"] for r in pilot) == {"bridge": 80, "comparison": 20}
    for t in ("bridge", "comparison"):
        for is_pilot in (True, False):
            grp = [r for r in rows if r["q_type"] == t and r["pilot"] == is_pilot]
            n_flip = sum(r["flipped"] for r in grp)
            assert abs(n_flip - (len(grp) - n_flip)) <= 1
    for r in rows:
        assert sorted(r["gold_order"]) == sorted(r["gold_titles_canonical"])
        assert r["gold_order"] == (r["gold_titles_canonical"][::-1] if r["flipped"] else r["gold_titles_canonical"])
    assert rows == select_sample(items, n_total=250, n_pilot=100, seed=7)


def test_select_sample_excludes_items_without_two_gold_in_context():
    bad = _item("bad", sf=("A",))
    missing = _item("missing", sf=("A", "Z"))
    rows = select_sample([bad, missing] + [_item(f"b{i}") for i in range(20)], n_total=10, n_pilot=5, seed=1)
    assert {"bad", "missing"}.isdisjoint(r["qid"] for r in rows)
