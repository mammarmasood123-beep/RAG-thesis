"""Tests for harness.schedule — written before the implementation.

Properties under test (Design00 Part III):
  * gold: one per round for rounds 1..G, force-inserted, never dropped
  * distractors: N per round, uniform over a fixed rank band, seeded, no duplicates
    within a question, never a gold title
  * nesting: for a fixed question and round, the N=1 draw is a prefix of N=2, which is a
    prefix of N=4 (common random numbers across N levels)
  * positions: randomised within each round, seeded, logged
  * counterbalancing: gold-order flips balanced within each group
  * determinism: same (master, qid, N, round) -> byte-identical plan
"""
from collections import Counter

import pytest

from harness.schedule import (
    PoolEntry,
    PoolTooSmall,
    assign_flips,
    build_plan,
    derive_seed,
    draw_distractors,
    gold_schedule,
    shuffle_round,
)

POOL = [PoolEntry(title=f"D{i:02d}", rank=i) for i in range(1, 51)]
GOLD = ["Gold A", "Gold B"]
MASTER = 20261008


# ---------------------------------------------------------------- seeds
def test_derive_seed_deterministic_and_part_sensitive():
    assert derive_seed(MASTER, "draw", "q1", 4) == derive_seed(MASTER, "draw", "q1", 4)
    assert derive_seed(MASTER, "draw", "q1", 4) != derive_seed(MASTER, "draw", "q2", 4)
    assert derive_seed(MASTER, "draw", "q1", 4) != derive_seed(MASTER + 1, "draw", "q1", 4)
    assert derive_seed(MASTER, "a", "b") != derive_seed(MASTER, "b", "a")
    assert derive_seed(MASTER, "ab", "c") != derive_seed(MASTER, "a", "bc")  # no concat collisions


# ---------------------------------------------------------------- gold
def test_gold_one_per_round_then_none():
    assert gold_schedule(GOLD, t_max=6) == ["Gold A", "Gold B", None, None, None, None]


def test_gold_order_respected():
    assert gold_schedule(list(reversed(GOLD)), t_max=3) == ["Gold B", "Gold A", None]


def test_gold_never_dropped_t_max_below_g_rejected():
    with pytest.raises(ValueError):
        gold_schedule(GOLD, t_max=1)


# ---------------------------------------------------------------- distractor draws
def _flat(draws):
    return [p for rnd in draws for p in rnd]


def test_draw_shape_and_no_duplicates():
    d = draw_distractors(POOL, n=4, n_max=4, t_max=6, seed=1)
    assert len(d) == 6 and all(len(r) == 4 for r in d)
    titles = [p.title for p in _flat(d)]
    assert len(titles) == len(set(titles)) == 24
    assert set(titles) <= {p.title for p in POOL}


def test_draw_n0_is_empty_every_round():
    assert draw_distractors(POOL, n=0, n_max=4, t_max=6, seed=1) == [[] for _ in range(6)]


def test_draw_deterministic_and_seed_sensitive():
    a = draw_distractors(POOL, n=4, n_max=4, t_max=6, seed=7)
    assert a == draw_distractors(POOL, n=4, n_max=4, t_max=6, seed=7)
    assert a != draw_distractors(POOL, n=4, n_max=4, t_max=6, seed=8)


def test_draw_independent_of_pool_input_order():
    shuffled = list(reversed(POOL))
    assert draw_distractors(shuffled, 4, 4, 6, seed=3) == draw_distractors(POOL, 4, 4, 6, seed=3)


def test_draw_nested_across_n():
    d1, d2, d4 = (draw_distractors(POOL, n=n, n_max=4, t_max=6, seed=11) for n in (1, 2, 4))
    for r in range(6):
        assert d1[r] == d2[r][:1] == d4[r][:1]
        assert d2[r] == d4[r][:2]


def test_pool_too_small_raises():
    with pytest.raises(PoolTooSmall):
        draw_distractors(POOL[:23], n=4, n_max=4, t_max=6, seed=1)
    draw_distractors(POOL[:24], n=4, n_max=4, t_max=6, seed=1)  # exactly enough is fine


def test_rank_band_enforced():
    d = draw_distractors(POOL, n=4, n_max=4, t_max=6, seed=5, rank_band=(1, 30))
    assert all(1 <= p.rank <= 30 for p in _flat(d))
    with pytest.raises(PoolTooSmall):
        draw_distractors(POOL, n=4, n_max=4, t_max=6, seed=5, rank_band=(1, 20))


def test_draws_uniform_over_rank_band():
    counts = Counter()
    for s in range(2000):
        counts.update(p.rank for p in _flat(draw_distractors(POOL, 4, 4, 6, seed=s)))
    expected = 2000 * 24 / 50  # 960 per rank
    assert set(counts) == set(range(1, 51))
    assert all(0.85 * expected < c < 1.15 * expected for c in counts.values())


# ---------------------------------------------------------------- positions
def test_shuffle_round_is_seeded_permutation():
    items = list("abcde")
    a = shuffle_round(items, seed=3)
    assert sorted(a) == items and a == shuffle_round(items, seed=3)
    assert items == list("abcde")  # input not mutated
    assert len({tuple(shuffle_round(items, seed=s)) for s in range(30)}) > 1


# ---------------------------------------------------------------- full plan
def _plan(qid="q1", n=4, gold=GOLD, t_max=6):
    return build_plan(qid=qid, gold_order=gold, pool=POOL, n=n, n_max=4, t_max=t_max, master=MASTER)


def test_plan_structure():
    plan = _plan()
    assert [rp.round for rp in plan] == [1, 2, 3, 4, 5, 6]
    for rp in plan:
        golds = [it for it in rp.items if it.source == "gold"]
        dists = [it for it in rp.items if it.source == "distractor"]
        assert len(golds) == (1 if rp.round <= 2 else 0)
        assert len(dists) == 4
        assert [it.position for it in rp.items] == list(range(len(rp.items)))
        assert all(it.pool_rank is None for it in golds)
        assert all(1 <= it.pool_rank <= 50 for it in dists)
    assert plan[0].gold_title == "Gold A" and plan[1].gold_title == "Gold B"


def test_plan_every_gold_exactly_once_and_cumulative_after_g():
    plan = _plan()
    gold_seen = [it.title for rp in plan for it in rp.items if it.source == "gold"]
    assert sorted(gold_seen) == sorted(GOLD)
    cumulative = set()
    for rp in plan:
        cumulative |= {it.title for it in rp.items if it.source == "gold"}
        if rp.round >= 2:
            assert cumulative == set(GOLD)


def test_plan_no_distractor_is_gold():
    pool_with_gold_title = [PoolEntry("Gold A", 1)] + POOL[1:]
    plan = build_plan("q1", GOLD, pool_with_gold_title, n=4, n_max=4, t_max=6, master=MASTER)
    assert all(it.title not in GOLD for rp in plan for it in rp.items if it.source == "distractor")


def test_plan_deterministic():
    assert _plan() == _plan()


def test_plan_n0_has_only_gold():
    plan = _plan(n=0)
    assert [len(rp.items) for rp in plan] == [1, 1, 0, 0, 0, 0]


def test_plan_gold_position_varies_across_questions():
    positions = set()
    for i in range(60):
        round1 = _plan(qid=f"q{i}")[0]
        positions.add(next(it.position for it in round1.items if it.source == "gold"))
    assert len(positions) >= 4  # 5 slots at N=4; gold must not sit in a fixed slot


def test_plan_distractors_nested_across_n_levels():
    p1, p4 = _plan(n=1), _plan(n=4)
    for a, b in zip(p1, p4):
        d1 = {it.title for it in a.items if it.source == "distractor"}
        d4 = {it.title for it in b.items if it.source == "distractor"}
        assert d1 <= d4


# ---------------------------------------------------------------- counterbalancing
def test_assign_flips_balanced_within_each_group():
    groups = {
        "bridge|pilot": [f"b{i}" for i in range(121)],
        "comparison|pilot": [f"c{i}" for i in range(29)],
        "bridge|rest": [f"x{i}" for i in range(280)],
    }
    flips = assign_flips(groups, seed=MASTER)
    assert set(flips) == {q for qs in groups.values() for q in qs}
    for qs in groups.values():
        n_true = sum(flips[q] for q in qs)
        assert abs(n_true - (len(qs) - n_true)) <= 1


def test_assign_flips_deterministic_and_not_trivial():
    groups = {"g": [f"q{i}" for i in range(40)]}
    a = assign_flips(groups, seed=1)
    assert a == assign_flips(groups, seed=1)
    assert a != assign_flips(groups, seed=2)
    assert [a[f"q{i}"] for i in range(40)] != [False] * 20 + [True] * 20


def test_assign_flips_independent_of_input_order():
    qs = [f"q{i}" for i in range(31)]
    assert assign_flips({"g": qs}, seed=4) == assign_flips({"g": list(reversed(qs))}, seed=4)
