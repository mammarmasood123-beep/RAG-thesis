"""Seeded, pure scheduling for Experiment A (Design00 Part III).

Nothing here touches the model. Every random choice is derived from
    derive_seed(master, purpose, qid, N, round)
so a (question, N, round) cell is byte-identical across runs and across conditions.

Distractor draws are NESTED across N levels: for each round, the harness draws an ordered
list of n_max distractors, and condition N takes its first N. So the N=1 distractor in a
round is also present in the N=2 and N=4 conditions (common random numbers). Draws are
without replacement across all rounds of a question, uniform over the rank band.
Consequence: draws depend on n_max, so n_max must be fixed per study and logged.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PoolEntry:
    title: str
    rank: int                     # pool_rank: rank among survivors; the draw band uses this
    bm25_rank: int | None = None  # original BM25 rank before III.5 answer exclusion (logged only)


@dataclass(frozen=True)
class Item:
    title: str
    source: str               # "gold" | "distractor"
    pool_rank: int | None     # None for gold
    position: int             # 0-based slot within the round, after shuffling
    bm25_rank: int | None = None  # None for gold


@dataclass(frozen=True)
class RoundPlan:
    round: int                # 1-based
    gold_title: str | None
    items: tuple[Item, ...]   # sorted by position


class PoolTooSmall(ValueError):
    pass


def derive_seed(master: int, *parts) -> int:
    payload = "\x1f".join([str(master), *(f"{type(p).__name__}:{p}" for p in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def assign_flips(groups: dict[str, list[str]], seed: int) -> dict[str, bool]:
    """Within each group, exactly floor(n/2) or ceil(n/2) items get flipped=True."""
    out: dict[str, bool] = {}
    for key in sorted(groups):
        qids = sorted(groups[key])
        n = len(qids)
        flags = [False] * (n // 2) + [True] * (n - n // 2)
        random.Random(derive_seed(seed, "flip", key)).shuffle(flags)
        out.update(zip(qids, flags))
    return out


def gold_schedule(gold_order: list[str], t_max: int) -> list[str | None]:
    if t_max < len(gold_order):
        raise ValueError(f"t_max={t_max} < G={len(gold_order)}: gold would be dropped")
    return list(gold_order) + [None] * (t_max - len(gold_order))


def draw_distractors(pool: list[PoolEntry], n: int, n_max: int, t_max: int, seed: int,
                     rank_band: tuple[int, int] = (1, 50)) -> list[list[PoolEntry]]:
    if not 0 <= n <= n_max:
        raise ValueError(f"n={n} outside [0, n_max={n_max}]")
    lo, hi = rank_band
    eligible = sorted((p for p in pool if lo <= p.rank <= hi), key=lambda p: (p.rank, p.title))
    need = n_max * t_max
    if len(eligible) < need:
        raise PoolTooSmall(f"{len(eligible)} candidates in band {rank_band}, need {need}")
    chosen = random.Random(seed).sample(eligible, need)
    per_round = [chosen[r * n_max:(r + 1) * n_max] for r in range(t_max)]
    return [rnd[:n] for rnd in per_round]


def shuffle_round(items: list, seed: int) -> list:
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def build_plan(qid: str, gold_order: list[str], pool: list[PoolEntry], n: int, n_max: int,
               t_max: int, master: int, rank_band: tuple[int, int] = (1, 50)) -> list[RoundPlan]:
    gold_set = set(gold_order)
    pool = [p for p in pool if p.title not in gold_set]          # defensive: never inject gold as noise
    golds = gold_schedule(gold_order, t_max)
    draws = draw_distractors(pool, n, n_max, t_max, derive_seed(master, "draw", qid), rank_band)
    plan = []
    for r in range(1, t_max + 1):
        raw = ([("gold", golds[r - 1], None)] if golds[r - 1] else []) + \
              [("distractor", p.title, p.rank) for p in draws[r - 1]]
        ordered = shuffle_round(raw, derive_seed(master, "position", qid, n, r))
        bm25 = {p.title: p.bm25_rank for p in draws[r - 1]}
        items = tuple(Item(title=t, source=s, pool_rank=rk, position=i, bm25_rank=bm25.get(t))
                      for i, (s, t, rk) in enumerate(ordered))
        plan.append(RoundPlan(round=r, gold_title=golds[r - 1], items=items))
    return plan
