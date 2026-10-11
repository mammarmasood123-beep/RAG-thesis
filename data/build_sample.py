"""Select the study sample (500) and the pilot subset (first 150), seeded.

* Eligibility: exactly two distinct supporting titles, both present in the shipped context.
* Stratified by question type, proportional (largest remainder), for both the 500 and the
  pilot, so the pilot is a type-proportional prefix of the full sample and pilot items carry
  into the full run.
* Gold order: canonical = order of first appearance in supporting_facts. Flips (reversed
  order) are counterbalanced within each (type, pilot/rest) group.

Usage:
  python -m data.build_sample --hotpot hotpot_dev_distractor_v1.json --out artifacts/ \
      --n_total 500 --n_pilot 150 --seed 20261008
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.hotpot_io import load_hotpot
from harness.schedule import assign_flips, derive_seed


def _proportional(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    quotas = {k: n * c / total for k, c in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def eligible(it) -> bool:
    return len(it.sf_titles) == 2 and set(it.sf_titles) <= set(it.context_titles)


def select_sample(items, n_total: int, n_pilot: int, seed: int) -> list[dict]:
    pool = [it for it in items if eligible(it)]
    by_type: dict[str, list] = defaultdict(list)
    for it in sorted(pool, key=lambda it: it.qid):
        by_type[it.q_type].append(it)
    for t, lst in by_type.items():
        random.Random(derive_seed(seed, "sample", t)).shuffle(lst)

    total_alloc = _proportional({t: len(v) for t, v in by_type.items()}, n_total)
    pilot_alloc = _proportional(total_alloc, n_pilot)
    chosen, groups = [], defaultdict(list)
    for t in sorted(by_type):
        for i, it in enumerate(by_type[t][: total_alloc[t]]):
            is_pilot = i < pilot_alloc[t]
            chosen.append((it, is_pilot))
            groups[f"{t}|{'pilot' if is_pilot else 'rest'}"].append(it.qid)
    flips = assign_flips(groups, seed)

    # Pilot first, then rest; within each, a seeded interleave of types.
    order = sorted(chosen, key=lambda x: (not x[1], derive_seed(seed, "order", x[0].qid)))
    rows = []
    for it, is_pilot in order:
        canon = list(it.sf_titles)
        rows.append({
            "qid": it.qid,
            "question": it.question,
            "answer": it.answer,
            "q_type": it.q_type,
            "level": it.level,
            "pilot": is_pilot,
            "gold_titles_canonical": canon,
            "flipped": flips[it.qid],
            "gold_order": canon[::-1] if flips[it.qid] else canon,
            "gold_text": {t: it.paragraph(t) for t in canon},
            "shipped_distractors": [t for t in it.context_titles if t not in set(canon)],
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_total", type=int, default=500)
    ap.add_argument("--n_pilot", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args(argv)

    items = load_hotpot(args.hotpot)
    n_elig = sum(eligible(it) for it in items)
    rows = select_sample(items, args.n_total, args.n_pilot, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"split: {len(items)} questions; eligible (2 gold in context): {n_elig}; "
          f"excluded: {len(items) - n_elig}")
    print(f"split types: {dict(Counter(it.q_type for it in items))}")
    for label, sub in [("sample", rows), ("pilot", [r for r in rows if r["pilot"]])]:
        print(f"{label:<7} n={len(sub):>3}  types={dict(Counter(r['q_type'] for r in sub))}  "
              f"flipped={sum(r['flipped'] for r in sub)}")
    print(f"wrote {out/'sample.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
