"""Read-only check: answer leakage in injected distractors, and drawn-rank hardness.
 (i)  from a run's records.jsonl (record field injected[].contains_answer = whole-token, TEXT only;
      unchanged since Task 6b), per N
 (ii) directly from the pool: the 500-question study sample (artifacts/sample.jsonl),
      N=4 draws over 6 rounds via build_plan with the study master seed. No generator.
      whole-token (title+text) = data.build_pool.contains_answer, the III.5 filter's own matcher
      plain substring (title+text) = same strings, no token boundaries (shows what the filter lets through)
Usage: python -m logs.check_scripts.leakage_report runs/<run_id>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from data.build_pool import answer_key, candidate_strings, contains_answer
from harness.schedule import PoolEntry, build_plan


def pct(a, b):
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else f"{a}/0 = n/a"


run = Path(sys.argv[1])
cfg = yaml.safe_load(open("configs/pilot.yaml"))
print("=" * 100)
print(f"(i) stub run {run.name}")
print("=" * 100)
recs = [json.loads(l) for l in open(run / "records.jsonl", encoding="utf-8")]
qids = {r["qid"] for r in recs}
yn = {r["qid"] for r in recs if answer_key(r["gold_answer"])[0] == "skip_yes_no"}
print(f"questions: {len(qids)}; yes/no-answer questions skipped: {len(yn)}")
for n in sorted({r["condition"]["N"] for r in recs if r["condition"]["arm"] == "distractor"}):
    flags = [it["contains_answer"] for r in recs if r["condition"]["N"] == n
             for it in r["injected"] if it["source"] == "distractor"]
    applicable = [f for f in flags if f is not None]
    print(f"N={n}: injected distractors={len(flags)}, applicable (non-yes/no)={len(applicable)}, "
          f"record field contains_answer (whole-token, text only): {pct(sum(applicable), len(applicable))}")

print()
print("=" * 100)
print("(ii) 500-question study sample (artifacts/sample.jsonl, seed 20261008), N=4, rounds 1-6, from pool")
print("=" * 100)
sample = [json.loads(l) for l in open(cfg["paths"]["sample"], encoding="utf-8")]
paragraphs = {p["title"]: p["text"] for p in map(json.loads, open(cfg["paths"]["paragraphs"], encoding="utf-8"))}
sq = {r["qid"] for r in sample}
pools = {p["qid"]: [PoolEntry(c["title"], c["pool_rank"], c["bm25_rank"]) for c in p["candidates"]]
         for p in map(json.loads, open(cfg["paths"]["pool"], encoding="utf-8")) if p["qid"] in sq}
tot = defaultdict(lambda: [0, 0, 0])   # key -> [applicable, whole-token title+text, plain substring title+text]
n_yn, misses, bm25_all, pool_all = 0, [], [], []
for r in sample:
    mode, key = answer_key(r["answer"])
    plan = build_plan(r["qid"], r["gold_order"], pools[r["qid"]], n=4, n_max=cfg["n_max"],
                      t_max=cfg["t_max"], master=cfg["master_seed"], rank_band=tuple(cfg["rank_band"]))
    dist = [it for rp in plan for it in rp.items if it.source == "distractor"]
    bm25_all += [it.bm25_rank for it in dist]
    pool_all += [it.pool_rank for it in dist]
    if mode == "skip_yes_no":
        n_yn += 1
        continue
    for rp in plan:
        for it in rp.items:
            if it.source != "distractor":
                continue
            norm, raw = candidate_strings([it.title], [paragraphs[it.title]])
            tok = contains_answer(mode, key, norm[0], raw[0])
            sub = bool(key) and key in (norm[0] if mode == "normalized" else raw[0])
            if sub and not tok:
                misses.append((r["qid"], r["answer"], it.title))
            band = "rank 1-10" if it.pool_rank <= 10 else "rank 11-25" if it.pool_rank <= 25 else "rank 26-50"
            for k in ("ALL", f"type={r['q_type']}", f"round={rp.round}", band):
                tot[k][0] += 1
                tot[k][1] += tok
                tot[k][2] += sub
print(f"questions: {len(sample)}; yes/no-answer questions skipped: {n_yn}; applicable: {len(sample) - n_yn}")
print(f"{'subset':<16} {'distractors':>11}  {'whole-token (title+text)':>26}  {'plain substring (title+text)':>30}")
order = ["ALL", "type=bridge", "type=comparison"] + [f"round={i}" for i in range(1, 7)] + ["rank 1-10", "rank 11-25", "rank 26-50"]
for k in order:
    a, h, s = tot[k]
    print(f"{k:<16} {a:>11}  {pct(h, a):>26}  {pct(s, a):>30}")
print(f"substring-only matches (not whole-token; let through by the filter): {len(misses)} "
      f"in {len({q for q, _, _ in misses})} questions")
for q, a, t in misses[:25]:
    print(f"  qid={q}  answer={a!r}  distractor={t!r}")

b, p = np.array(bm25_all), np.array(pool_all)
print()
print(f"HARDNESS of drawn distractors, all 500 questions incl. yes/no, N=4 ({len(b)} draws):")
print(f"  pool_rank : mean={p.mean():.2f}  median={np.median(p):.0f}  max={p.max()}")
print(f"  bm25_rank : mean={b.mean():.2f}  median={np.median(b):.0f}  p90={np.percentile(b, 90):.0f}  "
      f"p99={np.percentile(b, 99):.0f}  max={b.max()}")
print(f"  bm25_rank - pool_rank : mean={np.mean(b - p):.2f}  draws with bm25_rank > 50: {pct(int((b > 50).sum()), len(b))}")
