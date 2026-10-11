"""Read-only check (Task 5): which short-context questions (< 10 shipped paragraphs) are in the
stub run / pilot / sample, and does build_plan handle every one of them.
Usage: python -m logs.check_scripts.short_context hotpot_dev_distractor_v1.json runs/<run_id>"""
import json
import sys
from collections import Counter
from pathlib import Path

from data.hotpot_io import load_hotpot
from harness.schedule import PoolEntry, build_plan

items = load_hotpot(sys.argv[1])
run = Path(sys.argv[2])
short = {it.qid: len(it.context) for it in items if len(it.context) < 10}
print(f"questions with < 10 shipped paragraphs: {len(short)}")
print(f"  distribution of paragraph counts: {dict(sorted(Counter(short.values()).items()))}")
print(f"  gold (2 distinct sf titles) present in context for all of them: "
      f"{all(set(it.sf_titles) <= set(it.context_titles) and len(it.sf_titles) == 2 for it in items if it.qid in short)}")

sample = [json.loads(l) for l in open("artifacts/sample.jsonl", encoding="utf-8")]
run_qids = {json.loads(l)["qid"] for l in open(run / "records.jsonl", encoding="utf-8")}
pilot = {r["qid"] for r in sample if r["pilot"]}
allq = {r["qid"] for r in sample}
print(f"short-context questions in the 20-question stub run : {sorted(short.keys() & run_qids)}")
print(f"short-context questions in the 150 pilot            : {sorted(short.keys() & pilot)}")
print(f"short-context questions in the 500 sample           : {sorted(short.keys() & allq)}")

# Schedule check on all short-context questions, all design N levels, directly (no generator).
pools = {p["qid"]: [PoolEntry(c["title"], c["pool_rank"], c["bm25_rank"]) for c in p["candidates"]]
         for p in map(json.loads, open("artifacts/pool.jsonl", encoding="utf-8")) if p["qid"] in short}
by_qid = {it.qid: it for it in items}
ok, errors = 0, []
for qid in sorted(short):
    gold = by_qid[qid].sf_titles
    for n in (0, 1, 2, 4):
        try:
            plan = build_plan(qid, gold, pools[qid], n=n, n_max=4, t_max=6, master=20261008)
            golds = [it.title for rp in plan for it in rp.items if it.source == "gold"]
            dists = [it.title for rp in plan for it in rp.items if it.source == "distractor"]
            assert sorted(golds) == sorted(gold), "gold mismatch"
            assert len(dists) == 6 * n and len(set(dists)) == len(dists), "distractor count/dup"
            assert not set(dists) & set(gold), "gold injected as distractor"
            ok += 1
        except Exception as e:
            errors.append((qid, n, repr(e)))
print(f"build_plan on {len(short)} short-context questions x N in {{0,1,2,4}}: "
      f"{ok} plans OK, {len(errors)} errors")
for e in errors:
    print("  ERROR", e)
print(f"pool sizes for these questions: min={min(len(pools[q]) for q in short)} "
      f"max={max(len(pools[q]) for q in short)}")
