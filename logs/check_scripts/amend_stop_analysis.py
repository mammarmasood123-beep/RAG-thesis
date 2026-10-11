"""Read-only diagnosis of the III.5 build STOP. Changes nothing; writes no artifacts.
For every question: survivors / removed / max bm25_rank under
  (S) the specified matcher: plain substring on normalised text, cap 300   [as built]
  (W) whole-token matcher (eval.metrics.answer_in_text) on text, cap 300  [comparison only]
and, for questions failing (S), survivors under (S) with no cap.
Usage: python -m logs.check_scripts.amend_stop_analysis hotpot_dev_distractor_v1.json [artifacts/sample.jsonl]"""
import json
import sys
from collections import Counter

import numpy as np

from data.bm25 import BM25Index
from data.build_pool import MAX_BM25_RANK, POOL_SIZE, _order, build_corpus
from data.hotpot_io import load_hotpot
from eval.hotpot_official import normalize_answer

items = load_hotpot(sys.argv[1])
titles, texts, _, _ = build_corpus(items)
index = BM25Index([f"{t} {x}" for t, x in zip(titles, texts)])
norm = [normalize_answer(x) for x in texts]
padded = [f" {x} " for x in norm]


def scan(sc, gold, ans_norm, mode, cap):
    surv = removed = rank = 0
    for i in _order(sc):
        if sc[i] <= 0:
            break
        if titles[i] in gold:
            continue
        rank += 1
        if cap and rank > cap:
            rank -= 1
            break
        if mode == "S":
            hit = ans_norm in norm[i]
        else:  # W: whole-token; empty answer -> not applicable (no filter)
            hit = bool(ans_norm) and f" {ans_norm} " in padded[i]
        if hit:
            removed += 1
            continue
        surv += 1
        if surv == POOL_SIZE:
            break
    return surv, removed, rank


res = {}
for it in items:
    a = normalize_answer(it.answer)
    if a in ("yes", "no"):
        continue
    sc = index.scores(it.question)
    gold = set(it.sf_titles)
    s = scan(sc, gold, a, "S", MAX_BM25_RANK)
    w = scan(sc, gold, a, "W", MAX_BM25_RANK)
    res[it.qid] = (it, a, s, w, sc, gold)

fail_s = [q for q, v in res.items() if v[2][0] < POOL_SIZE]
fail_w = [q for q, v in res.items() if v[3][0] < POOL_SIZE]
past_s = sum(1 for v in res.values() if v[2][2] > POOL_SIZE)
past_w = sum(1 for v in res.values() if v[3][2] > POOL_SIZE)
rem_s = np.array([v[2][1] for v in res.values()])
rem_w = np.array([v[3][1] for v in res.values()])
mr_s = np.array([v[2][2] for v in res.values()])
mr_w = np.array([v[3][2] for v in res.values()])
print(f"filtered questions (non-yes/no): {len(res)}")
print(f"{'matcher':<34} {'fail <50':>8} {'past r50':>9} {'removed mean':>13} {'removed max':>12} "
      f"{'max_bm25 p50/p90/p99/max':>26}")
for name, f, p, rem, mr in (("S plain substring (as specified)", fail_s, past_s, rem_s, mr_s),
                            ("W whole-token (comparison only)", fail_w, past_w, rem_w, mr_w)):
    q = np.percentile(mr, [50, 90, 99]).astype(int).tolist() + [int(mr.max())]
    print(f"{name:<34} {len(f):>8} {p:>9} {rem.mean():>13.2f} {rem.max():>12} {str(q):>26}")
print(f"questions where substring removes MORE than whole-token: "
      f"{int((rem_s > rem_w).sum())}  (extra removals total: {int((rem_s - rem_w).sum())})")

print("\nfailing questions under the specified matcher (S):")
print(f"  {'qid':<26} {'type':<11} {'answer':<16} {'norm':<10} {'S cap300':>9} {'S no cap (surv/scan)':>21} {'W cap300':>9}")
for q in sorted(fail_s):
    it, a, s, w, sc, gold = res[q]
    nc = scan(sc, gold, a, "S", None)
    print(f"  {q:<26} {it.q_type:<11} {it.answer!r:<16} {a!r:<10} {s[0]:>9} {f'{nc[0]}/{nc[2]}':>21} {w[0]:>9}")

if len(sys.argv) > 2:
    sample = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
    where = {r["qid"]: ("pilot" if r["pilot"] else "sample-rest") for r in sample}
    print("\nfailing questions in the study sample:", {q: where[q] for q in fail_s if q in where} or "none")

top = Counter(res[q][1] for q in res if res[q][2][1] > 0 and res[q][2][1] > res[q][3][1])
print("\nanswers whose substring removals exceed whole-token removals (top 15, by question count):")
print("  " + ", ".join(f"{a!r}:{c}" for a, c in top.most_common(15)))
