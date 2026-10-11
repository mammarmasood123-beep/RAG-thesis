"""Read-only check (Task 4): field-level diff of two runs' records.jsonl, line by line.
Ignores ONLY the top-level `run_id` field. Usage:
  python -m logs.check_scripts.diff_runs runs/<run_A> runs/<run_B>"""
import json
import sys
from collections import Counter
from pathlib import Path

IGNORE = {"run_id"}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if path == "" and k in IGNORE:
                continue
            walk(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else k, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[]", out)
    elif a != b:
        out[path] += 1


A, B = (Path(p) for p in sys.argv[1:3])
la = open(A / "records.jsonl", encoding="utf-8").read().splitlines()
lb = open(B / "records.jsonl", encoding="utf-8").read().splitlines()
print(f"A = {A.name}: {len(la)} lines")
print(f"B = {B.name}: {len(lb)} lines")
diffs = Counter()
for x, y in zip(la, lb):
    walk(json.loads(x), json.loads(y), "", diffs)
ra = {json.loads(x)['run_id'] for x in la}
rb = {json.loads(x)['run_id'] for x in lb}
print(f"ignored field run_id: A={sorted(ra)} B={sorted(rb)}")
print(f"line-count equal: {len(la) == len(lb)}")
if diffs:
    print("fields that differ (count of records):")
    for k, v in sorted(diffs.items()):
        print(f"  {k}: {v}")
else:
    print("fields that differ besides run_id: NONE")
pa = open(A / "prompts.jsonl", encoding="utf-8").read()
pb = open(B / "prompts.jsonl", encoding="utf-8").read()
print(f"prompts.jsonl byte-identical: {pa == pb}")
