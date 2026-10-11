"""Read-only check (Task 3): print full assembled Reasoner inputs for chosen cells.
Usage: python -m logs.check_scripts.show_contexts runs/<run_id> [qid:N:round ...]   (run from expA/)"""
import json
import sys
from pathlib import Path

from harness.context import messages
from prompts.simrag import PROMPTS

run = Path(sys.argv[1])
recs = [json.loads(l) for l in open(run / "records.jsonl", encoding="utf-8")]
prompts = {(p["qid"], p["condition"]["N"], p["round"]): p["user_content"]
           for p in map(json.loads, open(run / "prompts.jsonl", encoding="utf-8"))}
manifest = json.loads((run / "manifest.json").read_text())
system = PROMPTS[manifest["config"]["prompt_id"]]["text"]

# default: first bridge N=4 r3, comparison N=4 r6, bridge N=0 r2; or pass specs "qid:N:round"
specs = sys.argv[2:]
if specs:
    wanted = [(f"q{i+1}", s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for i, s in enumerate(specs)]
else:
    wanted = [("a", "bridge", 4, 3), ("b", "comparison", 4, 6), ("c", "bridge", 0, 2)]
for label, key, n, rnd in wanted:
    rec = next(r for r in recs if (r["qid"] == key or r["q_type"] == key)
               and r["condition"]["N"] == n and r["round"] == rnd)
    qtype = rec["q_type"]
    user = prompts[(rec["qid"], n, rnd)]
    msgs = messages(system, user)
    print("#" * 100)
    print(f"({label}) {qtype}, N={n}, round {rnd}   qid={rec['qid']}   gold_answer={rec['gold_answer']!r}")
    print("#" * 100)
    print(">>> messages passed to generator.generate() (vLLM applies the Llama-3 chat template to these):")
    for m in msgs:
        print(f"----- role: {m['role']} -----")
        print(m["content"], end="" if m["content"].endswith("\n") else "\n")
        print(f"----- end {m['role']} -----")
    print(">>> record fields:")
    print(f"pool.type      = {rec['pool']['type']}")
    print(f"gold_order     = {rec['gold_order']}  flipped={rec['flipped']}  gold_this_round={rec['gold_this_round']!r}")
    print("injected[]     =")
    for it in rec["injected"]:
        print(f"    pos={it['position']}  source={it['source']:<10}  pool_rank={it['pool_rank']!s:<4}  "
              f"bm25_rank={it.get('bm25_rank')!s:<4}  contains_answer={it.get('contains_answer')!s:<5}  "
              f"n_tokens={it['n_tokens']:<4}  pid={it['pid']!r}")
    c = rec["context"]
    print(f"n_gold_cum={c['n_gold_cum']}  n_noise_cum={c['n_noise_cum']}  r_round={c['r_round']}  r_cum={c['r_cum']}  "
          f"prompt_tokens={c['prompt_tokens']} (stub: whitespace count, not Llama-3 tokens)")
    print(f"user_content_sha256={c['user_content_sha256']}")
    print()
