"""Experiment A runner: one pass per question to T_max, answer logged at every round.

  round 0          closed-book (condition arm="closed_book", N=null), one call per question
  rounds 1..T_max  for each N: inject plan[round], generate, log

One JSONL record per (question, round, condition) -> runs/<run_id>/records.jsonl.
Built-in checks, written to manifest.json; a failed hard check exits non-zero:
  HARD  N=0 user content byte-identical across rounds G..T_max for every question
  HARD  every record at round >= G holds all G gold paragraphs
  SOFT  N=0 outputs identical across rounds G..T_max (greedy vLLM is not guaranteed
        batch-invariant, so this is reported, not enforced)

Usage:
  python -m harness.run_expA --config configs/pilot.yaml [--backend stub|vllm] [--limit 20]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.metrics import answer_in_text, em, f1, is_unsure, parse_reasoner_output
from harness.context import build_context, messages
from harness.generator import make_generator
from harness.schedule import PoolEntry, build_plan, derive_seed
from prompts.simrag import PROMPTS

G = 2  # HotpotQA gold paragraphs per question


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "rank_bm25", "torch", "vllm", "transformers"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        env["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        env["nvidia_smi"] = None
    return env


def ratios(n_gold_cum, n_noise_cum, n_gold_round, n_noise_round):
    r_round = n_noise_round / (n_gold_round + n_noise_round) if (n_gold_round + n_noise_round) else None
    tot = n_gold_cum + n_noise_cum
    return r_round, (n_noise_cum / tot if tot else None)


class Runner:
    def __init__(self, cfg: dict, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.gen = make_generator(cfg)
        self.system = PROMPTS[cfg["prompt_id"]]["text"]
        self.records = open(run_dir / "records.jsonl", "w", encoding="utf-8")
        self.prompts = open(run_dir / "prompts.jsonl", "w", encoding="utf-8") if cfg["log_prompts"] else None
        self.n_records = 0
        self.n_not_fit = 0
        self.max_prompt_tokens = defaultdict(int)  # (N, round) -> max

    # ------------------------------------------------------------------ generation
    def _generate(self, jobs: list[dict]) -> None:
        """jobs: dicts with msgs, meta, prompt_tokens, fits. Fills job['result'] in place."""
        todo = [j for j in jobs if j["fits"]]
        bs = self.cfg["batch_size"]
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            for j, res in zip(chunk, self.gen.generate([j["msgs"] for j in chunk], [j["meta"] for j in chunk])):
                j["result"] = res

    def _job(self, row, user_content):
        msgs = messages(self.system, user_content)
        pt = self.gen.count_prompt_tokens(msgs)
        g = self.cfg["generator"]
        return {"row": row, "msgs": msgs, "user": user_content, "prompt_tokens": pt,
                "fits": pt + g["max_new_tokens"] <= g["max_model_len"],
                "meta": {"qid": row["qid"], "gold_answer": row["answer"],
                         "gold_titles": row["gold_titles_canonical"]}}

    # ------------------------------------------------------------------ logging
    def _write(self, job, condition, rnd, seeds, injected, ctx_counts, gold_this_round, pool_info):
        row, g, res = job["row"], self.cfg["generator"], job.get("result")
        n_gold_cum, n_noise_cum, n_gold_r, n_noise_r = ctx_counts
        r_round, r_cum = ratios(n_gold_cum, n_noise_cum, n_gold_r, n_noise_r)
        if res is not None:
            ans, rat, ok = parse_reasoner_output(res.text)
            module = {"module": "reasoner_answer", "calls": 1, "params": self.gen.params,
                      "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                      "param_tokens": self.gen.params * (res.input_tokens + res.output_tokens),
                      "wall_ms": res.wall_ms, "wall_source": res.wall_source}
            output = {"raw": res.text, "answer": ans, "rationale": rat, "parse_ok": ok,
                      "said_unsure": is_unsure(ans)}
            score = {"em": em(ans, row["answer"]), "f1": f1(ans, row["answer"])}
        else:
            self.n_not_fit += 1
            module = {"module": "reasoner_answer", "calls": 0, "params": self.gen.params,
                      "input_tokens": 0, "output_tokens": 0, "param_tokens": 0,
                      "wall_ms": None, "wall_source": None}
            output, score = None, None
        n_key = condition["N"]
        self.max_prompt_tokens[(n_key, rnd)] = max(self.max_prompt_tokens[(n_key, rnd)], job["prompt_tokens"])
        rec = {
            "run_id": self.cfg["_run_id"], "config_hash": self.cfg["_config_hash"],
            "experiment": "A", "condition": condition,
            "qid": row["qid"], "q_type": row["q_type"], "level": row["level"], "round": rnd,
            "seeds": seeds, "gold_order": row["gold_order"], "flipped": row["flipped"],
            "gold_this_round": gold_this_round, "injected": injected, "pool": pool_info,
            "context": {"n_gold_cum": n_gold_cum, "n_noise_cum": n_noise_cum,
                        "r_round": r_round, "r_cum": r_cum,
                        "prompt_tokens": job["prompt_tokens"], "max_model_len": g["max_model_len"],
                        "fits": job["fits"], "user_content_sha256": sha(job["user"])},
            "modules": [module], "output": output, "gold_answer": row["answer"], "score": score,
            "decoding": {k: g[k] for k in ("temperature", "top_p", "max_new_tokens")},
            "generator": {"backend": self.gen.backend, "model": self.gen.model},
            "prompt": {"id": self.cfg["prompt_id"], "sha256": PROMPTS[self.cfg["prompt_id"]]["sha256"]},
        }
        self.records.write(json.dumps(rec) + "\n")
        if self.prompts:
            self.prompts.write(json.dumps({"qid": row["qid"], "condition": condition, "round": rnd,
                                           "user_content": job["user"]}) + "\n")
        self.n_records += 1
        return rec

    # ------------------------------------------------------------------ experiment
    def closed_book(self, rows):
        jobs = [self._job(r, build_context(r["question"], [], 0)) for r in rows]
        self._generate(jobs)
        for j in jobs:
            self._write(j, {"arm": "closed_book", "N": None}, 0,
                        {"master": self.cfg["master_seed"]}, [], (0, 0, 0, 0), None, None)
        self.records.flush()

    def condition(self, rows, n, pools, paragraphs, checks):
        cfg = self.cfg
        t_max, master = cfg["t_max"], cfg["master_seed"]
        band = tuple(cfg["rank_band"])
        pool_info = {"type": cfg["pool_type"], "rank_band": list(band), "reused": False}
        plans, rounds = {}, {}
        for r in rows:
            plan = build_plan(r["qid"], r["gold_order"], pools[r["qid"]], n, cfg["n_max"], t_max, master, band)
            plans[r["qid"]] = plan
            rounds[r["qid"]] = [[(it.title, r["gold_text"][it.title] if it.source == "gold" else paragraphs[it.title])
                                 for it in rp.items] for rp in plan]
        user_sha, answers = defaultdict(dict), defaultdict(dict)
        for rnd in range(1, t_max + 1):
            jobs = [self._job(r, build_context(r["question"], rounds[r["qid"]], rnd)) for r in rows]
            self._generate(jobs)
            for j in jobs:
                r = j["row"]
                rp = plans[r["qid"]][rnd - 1]
                injected = [{"pid": it.title, "source": it.source, "pool_rank": it.pool_rank,
                             "bm25_rank": it.bm25_rank,   # III.5 amendment: original BM25 rank
                             "n_tokens": self.gen.count_text_tokens(text), "position": it.position,
                             # Task 6b (added field): distractor text contains the normalised gold
                             # answer; null for gold items and for yes/no-answer questions.
                             "contains_answer": (answer_in_text(r["answer"], text)
                                                 if it.source == "distractor" else None)}
                            for it, (_, text) in zip(rp.items, rounds[r["qid"]][rnd - 1])]
                upto = plans[r["qid"]][:rnd]
                n_gold_cum = sum(it.source == "gold" for p in upto for it in p.items)
                n_noise_cum = sum(it.source == "distractor" for p in upto for it in p.items)
                n_gold_r = sum(it.source == "gold" for it in rp.items)
                if rnd >= G and n_gold_cum != G:
                    checks["gold_integrity_violations"].append([r["qid"], n, rnd])
                seeds = {"master": master,
                         "draw": derive_seed(master, "draw", r["qid"]),
                         "position": derive_seed(master, "position", r["qid"], n, rnd),
                         "flip_group": f"{r['q_type']}|{'pilot' if r['pilot'] else 'rest'}"}
                rec = self._write(j, {"arm": "distractor", "N": n}, rnd, seeds, injected,
                                  (n_gold_cum, n_noise_cum, n_gold_r, len(rp.items) - n_gold_r),
                                  rp.gold_title, pool_info)
                user_sha[r["qid"]][rnd] = rec["context"]["user_content_sha256"]
                answers[r["qid"]][rnd] = rec["output"]["raw"] if rec["output"] else None
            self.records.flush()
        if n == 0:
            for qid in user_sha:
                if len({user_sha[qid][t] for t in range(G, t_max + 1)}) != 1:
                    checks["n0_prompt_identity_violations"].append(qid)
                if len({answers[qid][t] for t in range(G, t_max + 1)}) != 1:
                    checks["n0_output_identity_mismatches"].append(qid)

    def close(self):
        self.records.close()
        if self.prompts:
            self.prompts.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", choices=["stub", "vllm"])
    ap.add_argument("--limit", type=int, help="first k questions only (smoke test)")
    ap.add_argument("--run_suffix", default="")
    args = ap.parse_args(argv)

    raw_cfg = Path(args.config).read_text()
    cfg = yaml.safe_load(raw_cfg)
    if args.backend:
        cfg["generator"]["backend"] = args.backend
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg["_run_id"] = f"{cfg['run_name']}-{cfg['generator']['backend']}-{stamp}{args.run_suffix}"
    cfg["_config_hash"] = sha(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True))
    run_dir = Path(cfg["paths"]["runs_dir"]) / cfg["_run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = load_jsonl(cfg["paths"]["sample"])
    if cfg["sample_subset"] == "pilot":
        rows = [r for r in rows if r["pilot"]]
    if args.limit:
        rows = rows[: args.limit]
    pools = {}
    for p in load_jsonl(cfg["paths"]["pool"]):
        if p["pool_type"] != cfg["pool_type"]:
            raise SystemExit(f"pool_type mismatch: {p['pool_type']} != {cfg['pool_type']}")
        pools[p["qid"]] = [PoolEntry(c["title"], c["pool_rank"], c["bm25_rank"]) for c in p["candidates"]]
    paragraphs = {p["title"]: p["text"] for p in load_jsonl(cfg["paths"]["paragraphs"])}

    t0 = time.time()
    runner = Runner(cfg, run_dir)
    checks = {"n0_prompt_identity_violations": [], "n0_output_identity_mismatches": [],
              "gold_integrity_violations": []}
    print(f"[{cfg['_run_id']}] {len(rows)} questions, N={cfg['n_levels']}, rounds 0..{cfg['t_max']}", flush=True)
    runner.closed_book(rows)
    for n in cfg["n_levels"]:
        runner.condition(rows, n, pools, paragraphs, checks)
        print(f"  N={n} done ({time.time() - t0:.0f}s)", flush=True)
    runner.close()

    hard_fail = bool(checks["n0_prompt_identity_violations"] or checks["gold_integrity_violations"])
    manifest = {
        "run_id": cfg["_run_id"], "status": "FAILED_HARD_CHECK" if hard_fail else "OK",
        "config_hash": cfg["_config_hash"], "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_yaml": raw_cfg, "n_questions": len(rows), "n_records": runner.n_records,
        "n_records_not_fitting_context": runner.n_not_fit,
        "max_prompt_tokens_by_N_round": {f"N={k[0]}|round={k[1]}": v for k, v in sorted(
            runner.max_prompt_tokens.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))},
        "checks": {k: {"count": len(v), "items": v[:50]} for k, v in checks.items()},
        "prompt": {"id": cfg["prompt_id"], "sha256": PROMPTS[cfg["prompt_id"]]["sha256"]},
        "deviations_from_simrag": cfg["deviations_from_simrag"],
        "environment": environment(), "elapsed_s": round(time.time() - t0, 1),
        "inputs_sha256": {k: sha(Path(cfg["paths"][k]).read_text()) for k in ("sample", "pool", "paragraphs")},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run_dir}  records={runner.n_records}  not_fitting={runner.n_not_fit}  status={manifest['status']}")
    for k, v in checks.items():
        print(f"  check {k}: {len(v)}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
