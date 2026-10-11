"""Aggregate an Experiment A run into the pilot tables.

Outputs (in runs/<run_id>/results/):
  cells.csv                 EM/F1 per (N, T) with question-level bootstrap 95% CIs
  cells_sensitivity.csv     same, excluding items answered correctly closed-book
  closed_book_correct.txt   qids with closed-book EM = 1 (flag for sensitivity analysis)
  summary.md                headline tables, N=0 control, closed-book by type, paired
                            N-vs-0 differences, parse/unsure rates, token ceiling

T = 0 is the closed-book answer, shared by every N line (it does not depend on N).
Records whose prompt did not fit the context window are excluded and counted.

Usage: python -m eval.aggregate runs/<run_id>
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.hotpot_official import normalize_answer

N_BOOT = 2000
BOOT_SEED = 20261008


def load(run_dir: Path):
    recs = [json.loads(line) for line in open(run_dir / "records.jsonl", encoding="utf-8")]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return recs, manifest


def boot_ci(values: np.ndarray, rng) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def by_cell(recs, exclude=frozenset()):
    """{(N, T): {qid: (em, f1)}}, closed-book copied into every N's T=0."""
    closed = {r["qid"]: r for r in recs if r["condition"]["arm"] == "closed_book"}
    levels = sorted({r["condition"]["N"] for r in recs if r["condition"]["arm"] == "distractor"})
    cells, not_fit = defaultdict(dict), defaultdict(int)
    for r in recs:
        if r["qid"] in exclude:
            continue
        keys = [(n, 0) for n in levels] if r["condition"]["arm"] == "closed_book" else \
               [(r["condition"]["N"], r["round"])]
        for k in keys:
            if r["score"] is None:
                not_fit[k] += 1
            else:
                cells[k][r["qid"]] = (r["score"]["em"], r["score"]["f1"])
    return cells, not_fit, levels, closed


def cell_table(cells, not_fit, rng):
    rows = []
    for (n, t) in sorted(cells):
        v = np.array(list(cells[(n, t)].values()), dtype=float)
        em_ci, f1_ci = boot_ci(v[:, 0], rng), boot_ci(v[:, 1], rng)
        rows.append({"N": n, "T": t, "n": len(v), "not_fit": not_fit[(n, t)],
                     "EM": v[:, 0].mean(), "EM_lo": em_ci[0], "EM_hi": em_ci[1],
                     "F1": v[:, 1].mean(), "F1_lo": f1_ci[0], "F1_hi": f1_ci[1]})
    return rows


PAIRED_N_BOOT = 10_000


def paired_bootstrap(cells, n_a, n_b, t, metric=0, n_boot=PAIRED_N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap over questions: diff = score(N=n_a) - score(N=n_b) at round t, on the
    questions present in both cells. Resamples questions with replacement (n_boot times) and
    takes the 2.5/97.5 percentiles of the mean diff. Own RNG, so other tables are unaffected."""
    common = sorted(set(cells.get((n_a, t), {})) & set(cells.get((n_b, t), {})))
    d = np.array([cells[(n_a, t)][q][metric] - cells[(n_b, t)][q][metric] for q in common], dtype=float)
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": len(d)}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(row, m):
    return f"{100 * row[m]:.1f} [{100 * row[m + '_lo']:.1f}, {100 * row[m + '_hi']:.1f}]"


def main(argv=None):
    run_dir = Path((argv or sys.argv[1:])[0])
    out = run_dir / "results"
    out.mkdir(exist_ok=True)
    recs, manifest = load(run_dir)
    rng = np.random.default_rng(BOOT_SEED)
    design_t = manifest["config"]["design_t_levels"]

    cells, not_fit, levels, closed = by_cell(recs)
    table = cell_table(cells, not_fit, rng)
    write_csv(out / "cells.csv", table)

    cb_correct = sorted(q for q, r in closed.items() if r["score"] and r["score"]["em"] == 1)
    (out / "closed_book_correct.txt").write_text("\n".join(cb_correct) + ("\n" if cb_correct else ""))
    cells_s, nf_s, _, _ = by_cell(recs, exclude=frozenset(cb_correct))
    table_s = cell_table(cells_s, nf_s, rng)
    write_csv(out / "cells_sensitivity.csv", table_s)

    L = []
    stub = manifest["config"]["generator"]["backend"] == "stub"
    L.append(f"# Experiment A — {manifest['run_id']}\n")
    if stub:
        L.append("> **STUB GENERATOR. These numbers are synthetic and test the harness only.**\n")
    L.append(f"Questions: {manifest['n_questions']} · records: {manifest['n_records']} · "
             f"not fitting context: {manifest['n_records_not_fitting_context']} · status: {manifest['status']}\n")

    def grid(tbl, title):
        L.append(f"\n## {title}\n")
        ts = sorted({r["T"] for r in tbl})
        L.append("| N \\ T | " + " | ".join(f"{t}{'*' if t in design_t else ''}" for t in ts) + " |")
        L.append("|---|" + "---|" * len(ts))
        for metric in ("EM", "F1"):
            for n in levels:
                cell = {r["T"]: r for r in tbl if r["N"] == n}
                L.append(f"| {metric} N={n} | " + " | ".join(fmt(cell[t], metric) if t in cell else "—"
                                                       for t in ts) + " |")
        L.append("\n`*` = design T level. T=0 is closed-book (shared). Cells: % [95% bootstrap CI].")

    grid(table, "EM / F1 by (N, T)")

    # N=0 control
    if 0 in levels:
        em0 = {r["T"]: r["EM"] for r in table if r["N"] == 0}
        f10 = {r["T"]: r["F1"] for r in table if r["N"] == 0}
        devs = [abs(em0[t] - em0[2]) for t in em0 if t >= 2]
        L.append("\n## N=0 control (must be flat from T=2)\n")
        L.append(f"EM by T: " + ", ".join(f"T{t}={100*em0[t]:.1f}" for t in sorted(em0)))
        L.append(f"\nF1 by T: " + ", ".join(f"T{t}={100*f10[t]:.1f}" for t in sorted(f10)))
        L.append(f"\nmax |EM(T) − EM(2)| for T ≥ 2: **{100 * max(devs):.2f} pts**")
        c = manifest["checks"]
        L.append(f"\nprompt byte-identity violations (hard): {c['n0_prompt_identity_violations']['count']} · "
                 f"output mismatches across T≥2 (soft): {c['n0_output_identity_mismatches']['count']}")

    # Paired differences (10,000 paired bootstrap resamples over questions)
    if 0 in levels and len(levels) > 1:
        t_max = manifest["config"]["t_max"]
        paired_rows = []
        for n in [x for x in levels if x != 0]:
            for t in range(1, t_max + 1):
                for metric, name in ((0, "EM"), (1, "F1")):
                    r = paired_bootstrap(cells, n, 0, t, metric=metric)
                    paired_rows.append({"N": n, "vs_N": 0, "T": t, "metric": name, "n_pairs": r["n"],
                                        "diff": r["diff"], "lo": r["lo"], "hi": r["hi"],
                                        "n_boot": PAIRED_N_BOOT})
        write_csv(out / "paired_ci.csv", paired_rows)
        for name in ("EM", "F1"):
            L.append(f"\n## Paired difference vs N=0, {name} points "
                     f"(same questions; {PAIRED_N_BOOT:,} paired bootstrap resamples over questions)\n")
            L.append("| N | " + " | ".join(f"T={t}" for t in range(1, t_max + 1)) + " |")
            L.append("|---|" + "---|" * t_max)
            for n in [x for x in levels if x != 0]:
                rr = {r["T"]: r for r in paired_rows if r["N"] == n and r["metric"] == name}
                L.append(f"| {n} | " + " | ".join(
                    f"{100*rr[t]['diff']:+.1f} [{100*rr[t]['lo']:+.1f}, {100*rr[t]['hi']:+.1f}]"
                    for t in range(1, t_max + 1)) + " |")
        L.append(f"\nn_pairs per cell: {sorted({r['n_pairs'] for r in paired_rows})}. "
                 f"95% percentile interval; written to `results/paired_ci.csv`.")

    # Closed-book contamination check
    L.append("\n## Closed-book (round 0) — contamination check\n")
    L.append("| subset | n | EM | F1 |")
    L.append("|---|---|---|---|")
    groups = {"all": list(closed.values())}
    for r in closed.values():
        groups.setdefault(r["q_type"], []).append(r)
        if r["q_type"] == "comparison":
            yn = normalize_answer(r["gold_answer"]) in ("yes", "no")
            groups.setdefault("comparison: yes/no gold" if yn else "comparison: entity/category gold", []).append(r)
    for name, rs in groups.items():
        scored = [r for r in rs if r["score"]]
        e = np.mean([r["score"]["em"] for r in scored]) if scored else float("nan")
        f = np.mean([r["score"]["f1"] for r in scored]) if scored else float("nan")
        L.append(f"| {name} | {len(scored)} | {100*e:.1f} | {100*f:.1f} |")
    L.append(f"\nItems answered correctly closed-book (EM=1): **{len(cb_correct)}** "
             f"→ `results/closed_book_correct.txt`. Read closed-book correctness on comparison items "
             f"with care, because they have three answer types with different guessing odds: yes/no "
             f"(two options, so chance is about 50%); one of the two named entities (the question names "
             f"both, so a guess is right about half the time); and a shared category such as "
             f"'flowering plants' (an open answer, so there is no fixed chance level and a correct "
             f"closed-book answer is stronger evidence of prior knowledge). The \"entity/category\" row "
             f"above mixes the last two.")

    grid(table_s, f"Sensitivity: excluding {len(cb_correct)} closed-book-correct items")

    # Output hygiene
    L.append("\n## Output hygiene\n")
    hyg = defaultdict(lambda: [0, 0, 0])
    for r in recs:
        if r["output"]:
            k = ("closed_book", 0) if r["condition"]["arm"] == "closed_book" else (r["condition"]["N"], r["round"])
            hyg[k][0] += 1
            hyg[k][1] += not r["output"]["parse_ok"]
            hyg[k][2] += r["output"]["said_unsure"]
    tot = np.array(list(hyg.values())).sum(axis=0)
    L.append(f"parse failures: {tot[1]}/{tot[0]} · said 'unsure' despite no-abstain prompt: {tot[2]}/{tot[0]}")
    worst = sorted(hyg.items(), key=lambda kv: -kv[1][1])[:3]
    L.append("\nworst cells by parse failures: " + ", ".join(f"{k}: {v[1]}/{v[0]}" for k, v in worst))

    # Token ceiling (Design00 VIII.4 pilot measurement 1)
    L.append("\n## Token ceiling (prompt tokens incl. chat template, max over questions)\n")
    mpt = manifest["max_prompt_tokens_by_N_round"]
    mml = manifest["config"]["generator"]["max_model_len"]
    budget = mml - manifest["config"]["generator"]["max_new_tokens"]
    L.append(f"window {mml}, minus max_new_tokens → prompt budget {budget}\n")
    L.append("| N | " + " | ".join(f"T={t}" for t in range(0, manifest['config']['t_max'] + 1)) + " |")
    L.append("|---|" + "---|" * (manifest['config']['t_max'] + 1))
    for n in levels:
        L.append(f"| {n} | " + " | ".join(str(mpt.get(f"N={'None' if t == 0 else n}|round={t}", "—"))
                                            for t in range(0, manifest['config']['t_max'] + 1)) + " |")
    if stub:
        L.append("\n(stub counts whitespace tokens; the real ceiling comes from the vLLM run)")

    # Task 6a: parse-failure and said-unsure rates per (N, T)
    hyg_rows = hygiene_by_cell(recs)
    write_csv(out / "hygiene_by_cell.csv", hyg_rows)
    L.append("\n## Parse failures and 'unsure' answers by (N, T)\n")
    L.append("Rates over records that ran (records not fitting the context window are counted "
             "separately). Closed-book is round 0.\n")
    L.append("| N | T | n | not_fit | parse_fail | parse_fail_rate | said_unsure | said_unsure_rate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for h in hyg_rows:
        pr = "—" if h["parse_fail_rate"] is None else f"{100 * h['parse_fail_rate']:.1f}%"
        ur = "—" if h["said_unsure_rate"] is None else f"{100 * h['said_unsure_rate']:.1f}%"
        L.append(f"| {h['N']} | {h['T']} | {h['n']} | {h['not_fit']} | {h['parse_fail']} | {pr} | "
                 f"{h['said_unsure']} | {ur} |")

    (out / "summary.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out}/summary.md, cells.csv, cells_sensitivity.csv, closed_book_correct.txt, "
          f"hygiene_by_cell.csv")


def hygiene_by_cell(recs) -> list[dict]:
    """Task 6a. One row per (N, T): parse_ok failure rate and said_unsure rate.
    N is "closed_book" for round 0. Rates are None when no record in the cell ran."""
    cells: dict = defaultdict(lambda: {"n": 0, "not_fit": 0, "parse_fail": 0, "said_unsure": 0})
    for r in recs:
        key = ("closed_book", 0) if r["condition"]["arm"] == "closed_book" else (r["condition"]["N"], r["round"])
        c = cells[key]
        if r["output"] is None:
            c["not_fit"] += 1
            continue
        c["n"] += 1
        c["parse_fail"] += not r["output"]["parse_ok"]
        c["said_unsure"] += bool(r["output"]["said_unsure"])
    rows = []
    for (n, t) in sorted(cells, key=lambda k: (k[0] != "closed_book", str(k[0]) if k[0] == "closed_book" else k[0], k[1])):
        c = cells[(n, t)]
        rows.append({"N": n, "T": t, "n": c["n"], "not_fit": c["not_fit"],
                     "parse_fail": c["parse_fail"],
                     "parse_fail_rate": c["parse_fail"] / c["n"] if c["n"] else None,
                     "said_unsure": c["said_unsure"],
                     "said_unsure_rate": c["said_unsure"] / c["n"] if c["n"] else None})
    return rows


if __name__ == "__main__":
    main()
