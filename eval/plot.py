"""Accuracy vs round count, one line per N, from results/cells.csv.

Two panels, one shared y-scale each (EM, F1): no dual axes. T=0 (closed-book) is the same
value for every N and is drawn once as a neutral grey marker. Bands are 95% bootstrap CIs.
Colours: categorical slots 1-4 of the dataviz reference palette in fixed order by N level,
so N=0 is always blue and N=4 keeps its colour when N=1 and N=2 are added.

Usage: python -m eval.plot runs/<run_id>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 4: "#eda100"}   # fixed slot per N level
INK, INK_2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#8a8984"


def main(argv=None):
    run_dir = Path((argv or sys.argv[1:])[0])
    rows = list(csv.DictReader(open(run_dir / "results" / "cells.csv")))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    design_t = set(manifest["config"]["design_t_levels"])
    stub = manifest["config"]["generator"]["backend"] == "stub"
    levels = sorted({int(r["N"]) for r in rows})

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, facecolor=SURFACE)
    for ax, metric in zip(axes, ("EM", "F1")):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        t0 = next(r for r in rows if int(r["T"]) == 0)
        ax.errorbar([0], [100 * float(t0[metric])],
                    yerr=[[100 * (float(t0[metric]) - float(t0[metric + "_lo"]))],
                          [100 * (float(t0[metric + "_hi"]) - float(t0[metric]))]],
                    fmt="o", ms=8, color=NEUTRAL, mec=SURFACE, mew=2, capsize=0, lw=2, zorder=4)
        ax.annotate("closed-book", (0, 100 * float(t0[metric])), xytext=(8, -14),
                    textcoords="offset points", color=INK_2, fontsize=9)
        for n in levels:
            pts = sorted((int(r["T"]), r) for r in rows if int(r["N"]) == n and int(r["T"]) >= 1)
            ts = [t for t, _ in pts]
            y = [100 * float(r[metric]) for _, r in pts]
            lo = [100 * float(r[metric + "_lo"]) for _, r in pts]
            hi = [100 * float(r[metric + "_hi"]) for _, r in pts]
            c = SERIES.get(n, INK_2)
            ax.fill_between(ts, lo, hi, color=c, alpha=0.12, linewidth=0)
            ax.plot(ts, y, color=c, lw=2, zorder=3, label=f"N = {n} distractors / round")
            ax.plot(ts, y, "o", ms=8, color=c, mec=SURFACE, mew=2, zorder=4)
            ax.annotate(f"N={n}", (ts[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        ax.axvline(2.5, color=GRID, lw=1, ls="--", zorder=1)
        ax.text(2.55, 1.0, "gold-free rounds →", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8.5, color=INK_2)
        ax.set_xticks(range(0, 7))
        ax.set_xticklabels([f"{t}" if t in design_t or t == 0 else f"({t})" for t in range(0, 7)])
        ax.set_xlim(-0.4, 6.9)
        ax.set_ylim(0, 100)
        ax.set_xlabel("round T (forced; answer logged at end of round)")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(metric, loc="left", color=INK, fontsize=11, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left", ncol=len(levels), fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.005, 0.89 if stub else 0.93))
    title = f"Experiment A pilot — accuracy vs rounds (n={manifest['n_questions']}, bands = 95% bootstrap CI)"
    if stub:
        title += "\nSTUB GENERATOR — synthetic numbers, harness check only"
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.text(0.01, 0.005, "T in parentheses: logged but not a design level. Gold arrives in rounds 1–2; "
             "T=1 holds one hop only.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.83 if stub else 0.87))
    out = run_dir / "results" / "accuracy_vs_T.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
