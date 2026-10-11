"""Build the per-question distractor pool: BM25 over HotpotQA's own paragraph corpus,
with answer-bearing candidates excluded (Design00 III.5 amendment, accepted 2026-10-08).

Pilot pool (pool.type = "bm25_hotpot_corpus_top50"). The corpus is every paragraph that
appears in the split's distractor contexts, deduplicated by title. For each question:
  1. query = question text; rank all paragraphs by BM25 (score > 0; ties by title);
  2. drop the question's own gold titles (gold is excluded by TITLE and never screened);
     the remaining candidates carry bm25_rank = 1, 2, 3, ...;
  3. III.5 answer exclusion: drop any candidate whose TITLE + TEXT contains the gold answer as
     a WHOLE-TOKEN sequence after HotpotQA's normalize_answer on both sides. If
     normalize_answer(answer) is empty (e.g. 'The The', '!!!'), fall back to a whole-token match
     of the raw lowercased answer against the raw lowercased title + text (whitespace tokens).
     Skipped when the normalised answer is "yes" or "no";
  4. keep the first 50 survivors in BM25 order, scanning no deeper than bm25_rank 300;
     survivors carry pool_rank = 1..50.
Draws later use pool_rank with rank band 1-50.

STOP GATES (exit 2, nothing written): any question with fewer than 50 survivors within the
cap; or any question with fewer than --min_usable (24) survivors. Use --verify_only to print
the report and write nothing.

The full study replaces this corpus with full Wikipedia; this file is the pilot version.

Usage:
  python -m data.build_pool --hotpot hotpot_dev_distractor_v1.json --out artifacts/ --verify_only
  python -m data.build_pool --hotpot hotpot_dev_distractor_v1.json --out artifacts/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.bm25 import BM25Index
from data.hotpot_io import load_hotpot
from eval.hotpot_official import normalize_answer

POOL_TYPE = "bm25_hotpot_corpus_top50"
POOL_SIZE = 50
MAX_BM25_RANK = 300
ANSWER_FILTER = ("whole-token on title + text after normalize_answer (both sides); raw-lowercase "
                 "whole-token fallback when normalize_answer(answer) is empty; skipped for yes/no; "
                 "gold never screened")


def build_corpus(items):
    """Dedup paragraphs by title. Returns (titles, texts, n_slots, conflicts)."""
    first_text: dict[str, str] = {}
    variants: dict[str, set] = defaultdict(set)
    n_slots = 0
    for it in items:
        for title, text in it.context:
            n_slots += 1
            variants[title].add(text)
            first_text.setdefault(title, text)
    titles = sorted(first_text)
    texts = [first_text[t] for t in titles]
    conflicts = {t: len(v) for t, v in variants.items() if len(v) > 1}
    return titles, texts, n_slots, conflicts


def _order(scores: np.ndarray):
    return np.lexsort((np.arange(len(scores)), -scores))  # score desc; ties by index = title asc


def top_k_excluding(scores: np.ndarray, titles: list[str], exclude: set[str], k: int):
    """Unfiltered top-k by score (desc), ties by title (asc); score must be > 0.
    Kept for the pre-amendment comparison (shipped-8 recovery before filtering)."""
    out = []
    for i in _order(scores):
        s = float(scores[i])
        if s <= 0:
            break
        if titles[i] in exclude:
            continue
        out.append({"title": titles[i], "rank": len(out) + 1, "score": round(s, 6)})
        if len(out) == k:
            break
    return out


def _pad(s: str) -> str:
    return " " + " ".join(s.split()) + " "


def candidate_strings(titles: list[str], texts: list[str]) -> tuple[list[str], list[str]]:
    """Per candidate, the strings the matcher searches, whitespace-padded:
    (normalize_answer("title text"), raw lowercased "title text")."""
    norm = [_pad(normalize_answer(f"{t} {x}")) for t, x in zip(titles, texts)]
    raw = [_pad(f"{t} {x}".lower()) for t, x in zip(titles, texts)]
    return norm, raw


def answer_key(answer: str) -> tuple[str, str]:
    """(mode, key): mode is 'skip_yes_no', 'normalized', or 'raw_fallback'."""
    na = normalize_answer(answer)
    if na in ("yes", "no"):
        return "skip_yes_no", na
    if na:
        return "normalized", " ".join(na.split())
    return "raw_fallback", " ".join(answer.lower().split())


def contains_answer(mode: str, key: str, norm_padded: str, raw_padded: str) -> bool:
    """III.5 matcher: whole-token match of key in the padded candidate string for its mode."""
    if mode == "skip_yes_no" or not key:
        return False
    return f" {key} " in (norm_padded if mode == "normalized" else raw_padded)


def filter_candidates(scores: np.ndarray, titles: list[str], cand_norm: list[str],
                      cand_raw: list[str], exclude: set[str], answer: str,
                      k: int = POOL_SIZE, max_bm25_rank: int = MAX_BM25_RANK):
    """III.5 pool: first k non-gold candidates in BM25 order that do not contain the answer
    (see contains_answer). Returns (candidates, stats)."""
    mode, key = answer_key(answer)
    out, removed, bm25_rank = [], 0, 0
    for i in _order(scores):
        s = float(scores[i])
        if s <= 0:
            break
        if titles[i] in exclude:
            continue
        bm25_rank += 1
        if bm25_rank > max_bm25_rank:
            bm25_rank -= 1
            break
        if contains_answer(mode, key, cand_norm[i], cand_raw[i]):
            removed += 1
            continue
        out.append({"title": titles[i], "pool_rank": len(out) + 1, "bm25_rank": bm25_rank,
                    "score": round(s, 6)})
        if len(out) == k:
            break
    stats = {"applied": mode != "skip_yes_no", "match_mode": mode, "match_key": key,
             "n_removed": removed, "max_bm25_rank": out[-1]["bm25_rank"] if out else bm25_rank,
             "scanned_to_bm25_rank": bm25_rank}
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_usable", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--n_show", type=int, default=5)
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args(argv)

    items = load_hotpot(args.hotpot)
    titles, texts, n_slots, conflicts = build_corpus(items)

    print("=" * 72)
    print(f"POOL VERIFICATION  ({POOL_TYPE}, III.5 answer exclusion)")
    print("=" * 72)
    print(f"questions in split                : {len(items)}")
    print(f"paragraph slots (questions x 10)  : {len(items) * 10}")
    print(f"paragraph slots (actual count)    : {n_slots}")
    print(f"UNIQUE paragraph titles           : {len(titles)}")
    print(f"  duplicate slots removed         : {n_slots - len(titles)} "
          f"({100 * (n_slots - len(titles)) / n_slots:.1f}% of slots)")
    print(f"  titles with >1 distinct text    : {len(conflicts)}"
          + (" (first occurrence kept)" if conflicts else ""))
    print(f"answer filter                     : {ANSWER_FILTER}")
    print(f"pool size / scan cap              : {POOL_SIZE} survivors / bm25_rank <= {MAX_BM25_RANK}")

    print("\nIndexing (title + paragraph text)...", flush=True)
    index = BM25Index([f"{t} {x}" for t, x in zip(titles, texts)])
    cand_norm, cand_raw = candidate_strings(titles, texts)

    rng = random.Random(args.seed)
    check = rng.sample(items, min(3, len(items)))
    max_diff = max(float(np.max(np.abs(index.scores(it.question) - index.reference_scores(it.question))))
                   for it in check)
    print(f"fast scorer vs rank_bm25.get_scores, max |diff| on {len(check)} queries: {max_diff:.2e}")
    if max_diff > 1e-9:
        print("STOP: fast BM25 scorer disagrees with rank_bm25. Do not use this pool.")
        return 3

    print("Scoring every question...", flush=True)
    title_idx = {t: i for i, t in enumerate(titles)}
    pools, stats, recall_new, recall_old = {}, {}, [], []
    substring_left, fallback_qids = 0, []
    for it in items:
        sc = index.scores(it.question)
        gold = set(it.sf_titles)
        cands, st = filter_candidates(sc, titles, cand_norm, cand_raw, gold, it.answer)
        pools[it.qid], stats[it.qid] = cands, st
        if st["match_mode"] == "raw_fallback":
            fallback_qids.append((it.qid, it.answer, st["match_key"]))
        if st["applied"] and st["match_key"]:   # audit only: plain substring still present in survivors
            hay = cand_norm if st["match_mode"] == "normalized" else cand_raw
            substring_left += sum(1 for c in cands if st["match_key"] in hay[title_idx[c["title"]]])
        shipped = set(it.context_titles) - gold
        if shipped:
            recall_new.append(len(shipped & {c["title"] for c in cands}) / len(shipped))
            old = top_k_excluding(sc, titles, gold, POOL_SIZE)
            recall_old.append(len(shipped & {c["title"] for c in old}) / len(shipped))

    by_qid = {it.qid: it for it in items}
    usable = np.array([len(pools[q]) for q in pools])
    removed = np.array([stats[q]["n_removed"] for q in pools])
    maxrank = np.array([stats[q]["max_bm25_rank"] for q in pools])
    applied = np.array([stats[q]["applied"] for q in pools])

    show = rng.sample(items, min(args.n_show, len(items)))
    print(f"\n{args.n_show} random questions (seed {args.seed}): survivors after gold + answer exclusion")
    for it in show:
        st, c = stats[it.qid], pools[it.qid]
        print(f"  {it.qid}  [{it.q_type:<10}] usable={len(c):>2}  removed={st['n_removed']:>3}  "
              f"max_bm25_rank={st['max_bm25_rank']:>3}  filter={'on ' if st['applied'] else 'off'}  "
              f"answer={it.answer[:40]!r}")

    print(f"\nWHOLE SPLIT ({len(items)} questions)")
    print(f"  usable candidates (survivors)     : min={usable.min()}  median={np.median(usable):.0f}  "
          f"questions with < {POOL_SIZE}: {(usable < POOL_SIZE).sum()}  with < {args.min_usable}: "
          f"{(usable < args.min_usable).sum()}")
    print(f"  filter applied / skipped (yes/no) : {applied.sum()} / {(~applied).sum()}")
    print(f"  candidates removed per question   : mean={removed.mean():.2f}  max={removed.max()}  "
          f"questions with >=1 removed: {(removed > 0).sum()}")
    print(f"  questions needing bm25_rank > 50  : {(maxrank > POOL_SIZE).sum()}  "
          f"max bm25_rank used: {maxrank.max()}")
    print(f"  shipped-8 recovered in pool       : mean {100 * np.mean(recall_new):.1f}% after filter "
          f"(before filter: {100 * np.mean(recall_old):.1f}%)")
    print(f"  survivors still containing the answer as a PLAIN SUBSTRING (title+text, audit only): "
          f"{substring_left}")
    print(f"  raw-lowercase fallback used (normalize_answer empty): {len(fallback_qids)} "
          + str([(q, a, k) for q, a, k in fallback_qids]))

    short = sorted(q for q in pools if len(pools[q]) < POOL_SIZE)
    if short:
        print(f"\nSTOP: {len(short)} question(s) cannot reach {POOL_SIZE} survivors within "
              f"bm25_rank {MAX_BM25_RANK}:")
        for q in short[:50]:
            st = stats[q]
            print(f"  {q}  [{by_qid[q].q_type}]  usable={len(pools[q])}  removed={st['n_removed']}  "
                  f"scanned_to={st['scanned_to_bm25_rank']}  answer={by_qid[q].answer!r}")
        print("Pool NOT written.")
        return 2

    print(f"\nPASS: every question has {POOL_SIZE} survivors (>= {args.min_usable} required).")
    if args.verify_only:
        print("--verify_only: nothing written.")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pool.jsonl", "w", encoding="utf-8") as f:
        for qid, cands in pools.items():
            f.write(json.dumps({"qid": qid, "pool_type": POOL_TYPE, "answer_filter": stats[qid],
                                "candidates": cands}) + "\n")
    with open(out / "paragraphs.jsonl", "w", encoding="utf-8") as f:
        for t, x in zip(titles, texts):
            f.write(json.dumps({"title": t, "text": x}) + "\n")
    (out / "pool_report.json").write_text(json.dumps({
        "pool_type": POOL_TYPE, "hotpot_file": str(args.hotpot), "questions": len(items),
        "slots": n_slots, "unique_titles": len(titles), "title_text_conflicts": len(conflicts),
        "pool_size": POOL_SIZE, "max_bm25_rank_cap": MAX_BM25_RANK, "answer_filter": ANSWER_FILTER,
        "min_usable_required": args.min_usable, "usable_min": int(usable.min()),
        "filter_applied": int(applied.sum()), "filter_skipped_yes_no": int((~applied).sum()),
        "removed_mean": float(removed.mean()), "removed_max": int(removed.max()),
        "questions_past_rank_50": int((maxrank > POOL_SIZE).sum()), "max_bm25_rank_used": int(maxrank.max()),
        "shipped_recall_mean_after_filter": float(np.mean(recall_new)),
        "shipped_recall_mean_before_filter": float(np.mean(recall_old)),
        "survivors_with_plain_substring_answer": substring_left,
        "raw_fallback_questions": [q for q, _, _ in fallback_qids],
        "bm25_crosscheck_max_diff": max_diff,
    }, indent=2))
    print(f"wrote {out/'pool.jsonl'}, {out/'paragraphs.jsonl'}, {out/'pool_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
