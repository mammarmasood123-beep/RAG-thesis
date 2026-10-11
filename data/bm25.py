"""BM25 over a paragraph corpus, using rank_bm25 as the reference implementation.

rank_bm25.BM25Okapi supplies the corpus statistics (idf, avgdl, k1, b, epsilon floor).
Its get_scores() loops in Python over every document for every query term, which is
~0.2 s/query over ~60k paragraphs. `scores()` below evaluates the *same* BM25Okapi
formula through an inverted index (only documents containing a query term receive a
non-zero term contribution, so the result is identical). `reference_scores()` calls
rank_bm25 directly; build_pool.py cross-checks the two on real queries.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, docs: list[str]):
        tokenized = [tokenize(d) for d in docs]
        self.ref = BM25Okapi(tokenized)
        self.doc_len = np.asarray(self.ref.doc_len, dtype=np.float64)
        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for i, freqs in enumerate(self.ref.doc_freqs):
            for term, tf in freqs.items():
                ids, tfs = postings[term]
                ids.append(i)
                tfs.append(tf)
        self.postings = {t: (np.asarray(i), np.asarray(f, dtype=np.float64))
                         for t, (i, f) in postings.items()}
        self.n_docs = len(docs)

    def scores(self, query: str) -> np.ndarray:
        k1, b, avgdl = self.ref.k1, self.ref.b, self.ref.avgdl
        out = np.zeros(self.n_docs)
        for q in tokenize(query):          # repeats kept, exactly as BM25Okapi.get_scores
            if q not in self.postings:
                continue                   # BM25Okapi: idf.get(q) or 0 -> contributes 0
            ids, tf = self.postings[q]
            idf = self.ref.idf.get(q) or 0
            out[ids] += idf * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.doc_len[ids] / avgdl)))
        return out

    def reference_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.ref.get_scores(tokenize(query)))
