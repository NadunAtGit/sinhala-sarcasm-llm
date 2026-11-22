"""Exact nearest-neighbour retrieval over train embeddings — Dev Plan v2 §6, §7.

- Index = train only (by construction: only train_<view>.npy is ever loaded as the index).
- Queries use precomputed val/test vectors (notebook 05), so no model or GPU is needed here.
- Exact cosine similarity (vectors are L2-normalised): deterministic, unlike approximate search.
- At most `max_per_post` neighbours per train post, so k examples are not one comment thread.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    pass


class Retriever:
    def __init__(self, emb_dir: str | Path, view: str, train: pd.DataFrame,
                 forbidden_posts: set[str] = frozenset(), max_per_post: int = 2):
        self.dir, self.view, self.max_per_post = Path(emb_dir), view, max_per_post
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        ids = (self.dir / "train_ids.txt").read_text(encoding="utf-8").split()
        if ids != train["id"].astype(str).tolist():
            raise ValueError("train_ids.txt order differs from the train DataFrame")
        self.E = np.load(self.dir / f"train_{view}.npy").astype(np.float32)
        self.train_ids = np.array(ids)
        self.train_posts = train["post_id"].astype(str).to_numpy()
        bad = set(self.train_posts) & set(forbidden_posts)
        if bad:
            raise LeakageError(f"{len(bad)} train posts also appear in val/test")
        self._q: dict[str, tuple[dict, np.ndarray]] = {}

    def _queries(self, split: str):
        if split == "train":
            raise LeakageError("queries must come from val or test, never train")
        if split not in self._q:
            ids = (self.dir / f"{split}_ids.txt").read_text(encoding="utf-8").split()
            Q = np.load(self.dir / f"{split}_{self.view}.npy").astype(np.float32)
            self._q[split] = ({i: n for n, i in enumerate(ids)}, Q)
        return self._q[split]

    def _select(self, scores: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        n = len(scores)
        pool = min(max(8 * k, 64), n)
        while True:
            cand = np.argpartition(-scores, pool - 1)[:pool] if pool < n else np.arange(n)
            cand = cand[np.argsort(-scores[cand], kind="stable")]
            picked, per_post = [], {}
            for j in cand:
                p = self.train_posts[j]
                if per_post.get(p, 0) >= self.max_per_post:
                    continue
                per_post[p] = per_post.get(p, 0) + 1
                picked.append(int(j))
                if len(picked) == k:
                    break
            if len(picked) == k or pool == n:
                return picked, [float(scores[j]) for j in picked]
            pool = min(pool * 4, n)        # too many candidates from the same posts: widen the pool

    def neighbours(self, split: str, query_id: str, k: int) -> list[dict]:
        idx, Q = self._queries(split)
        picked, sims = self._select(self.E @ Q[idx[str(query_id)]], k)
        return [{"id": str(self.train_ids[j]), "sim": s} for j, s in zip(picked, sims)]

    def neighbours_all(self, split: str, k: int, chunk: int = 1024) -> tuple[list[str], np.ndarray]:
        """Top-k train row indices for every query of a split (for the sanity check)."""
        idx, Q = self._queries(split)
        q_ids = sorted(idx, key=idx.get)
        out = np.empty((len(q_ids), k), dtype=np.int64)
        for s in range(0, len(q_ids), chunk):
            S = Q[s:s + chunk] @ self.E.T
            for r in range(S.shape[0]):
                out[s + r] = self._select(S[r], k)[0]
        return q_ids, out