"""Hybrid clause search over the policy wordings.

Each '## <id>. <title>' section of a wording becomes one clause. Search combines BM25 (exact terms
such as "depreciation" or "FIR") with dense embeddings (paraphrases such as "drunk" vs "under the
influence of alcohol") using reciprocal rank fusion, then reranks the fused list by term overlap with
the query. Results are always restricted to the product of the claim being processed.

Embeddings: OpenAI text-embedding-3-small when an OpenAI key is configured; otherwise a local
feature-hashing embedding so the system runs offline (weaker, but BM25 still carries exact terms).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the and or of to in on for is are be by with at as it this that any from not "
                  "was were will shall may per than its which who".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


@dataclass
class Clause:
    clause_id: str
    product_code: str
    title: str
    text: str

    def as_dict(self) -> dict:
        return {"clause_id": self.clause_id, "title": self.title, "text": self.text}


def parse_wording(product_code: str, markdown: str) -> list[Clause]:
    clauses, cur_id, cur_title, buf = [], None, None, []
    for line in markdown.splitlines():
        m = re.match(r"^##\s+([A-Z]\d+)\.\s+(.*)$", line)
        if m:
            if cur_id:
                clauses.append(Clause(cur_id, product_code, cur_title, "\n".join(buf).strip()))
            cur_id, cur_title, buf = m.group(1), m.group(2).strip(), []
        elif cur_id:
            buf.append(line)
    if cur_id:
        clauses.append(Clause(cur_id, product_code, cur_title, "\n".join(buf).strip()))
    return clauses


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in set(query):
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class HashingEmbedder:
    """Offline fallback: signed feature hashing of unigrams and bigrams, L2-normalised."""
    name = "local-hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for f in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OpenAIEmbedder:
    def __init__(self, model: str, timeout_s: float = 30.0):
        self.model, self.timeout_s, self.name = model, timeout_s, model

    def embed(self, texts: list[str]) -> np.ndarray:
        base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
        r = requests.post(f"{base}/v1/embeddings", timeout=self.timeout_s,
                          headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
                          json={"model": self.model, "input": texts})
        r.raise_for_status()
        vecs = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class ClauseIndex:
    def __init__(self, clauses: list[Clause], embedder):
        self.clauses = clauses
        self.by_id = {c.clause_id: c for c in clauses}
        self.embedder = embedder
        self._tokens = [tokenize(f"{c.title} {c.text}") for c in clauses]
        self._vectors = embedder.embed([f"{c.title}\n{c.text}" for c in clauses]) if clauses else None

    @classmethod
    def from_dir(cls, policies_dir: Path, products: dict, embedder) -> "ClauseIndex":
        clauses = []
        for code, prod in products.items():
            clauses += parse_wording(code, (Path(policies_dir) / prod["wording_file"]).read_text())
        return cls(clauses, embedder)

    def get(self, clause_ref: str | None) -> Clause | None:
        if not clause_ref:
            return None
        return self.by_id.get(re.match(r"[A-Z]\d+", clause_ref).group(0)) if re.match(r"[A-Z]\d+", clause_ref) else None

    def search(self, query: str, product_code: str, k: int = 4, rrf_k: int = 60) -> list[Clause]:
        idx = [i for i, c in enumerate(self.clauses) if c.product_code == product_code]
        if not idx or not query.strip():
            return []
        q = tokenize(query)
        bm = BM25([self._tokens[i] for i in idx]).scores(q)
        qv = self.embedder.embed([query])[0]
        dense = [float(self._vectors[i] @ qv) for i in idx]
        fused: dict[int, float] = {}
        for ranking in (sorted(range(len(idx)), key=lambda j: -bm[j]), sorted(range(len(idx)), key=lambda j: -dense[j])):
            for r, j in enumerate(ranking):
                fused[j] = fused.get(j, 0.0) + 1.0 / (rrf_k + r + 1)
        top = sorted(fused, key=lambda j: -fused[j])[: max(k * 2, k)]
        qs = set(q)
        rerank = sorted(top, key=lambda j: (-(len(qs & set(self._tokens[idx[j]])) / (len(qs) or 1)), -fused[j]))
        return [self.clauses[idx[j]] for j in rerank[:k]]
