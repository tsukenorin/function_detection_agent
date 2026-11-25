"""领域知识库 RAG 检索器（代码感知 BM25 + 可插拔向量重排协议）。

特点：
1. 严格遵循 ``agent.interfaces.KnowledgeRetriever`` 协议；
2. 零重量级外部依赖即可离线运行（Okapi BM25 + 代码标识符/安全 API 特征 + 项目弱加权）；
3. 预留 ``EmbeddingProvider`` 协议，支持随时挂载 Qwen/OpenAI/本地嵌入模型做混合向量检索；
4. 内置 ``exclude_commit`` 防泄露守卫，绝不返回与待测样本同 commit 的案例。
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from agent.schema import KnowledgeHit, KnowledgeRetrieveRecord

from .store import CVECaseRecord, load_cve_cases, tokenize_code

__all__ = ["EmbeddingProvider", "CVEKnowledgeRetriever"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """可选的向量嵌入协议（如 Qwen 向量 API 或开源嵌入模型）。"""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """返回与 ``texts`` 等长的浮点向量列表。"""
        ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class CVEKnowledgeRetriever:
    """基于历史 CVE 成对案例的领域知识库检索器（BM25 + 可选向量重排）。"""

    def __init__(
        self,
        cases: Sequence[CVECaseRecord],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.cases: list[CVECaseRecord] = list(cases)
        self.k1 = k1
        self.b = b
        self.embedding_provider = embedding_provider

        self._doc_lens: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._idf: dict[str, float] = {}
        self._avgdl: float = 1.0
        self._build_index()

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path = "data/knowledge.jsonl",
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> CVEKnowledgeRetriever:
        cases = load_cve_cases(path)
        return cls(cases, embedding_provider=embedding_provider)

    def _build_index(self) -> None:
        n = len(self.cases)
        if n == 0:
            return
        total_len = 0
        for doc_id, case in enumerate(self.cases):
            # 索引漏洞代码 + 修复差异 + CVE/CWE 描述 + 项目名
            doc_text = "\n".join(
                [
                    case.project,
                    " ".join(case.cwe),
                    case.cve_desc[:400],
                    case.commit_message[:250],
                    case.vuln_func[:3000],
                    case.fix_pattern[:600],
                ]
            )
            tokens = tokenize_code(doc_text, max_tokens=320)
            tf = Counter(tokens)
            dl = sum(tf.values())
            self._doc_lens.append(dl)
            total_len += dl
            for tok, freq in tf.items():
                self._postings[tok].append((doc_id, freq))

        self._avgdl = max(1.0, total_len / n)
        for tok, plist in self._postings.items():
            df = len(plist)
            self._idf[tok] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def retrieve(
        self,
        query: str,
        *,
        func_code: str = "",
        top_k: int = 3,
        exclude_commit: str | None = None,
    ) -> KnowledgeRetrieveRecord:
        """检索与待测函数最相关的历史 CVE 案例，并产出系统审计记录。"""
        t0 = time.perf_counter()
        query_summary = (query or "").strip()[:160] or "(func_code)"
        if top_k <= 0 or not self.cases:
            return KnowledgeRetrieveRecord(
                query=query_summary,
                top_k=top_k,
                hits=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        raw_query = f"{query}\n{func_code[:4000]}" if func_code else query
        q_tokens = tokenize_code(raw_query, max_tokens=160)
        q_counts = Counter(q_tokens)

        scores: dict[int, float] = defaultdict(float)
        excluded_sha = (exclude_commit or "").strip().lower()

        for tok, q_tf in q_counts.items():
            plist = self._postings.get(tok)
            if not plist:
                continue
            idf = self._idf[tok]
            q_weight = 1.0 + math.log(q_tf)
            for doc_id, tf in plist:
                dl = self._doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
                scores[doc_id] += q_weight * idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-9)

        # 过滤掉任何同 commit_id 案例（防泄露守卫）
        ranked = sorted(
            (
                (doc_id, sc)
                for doc_id, sc in scores.items()
                if not excluded_sha or self.cases[doc_id].commit_id != excluded_sha
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        # 可选：若挂载了向量嵌入提供者，对 top-20 候选做混合重排
        candidates = ranked[: max(top_k * 4, 15)]
        if self.embedding_provider is not None and candidates:
            cand_texts = [
                f"{self.cases[d].cve_desc[:300]}\n{self.cases[d].vuln_func[:1000]}"
                for d, _ in candidates
            ]
            vecs = self.embedding_provider.embed_texts([raw_query[:1200], *cand_texts])
            q_vec, doc_vecs = vecs[0], vecs[1:]
            max_bm25 = max((sc for _, sc in candidates), default=1.0) or 1.0
            reranked = []
            for (doc_id, bm25_sc), d_vec in zip(candidates, doc_vecs):
                hybrid = 0.55 * (bm25_sc / max_bm25) + 0.45 * _cosine(q_vec, d_vec)
                reranked.append((doc_id, hybrid))
            candidates = sorted(reranked, key=lambda x: x[1], reverse=True)

        hits: list[KnowledgeHit] = []
        for doc_id, score in candidates[:top_k]:
            c = self.cases[doc_id]
            hits.append(
                KnowledgeHit(
                    case_id=c.case_id,
                    cve=c.cve,
                    cwe=list(c.cwe),
                    project=c.project or None,
                    score=round(float(score), 4),
                    cve_desc=c.cve_desc[:450],
                    commit_message=c.commit_message[:250],
                    fix_pattern=c.fix_pattern,
                    vuln_snippet=c.vuln_func[:500],
                    fixed_snippet=c.fixed_func[:500],
                )
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return KnowledgeRetrieveRecord(
            query=query_summary,
            top_k=top_k,
            hits=hits,
            latency_ms=elapsed_ms,
        )
