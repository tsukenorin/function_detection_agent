"""领域知识库 RAG 检索器（双通道混合检索 + RRF 融合 + 去重多样性 + 相关性阈值）。

升级要点：
1. **双通道混合召回**：
   - 通道 A（Code BM25）：保留同项目/同 API 族的高精度词面匹配；
   - 通道 B（Security Pattern BM25）：基于离线提取的危险原语（``sink_tags``）、
     新增守卫（``guard_tags``）与 CWE 描述实现跨项目漏洞模式召回；
   - 通过 RRF（Reciprocal Rank Fusion）与归一化相关性分数融合，并支持挂载
     ``EmbeddingProvider`` 向量重排。
2. **反劫持多样性去重**：
   - 按 ``CVE`` 严格去重，避免同一 CVE 占满多个返回槽位；
   - 同一函数名最多占 ``max_same_func``（默认 1）个槽位，避免同名旧漏洞彻底劫持注意力。
3. **相关性阈值过滤（宁缺毋滥）**：
   - 低于 ``min_score`` 的低相关噪音直接丢弃，不强行塞满 ``top_k``。
4. **防泄露守卫**：内置 ``exclude_commit`` 过滤。
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from agent.schema import KnowledgeHit, KnowledgeRetrieveRecord

from .store import (
    CVECaseRecord,
    extract_security_signals,
    load_cve_cases,
    tokenize_code,
)

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


class _BM25Index:
    """轻量级内存 Okapi BM25 倒排索引。"""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl: float = 1.0

    def build(self, docs_tokens: Sequence[list[str]]) -> None:
        n = len(docs_tokens)
        if n == 0:
            return
        total_len = 0
        for doc_id, tokens in enumerate(docs_tokens):
            tf = Counter(tokens)
            dl = sum(tf.values())
            self.doc_lens.append(dl)
            total_len += dl
            for tok, freq in tf.items():
                self.postings[tok].append((doc_id, freq))

        self.avgdl = max(1.0, total_len / n)
        for tok, plist in self.postings.items():
            df = len(plist)
            self.idf[tok] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, q_tokens: Sequence[str]) -> dict[int, float]:
        if not q_tokens or not self.doc_lens:
            return {}
        q_counts = Counter(q_tokens)
        scores: dict[int, float] = defaultdict(float)
        for tok, q_tf in q_counts.items():
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = self.idf[tok]
            q_weight = 1.0 + math.log(q_tf)
            for doc_id, tf in plist:
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += q_weight * idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-9)
        return scores


class CVEKnowledgeRetriever:
    """基于历史 CVE 成对案例的双通道混合领域知识库检索器。"""

    def __init__(
        self,
        cases: Sequence[CVECaseRecord],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        default_min_score: float = 0.18,
        max_same_func: int = 1,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.cases: list[CVECaseRecord] = list(cases)
        self.default_min_score = default_min_score
        self.max_same_func = max(1, int(max_same_func))
        self.embedding_provider = embedding_provider

        self._code_index = _BM25Index(k1=k1, b=b)
        self._pattern_index = _BM25Index(k1=1.2, b=0.5)
        self._build_indices()

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path = "data/knowledge.jsonl",
        *,
        default_min_score: float = 0.18,
        max_same_func: int = 1,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> CVEKnowledgeRetriever:
        cases = load_cve_cases(path)
        return cls(
            cases,
            default_min_score=default_min_score,
            max_same_func=max_same_func,
            embedding_provider=embedding_provider,
        )

    def _build_indices(self) -> None:
        if not self.cases:
            return
        code_docs: list[list[str]] = []
        pattern_docs: list[list[str]] = []
        for case in self.cases:
            code_text = "\n".join(
                [
                    case.project,
                    case.func_name or "",
                    case.vuln_func[:3000],
                    case.fix_pattern[:600],
                ]
            )
            code_docs.append(tokenize_code(code_text, max_tokens=280))

            # 通道 B：强调跨项目通用漏洞模式（sink + guard + CWE + 漏洞描述）
            sec_tokens = list(case.sink_tags) * 2 + list(case.guard_tags)
            sec_tokens.extend(c.lower() for c in case.cwe)
            sec_tokens.extend(
                tokenize_code(
                    f"{case.cve_desc[:400]}\n{case.commit_message[:250]}",
                    max_tokens=100,
                )
            )
            pattern_docs.append(sec_tokens)

        self._code_index.build(code_docs)
        self._pattern_index.build(pattern_docs)

    def retrieve(
        self,
        query: str,
        *,
        func_code: str = "",
        top_k: int = 3,
        exclude_commit: str | None = None,
        min_score: float | None = None,
    ) -> KnowledgeRetrieveRecord:
        """双通道混合检索 + RRF 融合 + CVE/同名去重 + 最低相关性过滤。"""
        t0 = time.perf_counter()
        query_summary = (query or "").strip()[:160] or "(func_code)"
        threshold = self.default_min_score if min_score is None else float(min_score)

        if top_k <= 0 or not self.cases:
            return KnowledgeRetrieveRecord(
                query=query_summary,
                top_k=top_k,
                hits=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        excluded_sha = (exclude_commit or "").strip().lower()

        # 通道 A：代码词面 BM25
        raw_code_q = f"{query}\n{func_code[:4000]}" if func_code else query
        code_q_tokens = tokenize_code(raw_code_q, max_tokens=160)
        code_scores = self._code_index.score(code_q_tokens)

        # 通道 B：安全危险原语 + 查询语义 BM25
        q_sinks, q_guards = extract_security_signals(func_code or query)
        sec_q_tokens = list(q_sinks) * 2 + list(q_guards) + tokenize_code(query, max_tokens=40)
        sec_scores = self._pattern_index.score(sec_q_tokens) if sec_q_tokens else {}

        # 计算各通道排名（过滤 excluded_sha）
        def _rank_map(sc_map: dict[int, float], cap: int = 80) -> tuple[dict[int, int], float]:
            ordered = sorted(
                (
                    (doc_id, sc)
                    for doc_id, sc in sc_map.items()
                    if sc > 0 and (not excluded_sha or self.cases[doc_id].commit_id != excluded_sha)
                ),
                key=lambda x: x[1],
                reverse=True,
            )[:cap]
            max_sc = ordered[0][1] if ordered else 1.0
            return {doc_id: rank for rank, (doc_id, _) in enumerate(ordered, 1)}, max_sc

        code_ranks, max_code_sc = _rank_map(code_scores)
        sec_ranks, max_sec_sc = _rank_map(sec_scores)

        candidate_ids = set(code_ranks) | set(sec_ranks)
        if not candidate_ids:
            return KnowledgeRetrieveRecord(
                query=query_summary,
                top_k=top_k,
                hits=[],
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # RRF 融合 + 绝对信号强度惩罚（避免仅靠极弱 1 个停用词就得到高归一化分）
        k_rrf = 30.0
        q_sink_set = set(q_sinks)
        code_q_len = max(1, len(set(code_q_tokens)))

        fused: list[tuple[int, float]] = []
        for doc_id in candidate_ids:
            rc = code_ranks.get(doc_id)
            rs = sec_ranks.get(doc_id)
            rrf = (0.65 / (k_rrf + rc) if rc else 0.0) + (0.35 / (k_rrf + rs) if rs else 0.0)
            # 绝对相关性强度（0..1）：代码 BM25 饱和强度 + 安全原语 Jaccard 覆盖
            raw_c = code_scores.get(doc_id, 0.0)
            code_strength = min(1.0, raw_c / max(8.0, 1.8 * math.sqrt(code_q_len)))
            case_sinks = set(self.cases[doc_id].sink_tags)
            sink_overlap = (
                len(q_sink_set & case_sinks) / max(1, len(q_sink_set | case_sinks))
                if (q_sink_set and case_sinks)
                else 0.0
            )
            rel_code = raw_c / max_code_sc if max_code_sc > 0 else 0.0
            rel_sec = sec_scores.get(doc_id, 0.0) / max_sec_sc if max_sec_sc > 0 else 0.0

            # 综合归一化分数：兼顾相对排序（RRF）与绝对匹配强度
            rrf_norm = rrf / (1.0 / (k_rrf + 1.0))
            norm_score = (
                0.40 * rrf_norm
                + 0.35 * code_strength * max(0.3, rel_code)
                + 0.25 * max(sink_overlap, 0.5 * rel_sec * (1.0 if q_sink_set & case_sinks else 0.0))
            )
            fused.append((doc_id, min(1.0, norm_score)))

        fused.sort(key=lambda x: x[1], reverse=True)

        # 可选：挂载向量嵌入提供者时对 top-15 做混合重排
        top_pool = fused[: max(top_k * 5, 15)]
        if self.embedding_provider is not None and top_pool:
            cand_texts = [
                f"{self.cases[d].pattern_card}\n{self.cases[d].fix_pattern[:400]}"
                for d, _ in top_pool
            ]
            vecs = self.embedding_provider.embed_texts([raw_code_q[:1200], *cand_texts])
            q_vec, doc_vecs = vecs[0], vecs[1:]
            reranked = [
                (doc_id, 0.55 * sc + 0.45 * max(0.0, _cosine(q_vec, d_vec)))
                for (doc_id, sc), d_vec in zip(top_pool, doc_vecs)
            ]
            top_pool = sorted(reranked, key=lambda x: x[1], reverse=True)

        # 反劫持去重 + 相关性阈值过滤
        hits: list[KnowledgeHit] = []
        seen_cves: set[str] = set()
        func_name_counts: Counter[str] = Counter()

        for doc_id, score in top_pool:
            if score < threshold:
                continue
            c = self.cases[doc_id]
            if c.cve and c.cve in seen_cves:
                continue
            fn_key = (c.func_name or "").strip().lower()
            if fn_key and func_name_counts[fn_key] >= self.max_same_func:
                continue

            if c.cve:
                seen_cves.add(c.cve)
            if fn_key:
                func_name_counts[fn_key] += 1

            hits.append(
                KnowledgeHit(
                    case_id=c.case_id,
                    cve=c.cve,
                    cwe=list(c.cwe),
                    project=c.project or None,
                    score=round(float(score), 4),
                    cve_desc=c.cve_desc[:350],
                    commit_message=c.commit_message[:200],
                    fix_pattern=c.fix_pattern,
                    pattern_card=c.pattern_card,
                    sink_tags=list(c.sink_tags),
                    guard_tags=list(c.guard_tags),
                    vuln_snippet=c.vuln_func[:400],
                    fixed_snippet=c.fixed_func[:400],
                )
            )
            if len(hits) >= top_k:
                break

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return KnowledgeRetrieveRecord(
            query=query_summary,
            top_k=top_k,
            hits=hits,
            latency_ms=elapsed_ms,
        )
