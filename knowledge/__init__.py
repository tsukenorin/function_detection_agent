"""领域知识库 RAG 包（基于课程历史 CVE 成对案例数据集）。"""

from .retriever import CVEKnowledgeRetriever, EmbeddingProvider
from .store import (
    CVECaseRecord,
    build_pattern_card,
    extract_fix_pattern,
    extract_security_signals,
    load_cve_cases,
    tokenize_code,
)

__all__ = [
    "CVECaseRecord",
    "CVEKnowledgeRetriever",
    "EmbeddingProvider",
    "build_pattern_card",
    "extract_fix_pattern",
    "extract_security_signals",
    "load_cve_cases",
    "tokenize_code",
]
