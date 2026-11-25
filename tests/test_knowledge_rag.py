"""领域知识库 RAG（双通道混合检索、结构化模式卡、去重多样性与阈值过滤）单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.interfaces import KnowledgeRetriever
from knowledge import (
    CVEKnowledgeRetriever,
    build_pattern_card,
    extract_fix_pattern,
    extract_security_signals,
    load_cve_cases,
    tokenize_code,
)


@pytest.fixture
def mini_knowledge_jsonl(tmp_path):
    rows = [
        {
            "idx": 1,
            "project": "ImageMagick",
            "commit_id": "1111111111111111111111111111111111111111",
            "commit_message": "Fix heap buffer overflow in GetEXIFProperty",
            "target": 1,
            "func": "void GetEXIFProperty(char *exif, size_t offset) {\n  memcpy(dst, exif + offset, 64);\n}",
            "cwe": ["CWE-125"],
            "cve": "CVE-2019-0001",
            "cve_desc": "Out-of-bounds read in EXIF parser when offset exceeds buffer length.",
        },
        {
            "idx": 2,
            "project": "ImageMagick",
            "commit_id": "1111111111111111111111111111111111111111",
            "commit_message": "Fix heap buffer overflow in GetEXIFProperty",
            "target": 0,
            "func": "void GetEXIFProperty(char *exif, size_t offset, size_t len) {\n  if (offset + 64 > len) return;\n  memcpy(dst, exif + offset, 64);\n}",
            "cwe": ["CWE-125"],
            "cve": "CVE-2019-0001",
            "cve_desc": "Out-of-bounds read in EXIF parser when offset exceeds buffer length.",
        },
        # 重复同 CVE 记录（测试 CVE 去重）
        {
            "idx": 11,
            "project": "ImageMagick",
            "commit_id": "1111111111111111111111111111111111111112",
            "commit_message": "Duplicate CVE entry for GetEXIFProperty",
            "target": 1,
            "func": "void GetEXIFProperty(char *exif, size_t offset) {\n  memcpy(dst, exif + offset, 32);\n}",
            "cwe": ["CWE-125"],
            "cve": "CVE-2019-0001",
            "cve_desc": "Duplicate entry for same CVE.",
        },
        {
            "idx": 12,
            "project": "ImageMagick",
            "commit_id": "1111111111111111111111111111111111111112",
            "commit_message": "Duplicate CVE entry for GetEXIFProperty",
            "target": 0,
            "func": "void GetEXIFProperty(char *exif, size_t offset, size_t len) {\n  if (offset + 32 > len) return;\n  memcpy(dst, exif + offset, 32);\n}",
            "cwe": ["CWE-125"],
            "cve": "CVE-2019-0001",
            "cve_desc": "Duplicate entry for same CVE.",
        },
        # 同名函数但不同 CVE（测试 max_same_func=1 多样性限制）
        {
            "idx": 21,
            "project": "ImageMagick",
            "commit_id": "1111111111111111111111111111111111111113",
            "commit_message": "Another old bug in GetEXIFProperty",
            "target": 1,
            "func": "void GetEXIFProperty(char *exif, size_t offset) {\n  memcpy(dst, exif + offset, 16);\n}",
            "cwe": ["CWE-787"],
            "cve": "CVE-2018-0009",
            "cve_desc": "Older bug in GetEXIFProperty.",
        },
        {
            "idx": 22,
            "project": "ImageMagick",
            "commit_id": "1111111111111111111111111111111111111113",
            "commit_message": "Another old bug in GetEXIFProperty",
            "target": 0,
            "func": "void GetEXIFProperty(char *exif, size_t offset, size_t len) {\n  if (offset > len) return;\n  memcpy(dst, exif + offset, 16);\n}",
            "cwe": ["CWE-787"],
            "cve": "CVE-2018-0009",
            "cve_desc": "Older bug in GetEXIFProperty.",
        },
        {
            "idx": 3,
            "project": "openssl",
            "commit_id": "2222222222222222222222222222222222222222",
            "commit_message": "Prevent NULL pointer dereference in ssl_handshake",
            "target": 1,
            "func": "int ssl_handshake(SSL *s) {\n  return s->session->cipher_id;\n}",
            "cwe": ["CWE-476"],
            "cve": "CVE-2019-0002",
            "cve_desc": "NULL pointer dereference in SSL handshake when session is null.",
        },
        {
            "idx": 4,
            "project": "openssl",
            "commit_id": "2222222222222222222222222222222222222222",
            "commit_message": "Prevent NULL pointer dereference in ssl_handshake",
            "target": 0,
            "func": "int ssl_handshake(SSL *s) {\n  if (!s || !s->session) return -1;\n  return s->session->cipher_id;\n}",
            "cwe": ["CWE-476"],
            "cve": "CVE-2019-0002",
            "cve_desc": "NULL pointer dereference in SSL handshake when session is null.",
        },
    ]
    path = tmp_path / "knowledge.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_tokenize_and_security_signals_and_pattern_card():
    toks = tokenize_code("static void GetEXIFProperty(size_t buffer_len) { memcpy(a, b, n); }")
    assert "getexifproperty" in toks
    assert "exif" in toks
    assert "memcpy" in toks
    assert "static" not in toks

    v = "int f(int *p) {\n  return p->val;\n}"
    f = "int f(int *p) {\n  if (!p) return 0;\n  return p->val;\n}"
    diff = extract_fix_pattern(v, f)
    assert "+  if (!p) return 0;" in diff

    sinks, guards = extract_security_signals(v, diff_text=diff)
    assert "sink_ptr_deref" in sinks
    assert "guard_null_check" in guards

    card = build_pattern_card(cwe=["CWE-476"], cve_desc="null deref", sink_tags=sinks, guard_tags=guards)
    assert "CWE-476" in card and "sink_ptr_deref" in card and "guard_null_check" in card


def test_hybrid_retriever_dedup_diversity_threshold_and_exclusion(mini_knowledge_jsonl):
    retriever = CVEKnowledgeRetriever.from_jsonl(mini_knowledge_jsonl, default_min_score=0.15, max_same_func=1)
    assert isinstance(retriever, KnowledgeRetriever)
    assert len(retriever.cases) == 4

    # 验证：CVE 去重 + 同名函数最多占 1 槽位
    rec = retriever.retrieve(
        "ImageMagick GetEXIFProperty",
        func_code="void parse_exif(char *exif, size_t offset) { memcpy(out, exif + offset, 32); }",
        top_k=3,
    )
    cves = [h.cve for h in rec.hits]
    assert len(cves) == len(set(cves))
    assert sum(1 for h in rec.hits if h.cve in {"CVE-2019-0001", "CVE-2018-0009"}) == 1
    assert rec.hits[0].pattern_card != ""

    # 验证：完全无关的查询经高阈值过滤返回空（宁缺毋滥）
    rec_noise = retriever.retrieve("totally_unrelated_xyz_123", func_code="int foo() { return 1; }", top_k=3, min_score=0.6)
    assert rec_noise.hits == []

    # 验证：防泄露排除 commit_id
    rec_excl = retriever.retrieve(
        "openssl ssl_handshake",
        func_code="int tls_check(SSL *s) { return s->session->cipher_id; }",
        top_k=3,
        exclude_commit="2222222222222222222222222222222222222222",
    )
    assert all(h.cve != "CVE-2019-0002" for h in rec_excl.hits)


def test_real_knowledge_dataset_dedup_and_cards():
    real_path = Path("data/knowledge.jsonl")
    if not real_path.exists():
        pytest.skip("data/knowledge.jsonl 不存在")

    cases = load_cve_cases(real_path)
    assert len(cases) == 3789
    retriever = CVEKnowledgeRetriever(cases)

    # 回归验证 Pair 11（ReadPCLImage）：不再重复返回同一 CVE 或全被同名旧函数占满
    rec = retriever.retrieve(
        "ImageMagick/ImageMagick ReadPCLImage",
        func_code="static Image *ReadPCLImage(const ImageInfo *image_info) { width=(size_t) floor(bounds.x2-bounds.x1+0.5); }",
        top_k=3,
    )
    assert len(rec.hits) >= 1
    cves = [h.cve for h in rec.hits if h.cve]
    assert len(cves) == len(set(cves))
    assert all(h.pattern_card for h in rec.hits)
