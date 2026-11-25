"""领域知识库 RAG 检索与防泄露机制的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.interfaces import KnowledgeRetriever
from knowledge import (
    CVEKnowledgeRetriever,
    extract_fix_pattern,
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


def test_tokenize_code_splits_camel_and_snake():
    toks = tokenize_code("static void GetEXIFProperty(size_t buffer_len) { memcpy(a, b, n); }")
    assert "getexifproperty" in toks
    assert "exif" in toks
    assert "buffer_len" in toks
    assert "memcpy" in toks
    assert "static" not in toks


def test_extract_fix_pattern_highlights_added_guard():
    v = "int f(int *p) {\n  return *p;\n}"
    f = "int f(int *p) {\n  if (!p) return 0;\n  return *p;\n}"
    diff = extract_fix_pattern(v, f)
    assert "+  if (!p) return 0;" in diff


def test_retriever_protocol_ranking_and_commit_exclusion(mini_knowledge_jsonl):
    retriever = CVEKnowledgeRetriever.from_jsonl(mini_knowledge_jsonl)
    assert isinstance(retriever, KnowledgeRetriever)
    assert len(retriever.cases) == 2

    rec = retriever.retrieve(
        "ImageMagick GetEXIFProperty",
        func_code="void parse_exif(char *exif, size_t offset) { memcpy(out, exif + offset, 32); }",
        top_k=2,
    )
    assert len(rec.hits) == 1
    assert rec.hits[0].cve == "CVE-2019-0001"
    assert "+  if (offset + 64 > len)" in rec.hits[0].fix_pattern

    # 当两个案例均有词项命中时，相关度更高的排在首位
    rec_both = retriever.retrieve(
        "ImageMagick GetEXIFProperty ssl",
        func_code="void parse_exif(char *exif, size_t offset) { memcpy(out, exif + offset, 32); }",
        top_k=2,
    )
    assert len(rec_both.hits) == 2
    assert rec_both.hits[0].cve == "CVE-2019-0001"

    # 防泄露：排除指定 commit_id
    rec_excl = retriever.retrieve(
        "ImageMagick GetEXIFProperty",
        func_code="memcpy(out, exif + offset, 32); ssl_handshake(s);",
        top_k=2,
        exclude_commit="1111111111111111111111111111111111111111",
    )
    assert all(h.cve != "CVE-2019-0001" for h in rec_excl.hits)


def test_real_knowledge_dataset_smoke():
    real_path = Path("data/knowledge.jsonl")
    if not real_path.exists():
        pytest.skip("data/knowledge.jsonl 不存在")

    cases = load_cve_cases(real_path)
    assert len(cases) == 3789
    retriever = CVEKnowledgeRetriever(cases)
    rec = retriever.retrieve(
        "ImageMagick GetEXIFProperty",
        func_code="static MagickBooleanType GetEXIFProperty(const Image *image) { ReadProfileByte(exif, offset); }",
        top_k=3,
    )
    assert len(rec.hits) == 3
    assert rec.hits[0].score > 0
    assert rec.hits[0].fix_pattern != ""
