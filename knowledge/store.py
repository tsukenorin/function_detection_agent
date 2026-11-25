"""领域知识库数据加载、成对 CVE 加工与代码分词工具。

红线约束：
- 仅使用课程提供的历史案例数据集（``data/knowledge.jsonl``），绝不引入外部安全知识；
- 历史案例集允许成对使用（漏洞版 + 修复版），本模块自动提取最小修复差异摘要
  （``fix_pattern``），帮助 Agent 理解真实 C/C++ 漏洞与有效防御检查之间的差别。
"""

from __future__ import annotations

import difflib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "CVECaseRecord",
    "tokenize_code",
    "extract_fix_pattern",
    "load_cve_cases",
]

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+")

#: 高频无区分度的 C/C++ 语法停用词，避免淹没真正相关的 API 与领域符号。
_STOP_TOKENS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "goto",
        "void",
        "int",
        "char",
        "short",
        "long",
        "float",
        "double",
        "unsigned",
        "signed",
        "const",
        "static",
        "extern",
        "inline",
        "volatile",
        "register",
        "struct",
        "union",
        "enum",
        "typedef",
        "auto",
        "bool",
        "true",
        "false",
        "null",
        "nullptr",
        "this",
        "class",
        "public",
        "private",
        "protected",
        "namespace",
        "using",
        "template",
        "typename",
        "new",
        "delete",
        "try",
        "catch",
        "throw",
        "define",
        "include",
        "ifdef",
        "ifndef",
        "endif",
        "undef",
        "pragma",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "not",
        "can",
        "has",
        "have",
        "when",
        "into",
        "via",
        "use",
    }
)


@dataclass(slots=True)
class CVECaseRecord:
    """一条由成对历史样本（漏洞版 + 修复版）聚合而成的 CVE 领域知识案例。"""

    case_id: str
    commit_id: str
    cve: str | None
    cwe: list[str]
    project: str
    cve_desc: str
    commit_message: str
    vuln_func: str
    fixed_func: str
    fix_pattern: str


def tokenize_code(text: str, *, max_tokens: int = 400) -> list[str]:
    """面向 C/C++ 代码与安全描述的混合分词器。

    同时保留完整标识符（如 ``GetEXIFProperty`` -> ``getexifproperty``）与
    拆分后的子词（``exif``, ``property``），提升跨项目 API 名称的检索召回率。
    """
    if not text:
        return []
    tokens: list[str] = []
    for raw in _IDENT_RE.findall(text):
        lower = raw.lower()
        if len(lower) >= 2 and lower not in _STOP_TOKENS:
            tokens.append(lower)
        # 拆分 snake_case 与 camelCase 子词
        if "_" in raw or any(c.isupper() for c in raw[1:]):
            for part in raw.split("_"):
                if not part:
                    continue
                for sub in _CAMEL_RE.findall(part):
                    sl = sub.lower()
                    if len(sl) >= 2 and sl != lower and sl not in _STOP_TOKENS:
                        tokens.append(sl)
        if len(tokens) >= max_tokens:
            break
    return tokens[:max_tokens]


def extract_fix_pattern(
    vuln_func: str,
    fixed_func: str,
    *,
    context_lines: int = 2,
    max_chars: int = 700,
) -> str:
    """从历史成对 CVE 样本中提取最小修复差异摘要（仅用于知识库案例！）。"""
    if not vuln_func or not fixed_func:
        return ""
    vuln_lines = vuln_func.splitlines()
    fixed_lines = fixed_func.splitlines()
    diff = list(
        difflib.unified_diff(
            vuln_lines,
            fixed_lines,
            fromfile="vuln",
            tofile="fixed",
            n=context_lines,
            lineterm="",
        )
    )
    # 跳过前两行文件名头，只保留 hunk
    body = [line for line in diff[2:] if line]
    if not body:
        return "(无文本行差异)"
    text = "\n".join(body)
    if len(text) > max_chars:
        text = text[: max_chars - 24] + "\n...[差异摘要截断]"
    return text


def _iter_raw_rows(source: Path, zip_member: str = "dataset/knowledge.jsonl") -> Iterable[dict]:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            text = zf.read(zip_member).decode("utf-8-sig")
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)
    else:
        with source.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_cve_cases(
    source: str | Path,
    *,
    zip_member: str = "dataset/knowledge.jsonl",
) -> list[CVECaseRecord]:
    """读取课程 ``knowledge.jsonl`` 并按成对样本聚合为 ``CVECaseRecord`` 列表。"""
    src = Path(source)
    rows = list(_iter_raw_rows(src, zip_member=zip_member))
    cases: list[CVECaseRecord] = []

    i = 0
    n = len(rows)
    while i < n:
        r1 = rows[i]
        # 优先按相邻 (1, 0) / (0, 1) 配对（knowledge.jsonl 默认成对排列）
        if i + 1 < n and {r1.get("target"), rows[i + 1].get("target")} == {0, 1}:
            r2 = rows[i + 1]
            vuln, fixed = (r1, r2) if r1.get("target") == 1 else (r2, r1)
            i += 2
        else:
            vuln = r1 if r1.get("target") == 1 else {}
            fixed = r1 if r1.get("target") == 0 else {}
            i += 1

        meta = vuln if vuln else fixed
        commit_id = str(meta.get("commit_id") or "").strip().lower()
        cve = meta.get("cve") or None
        raw_cwe = meta.get("cwe") or []
        cwe_list = [str(c) for c in raw_cwe if c] if isinstance(raw_cwe, list) else [str(raw_cwe)]
        project = str(meta.get("project") or "").strip()
        cve_desc = str(meta.get("cve_desc") or "").strip()
        commit_msg = str(meta.get("commit_message") or "").strip()
        vuln_func = str(vuln.get("func") or "")
        fixed_func = str(fixed.get("func") or "")
        case_id = f"{cve or 'CASE'}:{meta.get('idx', len(cases))}"

        cases.append(
            CVECaseRecord(
                case_id=case_id,
                commit_id=commit_id,
                cve=str(cve) if cve else None,
                cwe=cwe_list,
                project=project,
                cve_desc=cve_desc,
                commit_message=commit_msg,
                vuln_func=vuln_func,
                fixed_func=fixed_func,
                fix_pattern=extract_fix_pattern(vuln_func, fixed_func),
            )
        )

    return cases
