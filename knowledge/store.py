"""领域知识库数据加载、成对 CVE 结构化模式卡加工与安全信号提取。

红线约束：
- 仅使用课程提供的历史案例数据集（``data/knowledge.jsonl``），绝不引入外部安全知识；
- 历史案例集允许成对使用（漏洞版 + 修复版），本模块自动提取：
  1. 最小修复差异摘要（``fix_pattern``）；
  2. 跨项目通用的 C/C++ 危险原语（``sink_tags``）与新增守卫模式（``guard_tags``）；
  3. 精炼结构化漏洞模式卡（``pattern_card``），缓解纯词面检索的“跨项目失灵”与“同名旧洞劫持”。
"""

from __future__ import annotations

import difflib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from repo.loader import extract_function_name

__all__ = [
    "CVECaseRecord",
    "tokenize_code",
    "extract_fix_pattern",
    "extract_security_signals",
    "build_pattern_card",
    "load_cve_cases",
]

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+")

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
        "with",
        "from",
        "that",
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

#: 跨项目 C/C++ 危险操作原语（Sink）模式表，用于构建语义检索通道。
_SINK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sink_mem_copy", re.compile(r"\b(?:memcpy|memmove|memset|strcpy|strncpy|strcat|sprintf|snprintf|bcopy)\b")),
    ("sink_alloc", re.compile(r"\b(?:malloc|calloc|realloc|alloca|Acquire\w*Memory|gf_malloc|mrb_malloc|av_malloc)\b")),
    ("sink_free_lifecycle", re.compile(r"\b(?:free|Relinquish\w*Memory|gf_free|kfree|vfree|SSL_free|X509_free)\b")),
    ("sink_array_index", re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]+\]")),
    ("sink_ptr_deref", re.compile(r"->")),
    ("sink_narrow_cast", re.compile(r"\(\s*(?:unsigned\s+char|char|short|unsigned\s+short|int|u?int(?:8|16|32)_t|size_t|ssize_t)\s*\)")),
    ("sink_float_to_int", re.compile(r"\b(?:floor|ceil|round|CastDoubleToLong|double|float)\b")),
    ("sink_unsigned_sub", re.compile(r"(?:-=|\bsize\s*-|\blen\s*-|\bcount\s*-)")),
    ("sink_int_mul_add", re.compile(r"(?:\*\s*(?:sizeof|[A-Za-z_]\w*)|<<)")),
    ("sink_div_mod", re.compile(r"(?:/\s*[A-Za-z_(]|%\s*[A-Za-z_(])")),
    ("sink_bitstream", re.compile(r"\b(?:gf_bs_|ReadBlob|GetBits|bitstream|bs_read)\w*", re.IGNORECASE)),
    ("sink_tls_ssl", re.compile(r"\b(?:SSL_|X509_|TLS|EVP_|BIO_|ctx)\w*")),
    ("sink_unreachable_assert", re.compile(r"\b(?:llvm_unreachable|assert|abort|BUG_ON|CHECK|OP_REQUIRES)\b")),
    ("sink_tensor_shape", re.compile(r"\b(?:Tensor|TensorShape|dim_size|NumElements|OpKernelContext|flat)\b")),
    ("sink_session_auth", re.compile(r"\b(?:session|auth|token|cookie|password|user_id|uid|permission|chmod|mkdir)\b", re.IGNORECASE)),
    ("sink_compiler_stack", re.compile(r"\b(?:codegen|cursp|push|pop_n|genop|mrb_ary|OP_HASH|opcode)\b")),
)

#: 历史补丁 `+` 行中常见的防御守卫（Guard）模式表。
_GUARD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("guard_null_check", re.compile(r"!\s*[A-Za-z_]\w*|==\s*(?:NULL|nullptr|0)\b|!=\s*(?:NULL|nullptr)\b")),
    ("guard_lower_bound", re.compile(r"<\s*0\b|<=\s*0\b")),
    ("guard_upper_bound", re.compile(r"(?:>=|>)\s*(?:[A-Za-z_]\w*|[0-9]+)")),
    ("guard_overflow_check", re.compile(r"\b(?:overflow|INT_MAX|UINT_MAX|SIZE_MAX|MAX_|CastDoubleToLong|SafeMul|SafeAdd)\b", re.IGNORECASE)),
    ("guard_safe_cast", re.compile(r"\(\s*(?:ssize_t|size_t|int64_t|uint64_t|ptrdiff_t)\s*\)")),
    ("guard_align_sync", re.compile(r"\b(?:align|sync|flush|reset)\b", re.IGNORECASE)),
    ("guard_ssl_verify", re.compile(r"\b(?:X509_VERIFY|set1_host|SSL_CTX_set_verify|VERIFY_PEER)\b")),
    ("guard_break_return", re.compile(r"\b(?:return|break|goto)\b")),
    ("guard_cleanup_free", re.compile(r"\b(?:free|Relinquish|unref|release|close)\b", re.IGNORECASE)),
)


@dataclass(slots=True)
class CVECaseRecord:
    """一条由成对历史样本（漏洞版 + 修复版）聚合而成的 CVE 领域知识案例。"""

    case_id: str
    commit_id: str
    cve: str | None
    cwe: list[str]
    project: str
    func_name: str | None
    cve_desc: str
    commit_message: str
    vuln_func: str
    fixed_func: str
    fix_pattern: str
    sink_tags: list[str]
    guard_tags: list[str]
    pattern_card: str


def tokenize_code(text: str, *, max_tokens: int = 400) -> list[str]:
    """面向 C/C++ 代码与安全描述的混合分词器。"""
    if not text:
        return []
    tokens: list[str] = []
    for raw in _IDENT_RE.findall(text):
        lower = raw.lower()
        if len(lower) >= 2 and lower not in _STOP_TOKENS:
            tokens.append(lower)
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
    body = [line for line in diff[2:] if line]
    if not body:
        return "(无文本行差异)"
    text = "\n".join(body)
    if len(text) > max_chars:
        text = text[: max_chars - 24] + "\n...[差异摘要截断]"
    return text


def extract_security_signals(
    code: str,
    *,
    diff_text: str = "",
) -> tuple[list[str], list[str]]:
    """确定性提取 C/C++ 代码中的危险原语（sink_tags）与补丁新增守卫（guard_tags）。"""
    snippet = code[:6000] if code else ""
    sinks = [tag for tag, pat in _SINK_PATTERNS if pat.search(snippet)]

    added_lines = "\n".join(
        line[1:]
        for line in (diff_text or "").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    guard_source = added_lines if added_lines else snippet
    guards = [tag for tag, pat in _GUARD_PATTERNS if pat.search(guard_source)]
    return sinks, guards


def build_pattern_card(
    *,
    cwe: list[str],
    cve_desc: str,
    sink_tags: list[str],
    guard_tags: list[str],
) -> str:
    """生成精炼结构化漏洞模式卡，减少整段历史源码带来的上下文噪音。"""
    cwe_part = ", ".join(cwe) if cwe else "未标注 CWE"
    sink_part = ", ".join(sink_tags[:5]) if sink_tags else "通用逻辑"
    guard_part = ", ".join(guard_tags[:5]) if guard_tags else "逻辑重写"
    desc_short = (cve_desc or "").strip().splitlines()[0][:180] if cve_desc else ""
    lines = [
        f"类型=[{cwe_part}] | 危险原语=[{sink_part}] | 修复守卫=[{guard_part}]",
    ]
    if desc_short:
        lines.append(f"机理摘要：{desc_short}")
    return "\n".join(lines)


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
    """读取课程 ``knowledge.jsonl`` 并按成对样本聚合为带模式卡的 ``CVECaseRecord``。"""
    src = Path(source)
    rows = list(_iter_raw_rows(src, zip_member=zip_member))
    cases: list[CVECaseRecord] = []

    i = 0
    n = len(rows)
    while i < n:
        r1 = rows[i]
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
        fix_pattern = extract_fix_pattern(vuln_func, fixed_func)
        sink_tags, guard_tags = extract_security_signals(vuln_func, diff_text=fix_pattern)
        pattern_card = build_pattern_card(
            cwe=cwe_list,
            cve_desc=cve_desc,
            sink_tags=sink_tags,
            guard_tags=guard_tags,
        )
        func_name = extract_function_name(vuln_func or fixed_func)
        case_id = f"{cve or 'CASE'}:{meta.get('idx', len(cases))}"

        cases.append(
            CVECaseRecord(
                case_id=case_id,
                commit_id=commit_id,
                cve=str(cve) if cve else None,
                cwe=cwe_list,
                project=project,
                func_name=func_name,
                cve_desc=cve_desc,
                commit_message=commit_msg,
                vuln_func=vuln_func,
                fixed_func=fixed_func,
                fix_pattern=fix_pattern,
                sink_tags=sink_tags,
                guard_tags=guard_tags,
                pattern_card=pattern_card,
            )
        )

    return cases
