"""四要素检测报告持久化与可读化渲染。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schema import DetectionReport

__all__ = [
    "write_result_jsonl",
    "write_reports_jsonl",
    "format_report_markdown",
]


def write_result_jsonl(reports: Iterable[DetectionReport], path: str | Path) -> Path:
    """按 ``dataset/evaluation.py`` 约定写出 ``result.jsonl``（仅含 ``idx`` 与 ``prediction``）。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for report in reports:
            fh.write(json.dumps(report.to_result_entry(), ensure_ascii=False) + "\n")
    return out_path


def write_reports_jsonl(reports: Iterable[DetectionReport], path: str | Path) -> Path:
    """写出包含完整四要素（含分析过程记录）的报告 JSONL 文件。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for report in reports:
            fh.write(report.model_dump_json() + "\n")
    return out_path


def format_report_markdown(report: DetectionReport) -> str:
    """把四要素检测报告渲染为结构化 Markdown（便于 UI 展示或文档归档）。"""
    trace = report.analysis_record
    lines = [
        f"## 漏洞检测报告（idx={report.idx}）",
        "",
        f"- **检测函数 idx**：`{report.idx}`",
        f"- **检测结果**：**{report.label_text}** (`prediction={report.prediction}`, confidence=`{report.confidence}`)",
        f"- **候选 CWE**：{', '.join(report.cwe_candidates) if report.cwe_candidates else '无'}",
        f"- **判定理由**：{report.reason}",
    ]
    if report.guard_assessment:
        lines.append(f"- **防御检查核验**：{report.guard_assessment}")
    if report.evidence_cited:
        lines.append("- **关键证据引用**：")
        for ev in report.evidence_cited:
            lines.append(f"  - {ev}")

    lines.extend(["", "### 分析记录（系统自动采集）", ""])
    lines.append("#### 1. 历史 CVE 知识库检索记录")
    if not trace.knowledge_records:
        lines.append("- 未启用或未执行知识库检索。")
    else:
        for rec in trace.knowledge_records:
            lines.append(
                f"- 查询：`{rec.query}`（top_k={rec.top_k}，耗时 {rec.latency_ms:.1f}ms，命中 {len(rec.hits)} 条）"
            )
            for hit in rec.hits:
                cwe = "/".join(hit.cwe) if hit.cwe else "N/A"
                lines.append(
                    f"  - `{hit.cve or hit.case_id}` [{cwe}] ({hit.project or 'unknown'}, score={hit.score:.2f})"
                )

    lines.extend(["", "#### 2. 代码仓库按需读取记录"])
    if not trace.repo_records:
        lines.append("- 未调用代码仓库工具。")
    else:
        lines.append(
            f"- 共调用 {trace.tool_calls_count} 次，累计读取 {trace.total_read_chars} 字符，"
            f"终止原因：`{trace.stop_reason}`"
        )
        for rec in trace.repo_records:
            status = "OK" if rec.ok else "ERR"
            lines.append(
                f"  - [step {rec.step}] `{rec.tool_name}` ({status}, {rec.chars_returned} chars): {rec.summary}"
            )

    if trace.evidence_summaries:
        lines.extend(["", "#### 3. 证据摘要压缩子代理提纯记录"])
        for ev in trace.evidence_summaries:
            lines.append(
                f"- `{ev.tool_name}` ({ev.target or '-'}) [{ev.guard_status}] "
                f"{ev.raw_chars} -> {ev.compacted_chars} chars"
            )
            for fact in ev.verified_facts[:3]:
                lines.append(f"  - {fact}")

    return "\n".join(lines) + "\n"
