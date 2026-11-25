"""应用服务层（DetectionService）与成对评测统计工具。

``DetectionService`` 是面向图形化 UI（FastAPI + SSE）、CLI 与批量评测的统一入口：
- 严格执行“SampleLoader 裁剪 -> RepoFS 单版本隔离工作区 -> Agent 编排检测”；
- 每个样本拥有完全独立的 ``Workspace`` 与状态图，支持安全的单线程或并发批量检测；
- 提供与 ``dataset/evaluation.py`` 严格一致的 P-C / P-V / P-B / P-R 与等级惩罚统计。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent.interfaces import (
    EventCallback,
    SampleLoaderProtocol,
    SnapshotProvider,
    VulnerabilityDetector,
)
from agent.report import write_reports_jsonl, write_result_jsonl
from agent.schema import AgentConfig, DetectionReport, RawSample

__all__ = [
    "DetectionService",
    "compute_paired_metrics",
    "compute_grade",
]


def compute_grade(pc: int, pr: int) -> tuple[str, bool]:
    """按 PPT 第 9 页规则计算等级（按 P-C 对数分级，P-R > P-C/2 降一级，最低 D）。"""
    tiers = ["A", "B", "C", "D"]
    if pc >= 30:
        tier_idx = 0
    elif pc >= 25:
        tier_idx = 1
    elif pc >= 20:
        tier_idx = 2
    else:
        tier_idx = 3

    penalized = pr > (pc / 2.0)
    if penalized:
        tier_idx = min(len(tiers) - 1, tier_idx + 1)
    return tiers[tier_idx], penalized


def compute_paired_metrics(
    samples: Sequence[RawSample],
    reports: Sequence[DetectionReport],
) -> dict[str, Any]:
    """与 ``dataset/evaluation.py`` 保持完全一致的成对评测统计，并附带样本级与效率指标。"""
    pred_map = {r.idx: int(r.prediction) for r in reports}
    report_map = {r.idx: r for r in reports}

    categories = {(1, 0): "P-C", (1, 1): "P-V", (0, 0): "P-B", (0, 1): "P-R"}
    counts = {"P-C": 0, "P-V": 0, "P-B": 0, "P-R": 0}
    evaluated_pairs = 0

    tp = fp = tn = fn = 0
    for sample in samples:
        if sample.target is None or sample.idx not in pred_map:
            continue
        p = pred_map[sample.idx]
        t = int(sample.target)
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1

    for offset in range(0, len(samples) - 1, 2):
        first, second = samples[offset], samples[offset + 1]
        if first.target is None or second.target is None:
            continue
        if {first.target, second.target} != {0, 1}:
            continue
        if first.idx not in pred_map or second.idx not in pred_map:
            continue
        vuln, fixed = (first, second) if first.target == 1 else (second, first)
        key = (pred_map[vuln.idx], pred_map[fixed.idx])
        counts[categories[key]] += 1
        evaluated_pairs += 1

    total_eval_samples = tp + fp + tn + fn
    acc = (tp + tn) / total_eval_samples if total_eval_samples else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    used_reports = [report_map[s.idx] for s in samples if s.idx in report_map]
    avg_tools = (
        sum(r.analysis_record.tool_calls_count for r in used_reports) / len(used_reports)
        if used_reports
        else 0.0
    )
    avg_rag_hits = (
        sum(
            sum(len(k.hits) for k in r.analysis_record.knowledge_records)
            for r in used_reports
        )
        / len(used_reports)
        if used_reports
        else 0.0
    )
    avg_latency_ms = (
        sum(float(r.metadata.get("elapsed_ms", 0.0)) for r in used_reports) / len(used_reports)
        if used_reports
        else 0.0
    )

    grade, penalized = compute_grade(counts["P-C"], counts["P-R"])
    pc_rate = counts["P-C"] / evaluated_pairs if evaluated_pairs else 0.0

    return {
        "evaluated_pairs": evaluated_pairs,
        "evaluated_samples": total_eval_samples,
        "P-C": counts["P-C"],
        "P-V": counts["P-V"],
        "P-B": counts["P-B"],
        "P-R": counts["P-R"],
        "P-C_rate": round(pc_rate, 4),
        "grade": grade,
        "penalty_applied": penalized,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "avg_tool_calls": round(avg_tools, 2),
        "avg_rag_hits": round(avg_rag_hits, 2),
        "avg_latency_ms": round(avg_latency_ms, 1),
    }


class DetectionService:
    """实现 ``TaskServiceProtocol`` 的漏洞检测应用服务。"""

    def __init__(
        self,
        *,
        loader: SampleLoaderProtocol,
        detector: VulnerabilityDetector,
        snapshot_provider: SnapshotProvider | None = None,
        default_config: AgentConfig | None = None,
    ) -> None:
        self.loader = loader
        self.detector = detector
        self.snapshot_provider = snapshot_provider
        self.default_config = default_config or AgentConfig.improved()

    def detect_single(
        self,
        sample: RawSample,
        *,
        config: AgentConfig | None = None,
        on_event: EventCallback | None = None,
    ) -> DetectionReport:
        cfg = config or self.default_config
        # 1) 隔离边界：消费 target 并产出裁剪后的 AgentInput + RepoRef
        agent_input, repo_ref = self.loader.load(sample)

        # 2) 按需构建单版本只读工作区（惰性物化，不跨样本共享）
        workspace = None
        if cfg.enable_repo_tools and self.snapshot_provider is not None:
            workspace = self.snapshot_provider.get_workspace(repo_ref)

        # 3) 执行 Agent 编排检测
        return self.detector.detect(
            agent_input,
            workspace=workspace,
            config=cfg,
            on_event=on_event,
        )

    def detect_batch(
        self,
        samples: Iterable[RawSample],
        *,
        config: AgentConfig | None = None,
        result_path: str | Path | None = None,
        report_path: str | Path | None = None,
        on_event: EventCallback | None = None,
        max_workers: int = 1,
        checkpoint_path: str | Path | None = None,
    ) -> list[DetectionReport]:
        sample_list = list(samples)
        cfg = config or self.default_config

        # 支持从断点恢复已完成样本（大批量实验防中断）
        cached_reports: dict[int, DetectionReport] = {}
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        if ckpt and ckpt.exists():
            for line in ckpt.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rep = DetectionReport.model_validate(json.loads(line))
                cached_reports[rep.idx] = rep

        results_by_pos: list[DetectionReport | None] = [None] * len(sample_list)
        pending: list[tuple[int, RawSample]] = []
        for pos, sample in enumerate(sample_list):
            if sample.idx in cached_reports:
                results_by_pos[pos] = cached_reports[sample.idx]
            else:
                pending.append((pos, sample))

        def _append_ckpt(rep: DetectionReport) -> None:
            if ckpt is None:
                return
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            with ckpt.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(rep.model_dump_json() + "\n")

        if max_workers <= 1 or len(pending) <= 1:
            for pos, sample in pending:
                rep = self.detect_single(sample, config=cfg, on_event=on_event)
                results_by_pos[pos] = rep
                _append_ckpt(rep)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {
                    pool.submit(
                        self.detect_single, sample, config=cfg, on_event=on_event
                    ): pos
                    for pos, sample in pending
                }
                for fut in as_completed(future_map):
                    pos = future_map[fut]
                    rep = fut.result()
                    results_by_pos[pos] = rep
                    _append_ckpt(rep)

        reports = [r for r in results_by_pos if r is not None]
        if result_path is not None:
            write_result_jsonl(reports, result_path)
        if report_path is not None:
            write_reports_jsonl(reports, report_path)
        return reports
