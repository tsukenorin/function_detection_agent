"""课程数据集 Baseline vs Improved 对比实验运行器。

严格遵循 PPT 要求与三条红线：
1. 从 ``data/dataset.zip`` 读取 ``dataset/test.jsonl`` 成对测试样本；
2. 仅使用 ``data/knowledge.jsonl`` 构建领域知识库 RAG（零外部安全知识）；
3. 通过 ``SampleLoader`` + ``RepoFS`` 严格隔离每一对中的漏洞版（父提交 C^）与修复版（提交后 C）；
4. 对比以下两种（或三种）实验配置：
   - **Baseline**：孤立函数直接检测（关闭 RAG、关闭仓库上下文工具、关闭防御反事实自检）；
   - **Improved**：历史 CVE 成对 RAG + 按需只读代码仓库工具挖掘 + 防御检查反事实自检；
5. 输出与 ``dataset/evaluation.py`` 完全兼容的 ``result_*.jsonl`` 与详细四要素报告，
   并调用官方 ``dataset/evaluation.py`` 核验 P-C / P-V / P-B / P-R 指标。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

from agent import (
    AgentConfig,
    AgentEvent,
    DetectionService,
    RawSample,
    VulnerabilityDetectionAgent,
    compute_paired_metrics,
)
from knowledge import CVEKnowledgeRetriever
from repo import GitHubParentResolver, RepoFS, SampleLoader, parse_commit_url

#: 超大型单仓（整仓 tarball > 250MB），在中小规模快速实验中可选过滤，避免长时间下载。
HUGE_REPOS = frozenset({"torvalds/linux", "tensorflow/tensorflow"})


def select_pairs(
    all_samples: Sequence[RawSample],
    *,
    num_pairs: int = 12,
    exclude_huge_repos: bool = True,
    max_func_chars: int = 16000,
    max_per_repo: int = 3,
) -> list[RawSample]:
    """按多项目均衡策略从 150 对测试集中抽取成对样本（严格保留原始相邻配对）。"""
    selected: list[RawSample] = []
    repo_counts: dict[str, int] = {}

    for offset in range(0, len(all_samples) - 1, 2):
        if len(selected) // 2 >= num_pairs:
            break
        s1, s2 = all_samples[offset], all_samples[offset + 1]
        if {s1.target, s2.target} != {0, 1}:
            continue
        if len(s1.func) > max_func_chars or len(s2.func) > max_func_chars:
            continue
        try:
            _, owner, repo, _ = parse_commit_url(s1.commit_url)
        except Exception:
            continue
        full = f"{owner}/{repo}"
        if exclude_huge_repos and full in HUGE_REPOS:
            continue
        if repo_counts.get(full, 0) >= max_per_repo:
            continue
        repo_counts[full] = repo_counts.get(full, 0) + 1
        selected.extend([s1, s2])

    return selected


def verify_with_official_evaluator(
    dataset_zip: Path,
    result_jsonl: Path,
) -> str:
    """解压并调用官方 ``dataset/evaluation.py`` 进行第三方核验。"""
    with tempfile.TemporaryDirectory(prefix="eval_verify_") as td:
        td_path = Path(td)
        with zipfile.ZipFile(dataset_zip) as zf:
            zf.extract("dataset/evaluation.py", td_path)
            zf.extract("dataset/test.jsonl", td_path)
        proc = subprocess.run(
            [
                sys.executable,
                str(td_path / "dataset" / "evaluation.py"),
                "--dataset",
                str(td_path / "dataset" / "test.jsonl"),
                "--result",
                str(result_jsonl.resolve()),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return (proc.stdout + proc.stderr).strip()


def run_experiment(
    *,
    dataset_zip: Path = Path("data/dataset.zip"),
    knowledge_jsonl: Path = Path("data/knowledge.jsonl"),
    output_dir: Path = Path("experiments/results"),
    model_name: str = "deepseek-chat",
    num_pairs: int = 12,
    exclude_huge_repos: bool = True,
    max_per_repo: int = 3,
    max_workers: int = 2,
    enable_ablation_rag_only: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples = SampleLoader.load_raw_from_zip(dataset_zip, "dataset/test.jsonl")
    selected = select_pairs(
        all_samples,
        num_pairs=num_pairs,
        exclude_huge_repos=exclude_huge_repos,
        max_per_repo=max_per_repo,
    )

    print(
        f"[Experiment] Loaded {len(all_samples)} test rows ({len(all_samples)//2} pairs); "
        f"selected {len(selected)//2} pairs ({len(selected)} samples).",
        flush=True,
    )
    repos = sorted({parse_commit_url(s.commit_url)[1] + "/" + parse_commit_url(s.commit_url)[2] for s in selected})
    print(f"[Experiment] Covered repos ({len(repos)}): {', '.join(repos)}", flush=True)

    # 加载课程历史 CVE 领域知识库
    print(f"[Experiment] Indexing knowledge base from {knowledge_jsonl}...", flush=True)
    retriever = CVEKnowledgeRetriever.from_jsonl(knowledge_jsonl)
    print(f"[Experiment] Indexed {len(retriever.cases)} paired CVE cases.", flush=True)

    loader = SampleLoader(parent_resolver=GitHubParentResolver())
    repofs = RepoFS(cache_dir=".cache/snapshots")
    agent = VulnerabilityDetectionAgent(retriever=retriever)
    service = DetectionService(
        loader=loader,
        detector=agent,
        snapshot_provider=repofs,
    )

    def _log_event(ev: AgentEvent) -> None:
        if ev.event_type in {"knowledge_retrieved", "tool_result", "verdict"}:
            if ev.event_type == "knowledge_retrieved":
                hits = [h.get("cve") for h in ev.payload.get("hits", [])]
                print(f"  [idx={ev.idx}] RAG hits={hits}", flush=True)
            elif ev.event_type == "tool_result":
                print(
                    f"  [idx={ev.idx} step={ev.step}] TOOL {ev.payload.get('tool_name')} "
                    f"ok={ev.payload.get('ok')} -> {ev.payload.get('summary')}",
                    flush=True,
                )
            elif ev.event_type == "verdict":
                print(
                    f"  [idx={ev.idx}] VERDICT pred={ev.payload.get('prediction')} "
                    f"conf={ev.payload.get('confidence')} cwe={ev.payload.get('cwe_candidates')}",
                    flush=True,
                )

    runs: list[tuple[str, AgentConfig]] = [
        ("baseline", AgentConfig.baseline(model_name=model_name)),
    ]
    if enable_ablation_rag_only:
        runs.append(
            (
                "improved_rag_check",
                AgentConfig.improved(
                    model_name=model_name,
                    enable_repo_tools=False,
                    max_tool_calls=0,
                ),
            )
        )
    runs.append(
        (
            "improved",
            AgentConfig.improved(
                model_name=model_name,
                max_steps=5,
                max_tool_calls=5,
                max_read_chars=18000,
                top_k_knowledge=3,
            ),
        )
    )

    summary: dict = {
        "model_name": model_name,
        "num_pairs": len(selected) // 2,
        "num_samples": len(selected),
        "repos": repos,
        "selected_indices": [s.idx for s in selected],
        "runs": {},
    }

    for run_name, cfg in runs:
        print(f"\n=== Running [{run_name}] (mode={cfg.mode}, rag={cfg.enable_rag}, repo_tools={cfg.enable_repo_tools}, self_check={cfg.enable_self_check}) ===", flush=True)
        res_path = output_dir / f"result_{run_name}.jsonl"
        rep_path = output_dir / f"reports_{run_name}.jsonl"
        ckpt_path = output_dir / f"ckpt_{run_name}.jsonl"

        reports = service.detect_batch(
            selected,
            config=cfg,
            result_path=res_path,
            report_path=rep_path,
            checkpoint_path=ckpt_path,
            on_event=_log_event,
            max_workers=max_workers,
        )
        metrics = compute_paired_metrics(selected, reports)
        official_out = verify_with_official_evaluator(dataset_zip, res_path)
        print(f"\n[{run_name}] Metrics: {json.dumps(metrics, ensure_ascii=False)}", flush=True)
        print(f"[{run_name}] Official evaluation.py:\n{official_out}\n", flush=True)

        # 逐对详细对比（便于案例分析）
        rep_by_idx = {r.idx: r for r in reports}
        pair_details = []
        for offset in range(0, len(selected), 2):
            a, b = selected[offset], selected[offset + 1]
            v, f = (a, b) if a.target == 1 else (b, a)
            rv, rf = rep_by_idx[v.idx], rep_by_idx[f.idx]
            pair_cat = {(1, 0): "P-C", (1, 1): "P-V", (0, 0): "P-B", (0, 1): "P-R"}[
                (rv.prediction, rf.prediction)
            ]
            pair_details.append(
                {
                    "pair_no": offset // 2 + 1,
                    "repo": rv.metadata.get("repo_name"),
                    "function": rv.metadata.get("function_name"),
                    "vuln_idx": v.idx,
                    "fixed_idx": f.idx,
                    "vuln_pred": rv.prediction,
                    "fixed_pred": rf.prediction,
                    "category": pair_cat,
                    "vuln_tools": rv.analysis_record.tool_calls_count,
                    "fixed_tools": rf.analysis_record.tool_calls_count,
                }
            )

        summary["runs"][run_name] = {
            "config": cfg.model_dump(),
            "metrics": metrics,
            "official_evaluator_output": official_out,
            "pair_details": pair_details,
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Experiment] Saved summary to {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Baseline vs Improved 成对漏洞检测实验")
    parser.add_argument("--dataset-zip", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--knowledge", type=Path, default=Path("data/knowledge.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--include-huge-repos", action="store_true")
    parser.add_argument("--max-per-repo", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ablation-rag-only", action="store_true")
    args = parser.parse_args()

    run_experiment(
        dataset_zip=args.dataset_zip,
        knowledge_jsonl=args.knowledge,
        output_dir=args.output_dir,
        model_name=args.model,
        num_pairs=args.pairs,
        exclude_huge_repos=not args.include_huge_repos,
        max_per_repo=args.max_per_repo,
        max_workers=args.workers,
        enable_ablation_rag_only=args.ablation_rag_only,
    )


if __name__ == "__main__":
    main()
