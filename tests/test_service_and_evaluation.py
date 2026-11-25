"""DetectionService 端到端成对隔离集成测试与成对评测指标测试。"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import (
    AgentConfig,
    DetectionService,
    RawSample,
    TaskServiceProtocol,
    VulnerabilityDetectionAgent,
    compute_grade,
    compute_paired_metrics,
)
from experiments.run_experiment import verify_with_official_evaluator
from repo import RepoFS, SampleLoader


class ContextAwareFakeModel(BaseChatModel):
    """一个确定性模拟 LLM：调用 read_file 查看仓库后，根据是否包含边界检查给出 0/1。"""

    @property
    def _llm_type(self) -> str:
        return "context-aware-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            # 第一轮自主调用 read_file 检查外部辅助函数
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="需要核查 src/guard.c 中的边界校验实现。",
                            tool_calls=[
                                {
                                    "name": "read_file",
                                    "args": {"path": "src/guard.c", "offset": 1, "limit": 50},
                                    "id": "call_guard",
                                }
                            ],
                        )
                    )
                ]
            )

        # 读取到工作区内容：若工作区快照中有 "if (n > MAX)" 则判 0，否则判 1
        tool_out = str(tool_msgs[-1].content)
        has_guard = "if (n > MAX)" in tool_out
        verdict = {
            "prediction": 0 if has_guard else 1,
            "confidence": "high",
            "cwe_candidates": [] if has_guard else ["CWE-787"],
            "reason": "仓库辅助函数已做上界检查" if has_guard else "仓库辅助函数缺少上界检查",
            "evidence_cited": ["src/guard.c:2"],
            "guard_assessment": "已检查" if has_guard else "缺失检查",
        }
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=json.dumps(verdict, ensure_ascii=False))
                )
            ]
        )


class DictParentResolver:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def resolve_parent(self, owner: str, repo: str, commit_sha: str) -> str:
        return self.mapping[commit_sha]


def _make_tar(dest: Path, root_name: str, files: dict[str, bytes]) -> None:
    with tarfile.open(dest, "w:gz") as tf:
        for rel, content in files.items():
            info = tarfile.TarInfo(name=f"{root_name}/{rel}")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


def test_detection_service_end_to_end_pair_isolation_yields_pc(tmp_path):
    """端到端测试：同一函数体，仅靠版本隔离的外部仓库辅助函数区分漏洞版与修复版。"""
    commit_sha = "1234567890abcdef1234567890abcdef12345678"
    parent_sha = "fedcba0987654321fedcba0987654321fedcba09"

    def fake_dl(repo_ref, tar_path: Path):
        if repo_ref.allowed_sha == parent_sha:
            code = b"int copy_helper(char *d, char *s, int n) {\n  memcpy(d, s, n);\n}\n"
        else:
            code = b"int copy_helper(char *d, char *s, int n) {\n  if (n > MAX) return -1;\n  memcpy(d, s, n);\n}\n"
        _make_tar(tar_path, f"proj-{repo_ref.allowed_sha}", {"src/guard.c": code})

    loader = SampleLoader(parent_resolver=DictParentResolver({commit_sha: parent_sha}))
    repofs = RepoFS(cache_dir=tmp_path / "snaps", downloader=fake_dl)
    agent = VulnerabilityDetectionAgent(model=ContextAwareFakeModel())
    service = DetectionService(loader=loader, detector=agent, snapshot_provider=repofs)
    assert isinstance(service, TaskServiceProtocol)

    url = f"https://github.com/demo/proj/commit/{commit_sha}"
    samples = [
        RawSample(idx=194963, commit_url=url, func="int entry(char *d, char *s, int n) { return copy_helper(d, s, n); }", target=1),
        RawSample(idx=217569, commit_url=url, func="int entry(char *d, char *s, int n) { return copy_helper(d, s, n); }", target=0),
    ]

    result_file = tmp_path / "result.jsonl"
    report_file = tmp_path / "reports.jsonl"
    reports = service.detect_batch(
        samples,
        config=AgentConfig.improved(enable_rag=False, enable_self_check=False),
        result_path=result_file,
        report_path=report_file,
        max_workers=2,
    )

    assert len(reports) == 2
    assert reports[0].prediction == 1  # 漏洞版 -> 父提交仓库无检查 -> 判 1
    assert reports[1].prediction == 0  # 修复版 -> 提交后仓库有检查 -> 判 0

    metrics = compute_paired_metrics(samples, reports)
    assert metrics["P-C"] == 1
    assert metrics["P-V"] == 0
    assert metrics["P-B"] == 0
    assert metrics["P-R"] == 0
    assert metrics["accuracy"] == 1.0

    # 若课程 dataset.zip 存在，直接用官方 evaluation.py 校验输出格式完全兼容
    dataset_zip = Path("data/dataset.zip")
    if dataset_zip.exists():
        out = verify_with_official_evaluator(dataset_zip, result_file)
        assert "P-C（成对正确组数）：1" in out


def test_compute_grade_and_penalty_rule():
    assert compute_grade(32, 10) == ("A", False)
    assert compute_grade(32, 17) == ("B", True)  # P-R > P-C / 2 -> 降一级
    assert compute_grade(26, 5) == ("B", False)
    assert compute_grade(22, 12) == ("D", True)  # C 降为 D
    assert compute_grade(10, 8) == ("D", True)   # 最低为 D
