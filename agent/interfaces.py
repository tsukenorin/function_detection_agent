"""项目核心接口协议（Protocol）定义。

各层之间只依赖本模块的抽象协议，不强耦合具体实现：

- ``KnowledgeRetriever`` —— 领域知识库 RAG 检索接口（支持 BM25/混合检索/向量库替换）
- ``ParentCommitResolver`` —— 修复提交的父提交解析接口（支持 GitHub API / 离线缓存替换）
- ``SnapshotProvider`` —— 版本隔离的代码仓库快照提供接口（RepoFS）
- ``SampleLoaderProtocol`` —— 样本加载与敏感字段裁剪接口
- ``VulnerabilityDetector`` —— Agent 编排层核心检测接口
- ``TaskServiceProtocol`` —— 面向 UI / HTTP API / 批处理的应用服务接口
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from tool.workspace import Workspace

from .schema import (
    AgentConfig,
    AgentEvent,
    AgentInput,
    DetectionReport,
    EvidenceSummary,
    KnowledgeRetrieveRecord,
    RawSample,
    RepoRef,
)

__all__ = [
    "EventCallback",
    "KnowledgeRetriever",
    "ParentCommitResolver",
    "SnapshotProvider",
    "SampleLoaderProtocol",
    "EvidenceCompactorProtocol",
    "VulnerabilityDetector",
    "TaskServiceProtocol",
]

EventCallback = Callable[[AgentEvent], None]


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """领域知识库检索器协议（仅使用课程提供的历史 CVE 数据集）。"""

    def retrieve(
        self,
        query: str,
        *,
        func_code: str = "",
        top_k: int = 3,
        exclude_commit: str | None = None,
    ) -> KnowledgeRetrieveRecord:
        """检索与目标函数最相似的历史 CVE 案例并返回带审计信息的记录。"""
        ...


@runtime_checkable
class ParentCommitResolver(Protocol):
    """父提交 SHA 解析器协议（用于 target=1 样本定位修复前父提交 C^）。"""

    def resolve_parent(self, owner: str, repo: str, commit_sha: str) -> str:
        """返回 ``commit_sha`` 的第一个父提交完整 SHA。"""
        ...


@runtime_checkable
class SnapshotProvider(Protocol):
    """版本隔离仓库快照提供者协议（RepoFS 层）。"""

    def get_workspace(self, repo_ref: RepoRef) -> Workspace:
        """返回严格绑定到 ``repo_ref.allowed_sha`` 的只读 ``Workspace``。"""
        ...


@runtime_checkable
class SampleLoaderProtocol(Protocol):
    """样本加载器协议：隔离边界的执行者。"""

    def load(self, raw: RawSample) -> tuple[AgentInput, RepoRef]:
        """校验样本、解析允许访问的版本 SHA，并裁剪掉 ``target`` 后返回。"""
        ...


@runtime_checkable
class EvidenceCompactorProtocol(Protocol):
    """证据摘要压缩子代理协议：将大段仓库工具原始输出提纯为高密度结构化事实。"""

    def compact(
        self,
        *,
        function_name: str,
        audit_focus: str,
        tool_name: str,
        tool_args: dict[str, object],
        raw_output: str,
    ) -> EvidenceSummary:
        """把单次仓库工具原始输出压缩为带精确行号的 ``EvidenceSummary``。"""
        ...


@runtime_checkable
class VulnerabilityDetector(Protocol):
    """Agent 编排层核心检测接口。"""

    def detect(
        self,
        agent_input: AgentInput,
        *,
        workspace: Workspace | None = None,
        config: AgentConfig | None = None,
        on_event: EventCallback | None = None,
    ) -> DetectionReport:
        """对单个裁剪后的函数样本执行漏洞检测，返回四要素报告。"""
        ...


@runtime_checkable
class TaskServiceProtocol(Protocol):
    """应用任务服务协议（供图形化 UI / FastAPI + SSE / CLI 批处理调用）。"""

    def detect_single(
        self,
        sample: RawSample,
        *,
        config: AgentConfig | None = None,
        on_event: EventCallback | None = None,
    ) -> DetectionReport:
        """执行单样本检测并返回完整四要素报告。"""
        ...

    def detect_batch(
        self,
        samples: Iterable[RawSample],
        *,
        config: AgentConfig | None = None,
        result_path: str | Path | None = None,
        report_path: str | Path | None = None,
        on_event: EventCallback | None = None,
    ) -> list[DetectionReport]:
        """批量检测多个样本（严格逐样本隔离），并可选落盘 result.jsonl 与完整报告。"""
        ...
