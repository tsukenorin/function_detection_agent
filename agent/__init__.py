"""Agent 编排层与核心领域契约包。"""

from .interfaces import (
    EventCallback,
    KnowledgeRetriever,
    ParentCommitResolver,
    SampleLoaderProtocol,
    SnapshotProvider,
    TaskServiceProtocol,
    VulnerabilityDetector,
)
from .schema import (
    AgentConfig,
    AgentEvent,
    AgentInput,
    AgentVerdict,
    AnalysisTrace,
    DetectionReport,
    KnowledgeHit,
    KnowledgeRetrieveRecord,
    RawSample,
    RepoReadRecord,
    RepoRef,
)

__all__ = [
    "RawSample",
    "RepoRef",
    "AgentInput",
    "KnowledgeHit",
    "KnowledgeRetrieveRecord",
    "RepoReadRecord",
    "AnalysisTrace",
    "AgentVerdict",
    "DetectionReport",
    "AgentEvent",
    "AgentConfig",
    "EventCallback",
    "KnowledgeRetriever",
    "ParentCommitResolver",
    "SnapshotProvider",
    "SampleLoaderProtocol",
    "VulnerabilityDetector",
    "TaskServiceProtocol",
    "VulnerabilityDetectionAgent",
    "write_result_jsonl",
    "write_reports_jsonl",
    "format_report_markdown",
]


def __getattr__(name: str):
    if name in {"VulnerabilityDetectionAgent", "AgentState", "parse_verdict_from_text"}:
        from . import agent_loop

        return getattr(agent_loop, name)
    if name in {"write_result_jsonl", "write_reports_jsonl", "format_report_markdown"}:
        from . import report

        return getattr(report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
