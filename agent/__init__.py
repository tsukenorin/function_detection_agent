"""Agent 编排层与核心领域契约包。"""

from .interfaces import (
    EventCallback,
    EvidenceCompactorProtocol,
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
    EvidenceSummary,
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
    "EvidenceSummary",
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
    "EvidenceCompactorProtocol",
    "VulnerabilityDetector",
    "TaskServiceProtocol",
    "VulnerabilityDetectionAgent",
    "EvidenceCompactorSubAgent",
    "DetectionService",
    "compute_paired_metrics",
    "compute_grade",
    "write_result_jsonl",
    "write_reports_jsonl",
    "format_report_markdown",
]


def __getattr__(name: str):
    if name in {"VulnerabilityDetectionAgent", "AgentState", "parse_verdict_from_text"}:
        from . import agent_loop

        return getattr(agent_loop, name)
    if name in {"EvidenceCompactorSubAgent"}:
        from . import compactor

        return getattr(compactor, name)
    if name in {"DetectionService", "compute_paired_metrics", "compute_grade"}:
        from . import service

        return getattr(service, name)
    if name in {"write_result_jsonl", "write_reports_jsonl", "format_report_markdown"}:
        from . import report

        return getattr(report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
