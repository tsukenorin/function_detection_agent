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
]
