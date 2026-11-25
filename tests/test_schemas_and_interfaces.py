"""核心 Schema 契约、隔离不变量与 Protocol 接口的单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent import (
    AgentConfig,
    AgentEvent,
    AgentInput,
    AgentVerdict,
    AnalysisTrace,
    DetectionReport,
    KnowledgeHit,
    KnowledgeRetrieveRecord,
    KnowledgeRetriever,
    ParentCommitResolver,
    RawSample,
    RepoReadRecord,
    RepoRef,
    SampleLoaderProtocol,
    SnapshotProvider,
    TaskServiceProtocol,
    VulnerabilityDetector,
)


def test_raw_sample_validates_strict_ints():
    sample = RawSample(
        idx=194963,
        commit_url="https://github.com/ImageMagick/ImageMagick6/commit/dc070da861a015d3c97488fdcca6063b44d47a7b",
        func="int f(void) { return 0; }",
        target=1,
    )
    assert sample.idx == 194963
    assert sample.target == 1

    # 无标签单样本（UI 场景）允许 target=None
    unlabeled = RawSample(
        idx=1,
        commit_url="https://github.com/org/repo/commit/abcdef123456",
        func="void g(void) {}",
    )
    assert unlabeled.target is None

    # 拒绝布尔值伪装整数
    with pytest.raises(ValidationError):
        RawSample(idx=True, commit_url="https://example.com", func="int a();", target=1)
    with pytest.raises(ValidationError):
        RawSample(idx=1, commit_url="https://example.com", func="int a();", target=False)


def test_agent_input_forbids_leaking_fields():
    """红线校验：AgentInput 绝不允许携带 target / label / commit_message / git_diff。"""
    clean = AgentInput(
        idx=100,
        func="void foo(int n) {}",
        repo_name="org/repo",
        function_name="foo",
    )
    assert clean.idx == 100
    assert clean.function_name == "foo"

    for forbidden in (
        {"target": 1},
        {"label": 0},
        {"commit_message": "Fix buffer overflow"},
        {"git_diff": "- old\n+ new"},
        {"version_role": "parent"},
    ):
        with pytest.raises(ValidationError):
            AgentInput(idx=100, func="void foo() {}", **forbidden)


def test_detection_report_four_elements_and_result_entry():
    hit = KnowledgeHit(
        case_id="CVE-2013-6449:0",
        cve="CVE-2013-6449",
        cwe=["CWE-310"],
        project="openssl",
        score=0.88,
        cve_desc="wrong version field",
    )
    trace = AnalysisTrace(
        knowledge_records=[KnowledgeRetrieveRecord(query="ssl_get_algorithm2", top_k=3, hits=[hit])],
        repo_records=[
            RepoReadRecord(
                tool_name="read_file",
                args={"path": "ssl/s3_lib.c", "offset": 1, "limit": 50},
                path="ssl/s3_lib.c",
                ok=True,
                summary="read 50 lines",
                chars_returned=1200,
                step=1,
            )
        ],
        tool_calls_count=1,
        total_read_chars=1200,
        steps_taken=2,
        stop_reason="sufficient_evidence",
    )
    verdict = AgentVerdict(
        prediction=1,
        confidence="high",
        cwe_candidates=["CWE-125"],
        reason="缺少长度边界检查导致越界读。",
        evidence_cited=["ssl/s3_lib.c:42"],
        guard_assessment="未检查 len < sizeof(buf)",
    )
    report = DetectionReport.from_verdict(194989, verdict, trace, metadata={"mode": "improved"})

    assert report.idx == 194989
    assert report.prediction == 1
    assert report.label_text == "有漏洞"
    assert "越界读" in report.reason
    assert len(report.analysis_record.knowledge_records) == 1
    assert len(report.analysis_record.repo_records) == 1
    assert report.to_result_entry() == {"idx": 194989, "prediction": 1}

    benign_verdict = AgentVerdict(prediction=0, reason="已做完整边界检查。")
    benign_report = DetectionReport.from_verdict(218785, benign_verdict, AnalysisTrace())
    assert benign_report.label_text == "无漏洞"
    assert benign_report.to_result_entry() == {"idx": 218785, "prediction": 0}


def test_agent_config_presets():
    base = AgentConfig.baseline(model_name="deepseek-chat")
    imp = AgentConfig.improved(model_name="deepseek-chat")

    assert base.mode == "baseline"
    assert base.enable_rag is False
    assert base.enable_repo_tools is False
    assert base.enable_self_check is False
    assert base.max_tool_calls == 0

    assert imp.mode == "improved"
    assert imp.enable_rag is True
    assert imp.enable_repo_tools is True
    assert imp.enable_self_check is True
    assert imp.max_tool_calls > 0


def test_repo_ref_and_event():
    ref = RepoRef(
        owner="gpac",
        repo="gpac",
        commit_sha="ad18ece95fa064efc0995c4ab2c985f77fb166ec",
        allowed_sha="1111111111111111111111111111111111111111",
        version_role="parent",
        original_url="https://github.com/gpac/gpac/commit/ad18ece95fa064efc0995c4ab2c985f77fb166ec",
    )
    assert ref.repo_full_name == "gpac/gpac"

    ev = AgentEvent(event_type="tool_call", idx=1, step=1, payload={"tool": "grep"})
    assert ev.event_type == "tool_call"
    assert ev.timestamp > 0


def test_protocols_are_runtime_checkable():
    class DummyRetriever:
        def retrieve(self, query, *, func_code="", top_k=3, exclude_commit=None):
            return KnowledgeRetrieveRecord(query=query, top_k=top_k)

    class DummyResolver:
        def resolve_parent(self, owner, repo, commit_sha):
            return "a" * 40

    class DummyProvider:
        def get_workspace(self, repo_ref):
            raise NotImplementedError

    class DummyLoader:
        def load(self, raw):
            raise NotImplementedError

    class DummyDetector:
        def detect(self, agent_input, *, workspace=None, config=None, on_event=None):
            raise NotImplementedError

    class DummyService:
        def detect_single(self, sample, *, config=None, on_event=None):
            raise NotImplementedError

        def detect_batch(
            self,
            samples,
            *,
            config=None,
            result_path=None,
            report_path=None,
            on_event=None,
        ):
            raise NotImplementedError

    assert isinstance(DummyRetriever(), KnowledgeRetriever)
    assert isinstance(DummyResolver(), ParentCommitResolver)
    assert isinstance(DummyProvider(), SnapshotProvider)
    assert isinstance(DummyLoader(), SampleLoaderProtocol)
    assert isinstance(DummyDetector(), VulnerabilityDetector)
    assert isinstance(DummyService(), TaskServiceProtocol)
