"""LangGraph Agent 编排层、工具预算控制、自检节点与报告持久化的单元测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import (
    AgentConfig,
    AgentInput,
    KnowledgeHit,
    KnowledgeRetrieveRecord,
    VulnerabilityDetectionAgent,
    VulnerabilityDetector,
    format_report_markdown,
    write_reports_jsonl,
    write_result_jsonl,
)
from agent.agent_loop import parse_verdict_from_text
from tool import Workspace


class ScriptedChatModel(BaseChatModel):
    """按脚本依次返回预设 AIMessage 的确定性测试模型。"""

    responses: list[AIMessage]
    call_index: int = 0
    received_messages: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received_messages.append(list(messages))
        idx = min(self.call_index, len(self.responses) - 1)
        self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])


class StubRetriever:
    def retrieve(self, query, *, func_code="", top_k=3, exclude_commit=None):
        return KnowledgeRetrieveRecord(
            query=query,
            top_k=top_k,
            hits=[
                KnowledgeHit(
                    case_id="CVE-2020-9999:1",
                    cve="CVE-2020-9999",
                    cwe=["CWE-125"],
                    project="demo",
                    score=12.5,
                    cve_desc="Buffer over-read when len is unchecked.",
                    fix_pattern="+ if (len > MAX_LEN) return -1;",
                )
            ],
            latency_ms=1.2,
        )


@pytest.fixture
def sample_ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "helper.c").write_text(
        "#define MAX_LEN 64\n"
        "int check_bounds(int len) {\n"
        "    if (len < 0 || len > MAX_LEN) return -1;\n"
        "    return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    return Workspace(tmp_path, read_only=True)


def test_parse_verdict_from_various_formats():
    v1 = parse_verdict_from_text(
        '分析如下：\n```json\n{"prediction": 1, "confidence": "high", "cwe_candidates": ["CWE-787"], "reason": "越界写", "evidence_cited": ["L3"]}\n```'
    )
    assert v1 is not None and v1.prediction == 1 and v1.cwe_candidates == ["CWE-787"]

    v2 = parse_verdict_from_text('{"prediction": "0", "reason": "已校验边界"}')
    assert v2 is not None and v2.prediction == 0

    assert parse_verdict_from_text("完全没有 JSON 的普通文本") is None


def test_baseline_mode_runs_single_pass_without_rag_or_tools(sample_ws):
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 1,
                        "confidence": "medium",
                        "cwe_candidates": ["CWE-125"],
                        "reason": "未看到边界检查",
                        "evidence_cited": ["L2"],
                        "guard_assessment": "无检查",
                    },
                    ensure_ascii=False,
                )
            )
        ],
        received_messages=[],
    )
    agent = VulnerabilityDetectionAgent(model=model, retriever=StubRetriever())
    assert isinstance(agent, VulnerabilityDetector)

    agent_input = AgentInput(
        idx=101,
        func="int foo(int len) { return check_bounds(len); }",
        repo_name="org/demo",
        function_name="foo",
    )
    report = agent.detect(agent_input, workspace=sample_ws, config=AgentConfig.baseline())

    assert report.idx == 101
    assert report.prediction == 1
    assert report.label_text == "有漏洞"
    assert report.analysis_record.knowledge_records == []
    assert report.analysis_record.repo_records == []
    assert report.analysis_record.tool_calls_count == 0


def test_improved_mode_autonomous_rag_repo_tools_and_self_check(sample_ws):
    events = []
    model = ScriptedChatModel(
        responses=[
            # 第 1 轮：自主决定调用 find_symbol 核查 check_bounds 实现
            AIMessage(
                content="需要查证 check_bounds 是否包含边界检查。",
                tool_calls=[
                    {
                        "name": "find_symbol",
                        "args": {"symbol_name": "check_bounds"},
                        "id": "tc_1",
                    }
                ],
            ),
            # 第 2 轮：拿到仓库代码后，确认已有边界校验，停止调用工具
            AIMessage(
                content="已核查 src/helper.c，check_bounds 已检查 len < 0 || len > MAX_LEN，初步判 0。"
            ),
            # 第 3 轮（self_verify 节点）：输出最终结构化 JSON
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 0,
                        "confidence": "high",
                        "cwe_candidates": [],
                        "reason": "经查证 src/helper.c:3，check_bounds 已严格校验上下界。",
                        "evidence_cited": ["src/helper.c:3"],
                        "guard_assessment": "check_bounds 覆盖负数与上界检查",
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        received_messages=[],
    )
    agent = VulnerabilityDetectionAgent(model=model, retriever=StubRetriever())
    agent_input = AgentInput(
        idx=102,
        func="int foo(int len) { if (check_bounds(len) < 0) return -1; return buf[len]; }",
        repo_name="org/demo",
        function_name="foo",
    )
    report = agent.detect(
        agent_input,
        workspace=sample_ws,
        config=AgentConfig.improved(two_stage_gate=False, enable_evidence_compactor=False),
        on_event=events.append,
    )

    assert report.prediction == 0
    assert report.label_text == "无漏洞"
    assert len(report.analysis_record.knowledge_records) == 1
    assert len(report.analysis_record.repo_records) == 1
    assert report.analysis_record.repo_records[0].tool_name == "find_symbol"
    assert report.analysis_record.repo_records[0].ok is True
    assert report.analysis_record.stop_reason == "sufficient_evidence"

    event_types = [e.event_type for e in events]
    assert event_types[0] == "started"
    assert "knowledge_retrieved" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "self_check" in event_types
    assert "verdict" in event_types
    assert event_types[-1] == "report_ready"


def test_budget_exhausted_forces_termination_and_json_repair(sample_ws, tmp_path):
    # 模拟模型不停调用工具且返回非 JSON 文本，验证预算强制截断 + JSON 修复兜底
    loop_call = AIMessage(
        content="继续探索",
        tool_calls=[{"name": "list_dir", "args": {"path": "."}, "id": "tc_loop"}],
    )
    model = ScriptedChatModel(
        responses=[
            loop_call,
            AIMessage(content="我认为这里存在越界风险，但忘记写 JSON 了"),
            AIMessage(
                content='{"prediction": 1, "confidence": "medium", "cwe_candidates": ["CWE-119"], "reason": "修复后解析成功"}'
            ),
        ],
        received_messages=[],
    )
    cfg = AgentConfig.improved(
        max_tool_calls=1,
        max_steps=4,
        enable_self_check=False,
        two_stage_gate=False,
        enable_evidence_compactor=False,
    )
    agent = VulnerabilityDetectionAgent(model=model, retriever=StubRetriever())
    report = agent.detect(
        AgentInput(idx=103, func="void bar(char *p) { p[99] = 0; }", repo_name="org/demo"),
        workspace=sample_ws,
        config=cfg,
    )

    assert report.prediction == 1
    assert report.analysis_record.tool_calls_count == 1
    assert report.analysis_record.stop_reason == "budget_exhausted"

    # 验证报告导出工具
    res_file = write_result_jsonl([report], tmp_path / "result.jsonl")
    rep_file = write_reports_jsonl([report], tmp_path / "reports.jsonl")
    assert json.loads(res_file.read_text(encoding="utf-8").strip()) == {
        "idx": 103,
        "prediction": 1,
    }
    assert json.loads(rep_file.read_text(encoding="utf-8").strip())["label_text"] == "有漏洞"
    md = format_report_markdown(report)
    assert "idx=103" in md and "历史 CVE 知识库检索记录" in md and "代码仓库按需读取记录" in md


def test_evidence_gated_config_and_prompt_directive():
    from agent.prompts import build_improved_user_prompt

    cfg = AgentConfig.improved()
    assert cfg.evidence_gated_tools is True
    assert cfg.two_stage_gate is True
    assert cfg.enable_evidence_compactor is True
    assert cfg.reasoning_effort == "low"
    assert cfg.max_steps == 8
    assert cfg.max_tool_calls == 4
    assert cfg.max_read_chars == 24000

    prompt = build_improved_user_prompt(
        AgentInput(idx=1, func="int f() { return 0; }", repo_name="org/demo"),
        has_repo_tools=True,
        evidence_gated_tools=True,
    )
    assert "渐进式调用链策略" in prompt
    assert "默认 0 次调用" in prompt
