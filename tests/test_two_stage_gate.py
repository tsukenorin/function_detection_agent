"""代码级两阶段置信度门控（Two-Stage Confidence Gate）的单元测试。"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import (
    AgentConfig,
    AgentInput,
    EvidenceCompactorSubAgent,
    VulnerabilityDetectionAgent,
)
from repo import LazyWorkspace
from tool import Workspace


class GateScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    idx: int = 0
    bind_tools_called: int = 0
    seen_messages: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "gate-scripted-model"

    def bind_tools(self, tools, **kwargs):
        self.bind_tools_called += 1
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen_messages.append(list(messages))
        i = min(self.idx, len(self.responses) - 1)
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])


def test_two_stage_gate_closed_when_high_confidence_triggers_zero_repo_download(tmp_path):
    """高置信初判 -> 代码级门控关闭 -> 零工具调用、LazyWorkspace 零下载！"""
    materialized = {"count": 0}

    def materialize():
        materialized["count"] += 1
        return tmp_path

    lazy_ws = LazyWorkspace(tmp_path, materializer=materialize)

    model = GateScriptedModel(
        responses=[
            # Stage 1 (无工具初判): high confidence, needs_repo_check=False
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 0,
                        "confidence": "high",
                        "needs_repo_check": False,
                        "target_symbols": [],
                        "open_question": "",
                        "cwe_candidates": [],
                        "reason": "第 4 行已包含 !hmhd 完整空指针守卫。",
                        "evidence_cited": ["L4"],
                        "guard_assessment": "已防护",
                    },
                    ensure_ascii=False,
                )
            ),
            # Self-Verify 节点直接收敛输出最终判定
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 0,
                        "confidence": "high",
                        "cwe_candidates": [],
                        "reason": "反事实复核确认 L4 守卫有效，判定无漏洞。",
                        "evidence_cited": ["L4"],
                        "guard_assessment": "完整防护",
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        seen_messages=[],
    )

    events = []
    agent = VulnerabilityDetectionAgent(model=model)
    cfg = AgentConfig.improved(enable_rag=False, two_stage_gate=True)
    report = agent.detect(
        AgentInput(idx=219912, func="u32 GetHintFormat(GF_TrackBox *t) { if (!hmhd) return 0; }", repo_name="gpac/gpac"),
        workspace=lazy_ws,
        config=cfg,
        on_event=events.append,
    )

    assert report.prediction == 0
    assert report.analysis_record.tool_calls_count == 0
    assert report.metadata["two_stage_gate"] is True
    assert report.metadata["gate_opened"] is False
    assert materialized["count"] == 0  # LazyWorkspace 零物化！
    gate_events = [e for e in events if e.event_type == "gate_decision"]
    assert len(gate_events) == 1
    assert gate_events[0].payload["gate_opened"] is False


def test_two_stage_gate_blocks_when_high_confidence_even_if_needs_repo_true(tmp_path):
    """硬代码门控校验：即使模型误填 needs_repo_check=True，只要 confidence=high 仍强制关闭。"""
    (tmp_path / "a.c").write_text("int helper(int x) { return x; }\n", encoding="utf-8")
    ws = Workspace(tmp_path, read_only=True)

    model = GateScriptedModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 1,
                        "confidence": "high",
                        "needs_repo_check": True,
                        "target_symbols": ["helper"],
                        "open_question": "顺手看看",
                        "reason": "明显缺失判空",
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        seen_messages=[],
    )
    agent = VulnerabilityDetectionAgent(model=model)
    report = agent.detect(
        AgentInput(idx=195017, func="int f(int *p) { return *p; }", repo_name="demo/proj"),
        workspace=ws,
        config=AgentConfig.improved(enable_rag=False, enable_self_check=False, two_stage_gate=True),
    )

    assert report.prediction == 1
    assert report.metadata["gate_opened"] is False
    assert report.analysis_record.tool_calls_count == 0


def test_two_stage_gate_opens_on_medium_confidence_and_runs_focused_stage2(tmp_path):
    """低/中置信 + 明确外部符号缺口 -> 门控放行 -> 第 2 阶段 Tree-sitter 查证 + 客观压缩。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "helper.c").write_text(
        "int validate_idx(int x) {\n"
        "    if (x < 0 || x >= 32) return -1;\n"
        "    return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    ws = Workspace(tmp_path, read_only=True)

    main_model = GateScriptedModel(
        responses=[
            # Stage 1: medium confidence, needs_repo_check=True, target_symbols=["validate_idx"]
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 1,
                        "confidence": "medium",
                        "needs_repo_check": True,
                        "target_symbols": ["validate_idx"],
                        "open_question": "validate_idx 是否同时检查了负数与 >=32 上界？",
                        "reason": "函数内未直接看到 x 的上下界校验，依赖 validate_idx 实现。",
                    },
                    ensure_ascii=False,
                )
            ),
            # Stage 2 Turn 1: 调用 Tree-sitter read_symbol_body
            AIMessage(
                content="按锁定目标查证 validate_idx 实现。",
                tool_calls=[
                    {
                        "name": "read_symbol_body",
                        "args": {"symbol_name": "validate_idx"},
                        "id": "tc_s2_1",
                    }
                ],
            ),
            # Stage 2 Turn 2: 看到纯客观原代码行摘录后，确认已检查上下界，停止调用
            AIMessage(content="validate_idx 已明确检查 x < 0 || x >= 32，核心问题已解决。"),
            # Self-Verify 最终输出 0
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 0,
                        "confidence": "high",
                        "cwe_candidates": [],
                        "reason": "经第 2 阶段查证 src/helper.c:2，validate_idx 已完整校验上下界。",
                        "evidence_cited": ["src/helper.c:2"],
                        "guard_assessment": "上下界检查齐全",
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        seen_messages=[],
    )

    sub_model = GateScriptedModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "target": "validate_idx",
                        "verified_facts": ["src/helper.c:2 对 x 进行 < 0 || >= 32 校验并返回 -1"],
                        "key_code_lines": ["src/helper.c:2: if (x < 0 || x >= 32) return -1;"],
                        "evidence_completeness": "sufficient",
                        "next_frontier_symbols": [],
                        "filtered_noise_note": "",
                    },
                    ensure_ascii=False,
                )
            )
        ],
        seen_messages=[],
    )

    events = []
    agent = VulnerabilityDetectionAgent(
        model=main_model,
        compactor=EvidenceCompactorSubAgent(model=sub_model, min_chars=20),
    )
    report = agent.detect(
        AgentInput(
            idx=555,
            func="int entry(int x) { if (validate_idx(x) < 0) return -1; return arr[x]; }",
            repo_name="demo/proj",
            function_name="entry",
        ),
        workspace=ws,
        config=AgentConfig.improved(enable_rag=False, two_stage_gate=True, compactor_min_chars=20),
        on_event=events.append,
    )

    assert report.prediction == 0
    assert report.metadata["gate_opened"] is True
    assert report.analysis_record.tool_calls_count == 1
    assert len(report.analysis_record.evidence_summaries) == 1
    assert any(e.event_type == "gate_decision" and e.payload["gate_opened"] is True for e in events)
