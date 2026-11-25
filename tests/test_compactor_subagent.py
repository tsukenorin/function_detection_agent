"""纯客观证据摘要压缩子代理与渐进式多跳调用链编排的集成单元测试。"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import (
    AgentConfig,
    AgentInput,
    EvidenceCompactorProtocol,
    EvidenceCompactorSubAgent,
    VulnerabilityDetectionAgent,
    format_report_markdown,
)
from tool import Workspace


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    idx: int = 0
    seen_messages: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-subagent-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen_messages.append(list(messages))
        i = min(self.idx, len(self.responses) - 1)
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])


@pytest.fixture
def two_hop_ws(tmp_path):
    (tmp_path / "src").mkdir()
    padding = "\n".join(f"/* unrelated comment line {i} */" for i in range(40))
    (tmp_path / "src" / "chain.c").write_text(
        f"{padding}\n"
        "int leaf_check(int n) {\n"
        "    if (n < 0 || n >= 64) return -1;\n"
        "    return n;\n"
        "}\n"
        "int wrapper_call(int n) {\n"
        "    return leaf_check(n);\n"
        "}\n"
        "int entry_func(int n) {\n"
        "    return wrapper_call(n);\n"
        "}\n",
        encoding="utf-8",
    )
    return Workspace(tmp_path, read_only=True)


def test_neutral_compactor_extracts_facts_and_verbatim_lines_without_verdict_bias():
    sub_model = ScriptedModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "target": "leaf_check",
                        "verified_facts": ["src/chain.c:43: leaf_check 包含 n < 0 || n >= 64 条件返回 -1"],
                        "key_code_lines": ["src/chain.c:43: if (n < 0 || n >= 64) return -1;"],
                        "evidence_completeness": "sufficient",
                        "guard_status": "guards_present",  # 即使模型偶发带出也应被强制归零为 neutral
                        "next_frontier_symbols": [],
                        "filtered_noise_note": "已忽略 40 行无关注释",
                    },
                    ensure_ascii=False,
                )
            )
        ],
        seen_messages=[],
    )
    compactor = EvidenceCompactorSubAgent(model=sub_model, min_chars=200)
    assert isinstance(compactor, EvidenceCompactorProtocol)

    # 1) ERROR 输出直接短通，不消耗子代理 LLM 调用
    err_sum = compactor.compact(
        function_name="entry_func",
        audit_focus="check bounds",
        tool_name="read_symbol_body",
        tool_args={"symbol_name": "missing"},
        raw_output="ERROR: not found",
    )
    assert err_sum.evidence_completeness == "irrelevant"
    assert err_sum.guard_status == "neutral"
    assert len(sub_model.seen_messages) == 0

    # 2) 大段原始输出由子代理压缩，保留原代码行且绝不暴露主观 guard_status
    big_raw = "/* noise */\n" * 120 + "int leaf_check(int n) { if (n < 0 || n >= 64) return -1; }"
    compacted = compactor.compact(
        function_name="entry_func",
        audit_focus="核验 leaf_check 的条件语句",
        tool_name="read_symbol_body",
        tool_args={"symbol_name": "leaf_check"},
        raw_output=big_raw,
    )
    assert compacted.evidence_completeness == "sufficient"
    assert compacted.guard_status == "neutral"
    assert compacted.compacted_chars < compacted.raw_chars
    msg = compacted.to_compact_message()
    assert "if (n < 0 || n >= 64) return -1;" in msg
    assert "guards_present" not in msg
    assert "guards_missing" not in msg


def test_multihop_callchain_with_neutral_compactor_keeps_main_context_lean(two_hop_ws):
    main_model = ScriptedModel(
        responses=[
            AIMessage(
                content="先用 Tree-sitter 查看 entry_func 的 1 跳调用链。",
                tool_calls=[
                    {
                        "name": "inspect_call_chain",
                        "args": {"symbol_name": "entry_func", "direction": "callees", "depth": 1},
                        "id": "tc_hop1",
                    }
                ],
            ),
            AIMessage(
                content="前沿符号 wrapper_call 仍缺实现，继续展开下一跳。",
                tool_calls=[
                    {
                        "name": "read_symbol_body",
                        "args": {"symbol_name": "wrapper_call"},
                        "id": "tc_hop2",
                    }
                ],
            ),
            AIMessage(content="主代理亲自审阅原代码行摘录后，确认边界检查完整。"),
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 0,
                        "confidence": "high",
                        "cwe_candidates": [],
                        "reason": "经渐进式调用链查证，wrapper_call -> leaf_check 显式检查 n < 0 || n >= 64。",
                        "evidence_cited": ["src/chain.c:43"],
                        "guard_assessment": "下游已覆盖负数与上界检查",
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        seen_messages=[],
    )

    sub_model = ScriptedModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "target": "entry_func",
                        "verified_facts": ["src/chain.c:50: entry_func 调用 wrapper_call"],
                        "key_code_lines": ["src/chain.c:50: return wrapper_call(n);"],
                        "evidence_completeness": "needs_more_hops",
                        "next_frontier_symbols": ["wrapper_call"],
                        "filtered_noise_note": "过滤无关注释",
                    },
                    ensure_ascii=False,
                )
            ),
            AIMessage(
                content=json.dumps(
                    {
                        "target": "wrapper_call",
                        "verified_facts": ["src/chain.c:46: wrapper_call 转发给 leaf_check(n)"],
                        "key_code_lines": ["src/chain.c:43: if (n < 0 || n >= 64) return -1;"],
                        "evidence_completeness": "sufficient",
                        "next_frontier_symbols": [],
                        "filtered_noise_note": "仅保留原代码行",
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        seen_messages=[],
    )

    events = []
    compactor = EvidenceCompactorSubAgent(model=sub_model, min_chars=50)
    agent = VulnerabilityDetectionAgent(model=main_model, compactor=compactor)
    cfg = AgentConfig.improved(enable_rag=False, two_stage_gate=False, compactor_min_chars=50)

    report = agent.detect(
        AgentInput(
            idx=888,
            func="int entry_func(int n) { return wrapper_call(n); }",
            repo_name="demo/chain",
            function_name="entry_func",
        ),
        workspace=two_hop_ws,
        config=cfg,
        on_event=events.append,
    )

    assert report.prediction == 0
    assert report.analysis_record.tool_calls_count == 2
    assert len(report.analysis_record.evidence_summaries) == 2
    assert report.analysis_record.evidence_summaries[0].evidence_completeness == "needs_more_hops"
    assert report.analysis_record.evidence_summaries[1].evidence_completeness == "sufficient"
    assert all(s.guard_status == "neutral" for s in report.analysis_record.evidence_summaries)

    last_turn_msgs = main_model.seen_messages[-1]
    tool_msgs = [m for m in last_turn_msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert all("[压缩客观证据 |" in str(m.content) for m in tool_msgs)
    assert all("guards_present" not in str(m.content) for m in tool_msgs)
    assert all("unrelated comment line" not in str(m.content) for m in tool_msgs)

    assert sum(1 for e in events if e.event_type == "evidence_compacted") == 2
    md = format_report_markdown(report)
    assert "证据摘要压缩子代理提纯记录" in md
