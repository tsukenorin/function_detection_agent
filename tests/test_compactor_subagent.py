"""证据摘要压缩子代理与渐进式多跳调用链编排的集成单元测试。"""

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


def test_compactor_subagent_compresses_large_output_and_short_circuits_errors():
    sub_model = ScriptedModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "target": "leaf_check",
                        "verified_facts": ["src/chain.c:43: leaf_check 校验 n < 0 || n >= 64"],
                        "guard_status": "guards_present",
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
    assert err_sum.guard_status == "irrelevant"
    assert len(sub_model.seen_messages) == 0

    # 2) 大段原始输出由子代理压缩，字符数显著下降
    big_raw = "/* noise */\n" * 120 + "int leaf_check(int n) { if (n < 0 || n >= 64) return -1; }"
    compacted = compactor.compact(
        function_name="entry_func",
        audit_focus="核验 leaf_check 是否有边界校验",
        tool_name="read_symbol_body",
        tool_args={"symbol_name": "leaf_check"},
        raw_output=big_raw,
    )
    assert compacted.guard_status == "guards_present"
    assert compacted.compacted_chars < compacted.raw_chars
    assert len(sub_model.seen_messages) == 1


def test_multihop_callchain_with_compactor_subagent_keeps_main_context_lean(two_hop_ws):
    # 主代理：第 1 步展开 1 跳调用链 -> 收到压缩摘要(inconclusive, next=[wrapper_call])
    #      -> 第 2 步读取 wrapper_call -> 收到压缩摘要(guards_present) -> 第 3 步自检收敛判 0
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
                content="前沿符号 wrapper_call 仍不明确，继续展开下一跳并查看实现。",
                tool_calls=[
                    {
                        "name": "read_symbol_body",
                        "args": {"symbol_name": "wrapper_call"},
                        "id": "tc_hop2",
                    }
                ],
            ),
            AIMessage(content="根据压缩后的调用链核验结果，边界检查完整，准备裁决。"),
            AIMessage(
                content=json.dumps(
                    {
                        "prediction": 0,
                        "confidence": "high",
                        "cwe_candidates": [],
                        "reason": "经渐进式调用链查证，wrapper_call -> leaf_check 已完成上下界守卫。",
                        "evidence_cited": ["src/chain.c:43"],
                        "guard_assessment": "下游已覆盖负数与上界检查",
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        seen_messages=[],
    )

    # 独立压缩子代理（两轮分别返回高密度压缩事实）
    sub_model = ScriptedModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "target": "entry_func",
                        "verified_facts": ["src/chain.c:50: entry_func 直接调用 wrapper_call"],
                        "guard_status": "inconclusive",
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
                        "verified_facts": ["src/chain.c:46: wrapper_call 转发给 leaf_check(n) 完成 0<=n<64 校验"],
                        "guard_status": "guards_present",
                        "next_frontier_symbols": [],
                        "filtered_noise_note": "仅保留守卫事实",
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
    cfg = AgentConfig.improved(enable_rag=False, compactor_min_chars=50)

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
    assert report.analysis_record.evidence_summaries[0].next_frontier_symbols == ["wrapper_call"]
    assert report.analysis_record.evidence_summaries[1].guard_status == "guards_present"
    assert report.metadata["enable_evidence_compactor"] is True
    assert report.metadata["reasoning_effort"] == "low"

    # 验证主代理收到的 ToolMessage 全是压缩后的结构化证据，而不是原始 JSON / 大段注释
    last_turn_msgs = main_model.seen_messages[-1]
    tool_msgs = [m for m in last_turn_msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert all("[压缩证据 |" in str(m.content) for m in tool_msgs)
    assert all("unrelated comment line" not in str(m.content) for m in tool_msgs)

    # 验证事件流与 Markdown 报告渲染
    assert sum(1 for e in events if e.event_type == "evidence_compacted") == 2
    md = format_report_markdown(report)
    assert "证据摘要压缩子代理提纯记录" in md
