"""基于 LangGraph StateGraph 的自主证据驱动漏洞检测 Agent 编排层。

状态图拓扑：

    START
      │
      ▼
    retrieve_knowledge   (按配置检索历史 CVE 领域知识库并写系统日志)
      │
      ▼
    analyze_and_plan     (LLM 结合领域知识展开初步分析；自主决定是否调用仓库工具)
      ├──(有 tool_calls 且预算未耗尽)──► execute_tools ──► 返回 analyze_and_plan
      ├──(证据充分 / 预算耗尽 且开启自检)──► self_verify ──► finalize_verdict
      └──(证据充分 / 预算耗尽 且关闭自检)──► finalize_verdict
      │
      ▼
    build_report         (基于系统采集的 AnalysisTrace 组装四要素 DetectionReport)
      │
      ▼
    END

红线保障：
- 自主决策：由 LangGraph 条件边与 LLM tool-calling 自主决定“查什么、何时停止”；
- 严格隔离：只接收裁剪后的 ``AgentInput`` 与单版本只读 ``Workspace``；
- 约束写在代码里：工具预算（次数 + 字符数）、步数上限与审计日志均由编排层硬性强制。
"""

from __future__ import annotations

import json
import re
import time
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from tool.file_tools import build_repo_tools
from tool.trace import ToolBudget, ToolTraceRecorder
from tool.workspace import Workspace

from .compactor import EvidenceCompactorSubAgent
from .interfaces import EventCallback, EvidenceCompactorProtocol, KnowledgeRetriever
from .model_factory import build_chat_model
from .prompts import (
    BASELINE_SYSTEM_PROMPT,
    IMPROVED_SYSTEM_PROMPT,
    JSON_REPAIR_PROMPT,
    SELF_CHECK_PROMPT,
    build_baseline_user_prompt,
    build_improved_user_prompt,
)
from .schema import (
    AgentConfig,
    AgentEvent,
    AgentInput,
    AgentVerdict,
    AnalysisTrace,
    DetectionReport,
    EvidenceSummary,
    KnowledgeRetrieveRecord,
)

__all__ = [
    "AgentState",
    "parse_verdict_from_text",
    "VulnerabilityDetectionAgent",
]


class AgentState(TypedDict):
    """LangGraph 状态机的共享状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    step: int
    stop_reason: Literal[
        "sufficient_evidence",
        "budget_exhausted",
        "no_repo_available",
        "error",
    ]
    knowledge_record: KnowledgeRetrieveRecord | None
    pending_tool_outputs: list[dict[str, Any]]
    verdict: AgentVerdict | None
    report: DetectionReport | None


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content or "")


def parse_verdict_from_text(text: str) -> AgentVerdict | None:
    """从 LLM 输出中稳健提取并校验 ``AgentVerdict`` JSON。"""
    if not text or not text.strip():
        return None

    candidates: list[str] = []
    # 1) 优先提取 ```json ... ``` 代码块（取最后一个，防止前面有示例块）
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.extend(reversed(fenced))

    # 2) 尝试整段作为 JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    # 3) 提取文本中最后一个最外层 {...} 对象
    decoder = json.JSONDecoder()
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] == "{":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict) and "prediction" in obj:
                    candidates.append(json.dumps(obj, ensure_ascii=False))
                    break
            except json.JSONDecodeError:
                continue

    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "prediction" not in data:
            continue
        # 兼容部分模型输出字符串 "0"/"1" 的情况
        pred = data.get("prediction")
        if isinstance(pred, str) and pred.strip() in {"0", "1"}:
            data["prediction"] = int(pred.strip())
        if not data.get("reason"):
            data["reason"] = "模型未提供详细说明"
        if isinstance(data.get("cwe_candidates"), str):
            data["cwe_candidates"] = [data["cwe_candidates"]]
        if isinstance(data.get("evidence_cited"), str):
            data["evidence_cited"] = [data["evidence_cited"]]
        try:
            return AgentVerdict.model_validate(data)
        except ValidationError:
            continue
    return None


class VulnerabilityDetectionAgent:
    """漏洞检测 Agent 编排器（实现 ``VulnerabilityDetector`` 核心接口）。"""

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        retriever: KnowledgeRetriever | None = None,
        compactor: EvidenceCompactorProtocol | None = None,
        default_config: AgentConfig | None = None,
    ) -> None:
        self._model = model
        self.retriever = retriever
        self._compactor = compactor
        self.default_config = default_config or AgentConfig.improved()

    def _get_model(self, config: AgentConfig) -> BaseChatModel:
        if self._model is not None:
            return self._model
        return build_chat_model(config)

    def _get_compactor(self, config: AgentConfig) -> EvidenceCompactorProtocol | None:
        if not config.enable_evidence_compactor:
            return None
        if self._compactor is not None:
            return self._compactor
        return EvidenceCompactorSubAgent(
            model=self._model,
            config=config,
            min_chars=config.compactor_min_chars,
        )

    def build_graph(
        self,
        agent_input: AgentInput,
        *,
        workspace: Workspace | None,
        config: AgentConfig,
        recorder: ToolTraceRecorder,
        budget: ToolBudget,
        on_event: EventCallback | None = None,
        start_time: float = 0.0,
    ):
        """为单次样本检测编译一个独立的 LangGraph 状态图（无跨样本共享状态）。"""
        model = self._get_model(config)
        use_repo_tools = (
            config.enable_repo_tools
            and workspace is not None
            and config.max_tool_calls > 0
        )
        tools = (
            build_repo_tools(workspace, recorder=recorder, budget=budget)
            if use_repo_tools and workspace is not None
            else []
        )
        tool_map = {t.name: t for t in tools}
        model_with_tools = model.bind_tools(tools) if tools else model
        compactor = self._get_compactor(config)
        knowledge_records: list[KnowledgeRetrieveRecord] = []
        evidence_summaries: list[EvidenceSummary] = []

        def emit(
            event_type: Literal[
                "started",
                "knowledge_retrieved",
                "thought",
                "tool_call",
                "tool_result",
                "evidence_compacted",
                "self_check",
                "verdict",
                "report_ready",
                "error",
            ],
            step: int,
            payload: dict[str, Any],
        ) -> None:
            if on_event is not None:
                on_event(
                    AgentEvent(
                        event_type=event_type,
                        idx=agent_input.idx,
                        step=step,
                        payload=payload,
                    )
                )

        # ------------------------------------------------------------------
        # Node 1: 领域知识库 RAG 检索
        # ------------------------------------------------------------------
        def retrieve_knowledge_node(state: AgentState) -> dict[str, Any]:
            if (
                not config.enable_rag
                or self.retriever is None
                or config.top_k_knowledge <= 0
            ):
                return {"knowledge_record": None}

            query = " ".join(
                p
                for p in (agent_input.repo_name, agent_input.function_name or "")
                if p
            ) or f"sample_{agent_input.idx}"
            try:
                rec = self.retriever.retrieve(
                    query,
                    func_code=agent_input.func,
                    top_k=config.top_k_knowledge,
                    min_score=config.rag_min_score,
                )
            except TypeError:
                rec = self.retriever.retrieve(
                    query,
                    func_code=agent_input.func,
                    top_k=config.top_k_knowledge,
                )
            knowledge_records.append(rec)
            emit(
                "knowledge_retrieved",
                0,
                {
                    "query": rec.query,
                    "hits": [
                        {
                            "cve": h.cve,
                            "cwe": h.cwe,
                            "project": h.project,
                            "score": h.score,
                        }
                        for h in rec.hits
                    ],
                    "latency_ms": rec.latency_ms,
                },
            )
            return {"knowledge_record": rec}

        # ------------------------------------------------------------------
        # Node 2: 初步分析 / 多轮证据规划（核心自主推理节点）
        # ------------------------------------------------------------------
        def analyze_and_plan_node(state: AgentState) -> dict[str, Any]:
            next_step = state["step"] + 1
            recorder.current_step = next_step

            messages = list(state["messages"])
            new_messages: list[AnyMessage] = []
            if not messages:
                if config.mode == "baseline" and not config.enable_rag and not use_repo_tools:
                    sys_msg = SystemMessage(content=BASELINE_SYSTEM_PROMPT)
                    usr_msg = HumanMessage(
                        content=build_baseline_user_prompt(
                            agent_input, max_func_chars=config.max_func_chars
                        )
                    )
                else:
                    sys_msg = SystemMessage(content=IMPROVED_SYSTEM_PROMPT)
                    usr_msg = HumanMessage(
                        content=build_improved_user_prompt(
                            agent_input,
                            knowledge_record=state.get("knowledge_record"),
                            has_repo_tools=use_repo_tools,
                            evidence_gated_tools=config.evidence_gated_tools,
                            max_func_chars=config.max_func_chars,
                        )
                    )
                new_messages.extend([sys_msg, usr_msg])
                messages.extend([sys_msg, usr_msg])

            # 若已到达最大步数或工具预算已耗尽，则不再绑定工具，强制模型收敛给出结论
            allow_tools_now = (
                bool(tools)
                and not budget.exhausted
                and next_step < config.max_steps
            )
            active_model = model_with_tools if allow_tools_now else model
            response = active_model.invoke(messages)
            if not isinstance(response, AIMessage):
                response = AIMessage(content=_extract_text_content(response))

            new_messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            emit(
                "thought",
                next_step,
                {
                    "planned_tool_calls": [tc.get("name") for tc in tool_calls],
                    "content_preview": _extract_text_content(response.content)[:240],
                },
            )

            stop_reason = state["stop_reason"]
            if tool_calls and (not allow_tools_now):
                stop_reason = "budget_exhausted"
            elif not use_repo_tools and config.enable_repo_tools and workspace is None:
                stop_reason = "no_repo_available"

            return {
                "messages": new_messages,
                "step": next_step,
                "stop_reason": stop_reason,
            }

        # ------------------------------------------------------------------
        # 条件边：是否继续调用仓库工具
        # ------------------------------------------------------------------
        def route_after_analyze(
            state: AgentState,
        ) -> Literal["execute_tools", "self_verify", "finalize_verdict"]:
            last = state["messages"][-1] if state["messages"] else None
            tool_calls = getattr(last, "tool_calls", None) or []
            if (
                tool_calls
                and bool(tools)
                and not budget.exhausted
                and state["step"] < config.max_steps
            ):
                return "execute_tools"
            if config.enable_self_check:
                return "self_verify"
            return "finalize_verdict"

        # ------------------------------------------------------------------
        # Node 3: 执行仓库只读沙箱工具（硬预算 + 自动审计）
        # ------------------------------------------------------------------
        def execute_tools_node(state: AgentState) -> dict[str, Any]:
            last = state["messages"][-1]
            tool_calls = getattr(last, "tool_calls", None) or []
            audit_focus = _extract_text_content(last.content)[:500]
            pending: list[dict[str, Any]] = []
            stop_reason = state["stop_reason"]

            for tc in tool_calls:
                name = str(tc.get("name", ""))
                args = tc.get("args") or {}
                call_id = str(tc.get("id") or f"call_{state['step']}_{name}")
                tool_obj = tool_map.get(name)
                if tool_obj is None:
                    content = f"ERROR: 未知工具 `{name}`，可用工具：{list(tool_map)}"
                else:
                    try:
                        content = str(tool_obj.invoke(args))
                    except Exception as exc:  # pragma: no cover - 兜底容错
                        content = f"ERROR: 工具 `{name}` 执行失败：{exc}"
                pending.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": dict(args),
                        "raw_output": content,
                        "audit_focus": audit_focus,
                    }
                )

            if budget.exhausted:
                stop_reason = "budget_exhausted"
            return {"pending_tool_outputs": pending, "stop_reason": stop_reason}

        # ------------------------------------------------------------------
        # Node 3.5: 证据摘要压缩子代理节点（隔离短上下文提纯，防稀释主代理）
        # ------------------------------------------------------------------
        def compact_evidence_node(state: AgentState) -> dict[str, Any]:
            pending = state.get("pending_tool_outputs") or []
            tool_messages: list[ToolMessage] = []
            fn_name = agent_input.function_name or f"sample_{agent_input.idx}"

            for item in pending:
                raw_out = str(item["raw_output"])
                if compactor is not None:
                    summary = compactor.compact(
                        function_name=fn_name,
                        audit_focus=str(item.get("audit_focus") or ""),
                        tool_name=str(item["name"]),
                        tool_args=dict(item.get("args") or {}),
                        raw_output=raw_out,
                    )
                    evidence_summaries.append(summary)
                    emit(
                        "evidence_compacted",
                        state["step"],
                        {
                            "tool_name": summary.tool_name,
                            "target": summary.target,
                            "guard_status": summary.guard_status,
                            "raw_chars": summary.raw_chars,
                            "compacted_chars": summary.compacted_chars,
                            "next_frontier_symbols": summary.next_frontier_symbols,
                        },
                    )
                    msg_content = summary.to_compact_message()
                else:
                    msg_content = raw_out

                tool_messages.append(
                    ToolMessage(
                        content=msg_content,
                        tool_call_id=str(item["call_id"]),
                        name=str(item["name"]),
                    )
                )

            return {"messages": tool_messages, "pending_tool_outputs": []}

        # ------------------------------------------------------------------
        # Node 4: 防御检查核验与反事实自检节点（Improved 改进措施）
        # ------------------------------------------------------------------
        def self_verify_node(state: AgentState) -> dict[str, Any]:
            next_step = state["step"] + 1
            prompt_msg = HumanMessage(content=SELF_CHECK_PROMPT)
            messages = list(state["messages"]) + [prompt_msg]
            response = model.invoke(messages)
            if not isinstance(response, AIMessage):
                response = AIMessage(content=_extract_text_content(response))
            emit(
                "self_check",
                next_step,
                {"content_preview": _extract_text_content(response.content)[:240]},
            )
            return {"messages": [prompt_msg, response], "step": next_step}

        # ------------------------------------------------------------------
        # Node 5: 结构化判定解析（含 JSON Schema 修复兜底）
        # ------------------------------------------------------------------
        def finalize_verdict_node(state: AgentState) -> dict[str, Any]:
            last = state["messages"][-1] if state["messages"] else None
            raw_text = _extract_text_content(last.content) if last is not None else ""
            verdict = parse_verdict_from_text(raw_text)
            extra_messages: list[AnyMessage] = []

            if verdict is None:
                repair_msg = HumanMessage(content=JSON_REPAIR_PROMPT)
                repaired = model.invoke(list(state["messages"]) + [repair_msg])
                if not isinstance(repaired, AIMessage):
                    repaired = AIMessage(content=_extract_text_content(repaired))
                extra_messages.extend([repair_msg, repaired])
                verdict = parse_verdict_from_text(_extract_text_content(repaired.content))

            stop_reason = state["stop_reason"]
            if verdict is None:
                # 最终保底：绝不崩溃，按保守策略输出可追踪错误判定
                stop_reason = "error"
                verdict = AgentVerdict(
                    prediction=0,
                    confidence="low",
                    cwe_candidates=[],
                    reason=f"未能从模型输出中解析合法 JSON，已回退为保守判定。原始摘要：{raw_text[:200]}",
                    evidence_cited=[],
                    guard_assessment="解析失败",
                )

            emit("verdict", state["step"], verdict.model_dump())
            return {
                "messages": extra_messages,
                "verdict": verdict,
                "stop_reason": stop_reason,
            }

        # ------------------------------------------------------------------
        # Node 6: 拼装四要素 DetectionReport（严格使用系统 trace）
        # ------------------------------------------------------------------
        def build_report_node(state: AgentState) -> dict[str, Any]:
            verdict = state["verdict"]
            assert verdict is not None
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2) if start_time else 0.0
            trace = AnalysisTrace(
                knowledge_records=list(knowledge_records),
                repo_records=list(recorder.records),
                evidence_summaries=list(evidence_summaries),
                tool_calls_count=len(recorder.records),
                total_read_chars=sum(r.chars_returned for r in recorder.records),
                steps_taken=state["step"],
                stop_reason=state["stop_reason"],
            )
            report = DetectionReport.from_verdict(
                idx=agent_input.idx,
                verdict=verdict,
                trace=trace,
                metadata={
                    "mode": config.mode,
                    "model_name": config.model_name,
                    "reasoning_effort": config.reasoning_effort,
                    "repo_name": agent_input.repo_name,
                    "function_name": agent_input.function_name,
                    "enable_rag": config.enable_rag,
                    "enable_repo_tools": use_repo_tools,
                    "evidence_gated_tools": config.evidence_gated_tools,
                    "enable_evidence_compactor": compactor is not None,
                    "enable_self_check": config.enable_self_check,
                    "elapsed_ms": elapsed_ms,
                },
            )
            emit("report_ready", state["step"], report.to_result_entry())
            return {"report": report}

        graph = StateGraph(AgentState)
        graph.add_node("retrieve_knowledge", retrieve_knowledge_node)
        graph.add_node("analyze_and_plan", analyze_and_plan_node)
        graph.add_node("execute_tools", execute_tools_node)
        graph.add_node("compact_evidence", compact_evidence_node)
        graph.add_node("self_verify", self_verify_node)
        graph.add_node("finalize_verdict", finalize_verdict_node)
        graph.add_node("build_report", build_report_node)

        graph.add_edge(START, "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "analyze_and_plan")
        graph.add_conditional_edges(
            "analyze_and_plan",
            route_after_analyze,
            {
                "execute_tools": "execute_tools",
                "self_verify": "self_verify",
                "finalize_verdict": "finalize_verdict",
            },
        )
        graph.add_edge("execute_tools", "compact_evidence")
        graph.add_edge("compact_evidence", "analyze_and_plan")
        graph.add_edge("self_verify", "finalize_verdict")
        graph.add_edge("finalize_verdict", "build_report")
        graph.add_edge("build_report", END)
        return graph.compile()

    def detect(
        self,
        agent_input: AgentInput,
        *,
        workspace: Workspace | None = None,
        config: AgentConfig | None = None,
        on_event: EventCallback | None = None,
    ) -> DetectionReport:
        """执行单个隔离样本的完整漏洞检测流程，返回四要素 ``DetectionReport``。"""
        cfg = config or self.default_config
        t0 = time.perf_counter()
        if on_event is not None:
            on_event(
                AgentEvent(
                    event_type="started",
                    idx=agent_input.idx,
                    step=0,
                    payload={"mode": cfg.mode, "repo_name": agent_input.repo_name},
                )
            )

        recorder = ToolTraceRecorder(sample_idx=agent_input.idx, on_event=on_event)
        budget = ToolBudget(
            max_tool_calls=cfg.max_tool_calls,
            max_read_chars=cfg.max_read_chars,
        )
        compiled = self.build_graph(
            agent_input,
            workspace=workspace,
            config=cfg,
            recorder=recorder,
            budget=budget,
            on_event=on_event,
            start_time=t0,
        )
        init_state: AgentState = {
            "messages": [],
            "step": 0,
            "stop_reason": "sufficient_evidence",
            "knowledge_record": None,
            "pending_tool_outputs": [],
            "verdict": None,
            "report": None,
        }
        final_state = compiled.invoke(init_state)
        report = final_state["report"]
        assert report is not None
        return report
