"""仓库工具调用的预算控制与系统级审计日志采集。

对应系统设计原则“约束写在代码里，不写在提示词里”：
- ``ToolBudget`` 硬性限制工具调用次数与累计返回字符数；
- ``ToolTraceRecorder`` 自动记录每一次仓库工具调用，生成可追溯的
  ``RepoReadRecord``，绝不依赖 LLM 自述读取了哪些文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.interfaces import EventCallback
from agent.schema import AgentEvent, RepoReadRecord

__all__ = ["ToolBudget", "ToolTraceRecorder", "BudgetExceededError"]


class BudgetExceededError(RuntimeError):
    """仓库工具调用次数或读取字符数达到预算上限。"""


@dataclass
class ToolBudget:
    """单样本仓库探索预算控制器。"""

    max_tool_calls: int = 6
    max_read_chars: int = 24000
    calls_used: int = 0
    chars_used: int = 0

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_tool_calls - self.calls_used)

    @property
    def chars_remaining(self) -> int:
        return max(0, self.max_read_chars - self.chars_used)

    @property
    def exhausted(self) -> bool:
        return self.calls_used >= self.max_tool_calls or self.chars_used >= self.max_read_chars

    def consume_call(self) -> None:
        if self.calls_used >= self.max_tool_calls:
            raise BudgetExceededError(
                f"仓库工具调用次数已达上限（{self.max_tool_calls}），请基于已有证据做出最终判定"
            )
        self.calls_used += 1

    def trim_and_account(self, output: str) -> str:
        remaining = self.chars_remaining
        if remaining <= 0:
            return (
                f"ERROR: 仓库读取字符预算已耗尽（上限 {self.max_read_chars} 字符），"
                "请基于已有证据完成判定。"
            )
        if len(output) > remaining:
            marker = "\n...[因仓库读取字符预算上限被系统截断]"
            keep = max(0, remaining - len(marker))
            trimmed = output[:keep] + marker
            self.chars_used += len(trimmed)
            return trimmed
        self.chars_used += len(output)
        return output


@dataclass
class ToolTraceRecorder:
    """自动采集所有仓库工具调用记录，并可选向 UI / CLI 推送实时事件。"""

    sample_idx: int = 0
    on_event: EventCallback | None = None
    current_step: int = 0
    records: list[RepoReadRecord] = field(default_factory=list)

    def record_call_start(self, tool_name: str, args: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(
                AgentEvent(
                    event_type="tool_call",
                    idx=self.sample_idx,
                    step=self.current_step,
                    payload={"tool_name": tool_name, "args": args},
                )
            )

    def record_call_end(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        path: str | None,
        ok: bool,
        summary: str,
        chars_returned: int,
        latency_ms: float,
    ) -> RepoReadRecord:
        rec = RepoReadRecord(
            tool_name=tool_name,
            args=dict(args),
            path=path,
            ok=ok,
            summary=summary,
            chars_returned=chars_returned,
            step=self.current_step,
            latency_ms=round(latency_ms, 2),
        )
        self.records.append(rec)
        if self.on_event is not None:
            self.on_event(
                AgentEvent(
                    event_type="tool_result",
                    idx=self.sample_idx,
                    step=self.current_step,
                    payload=rec.model_dump(),
                )
            )
        return rec
