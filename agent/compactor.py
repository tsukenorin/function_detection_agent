"""证据摘要压缩子代理（EvidenceCompactorSubAgent）。

职责定位（多代理分工）：
- 主代理（``VulnerabilityDetectionAgent``）负责全局假设规划、决定渐进式调用链展开方向
  与最终 0/1 裁决，必须保持主上下文高度干净；
- 本子代理在完全隔离的单轮上下文中运行（固定 ``reasoning_effort="low"``、短输出上限），
  消费单次仓库/Tree-sitter 工具返回的大段原始输出，提纯出：
  1. 2~4 条带精确 ``path:line`` 的核心核验事实；
  2. 明确的守卫状态（``guards_present`` / ``guards_missing`` / ``inconclusive`` / ``irrelevant``）；
  3. 若仍 ``inconclusive``，列出最多 2 个值得下一跳继续展开的前沿符号（``next_frontier_symbols``）；
  4. 显式丢弃与核心审计焦点无关的旁支代码或无关风格问题，从源头防止“旁支瑕疵误报（P-V）”
     与“多跳上下文稀释（P-B）”。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .model_factory import build_chat_model
from .schema import AgentConfig, EvidenceSummary

__all__ = ["COMPACTOR_SYSTEM_PROMPT", "EvidenceCompactorSubAgent"]

COMPACTOR_SYSTEM_PROMPT = """你是漏洞检测主代理派遣的【证据摘要压缩子代理（Evidence Compactor）】。
你的唯一任务是阅读单次代码仓库/调用链工具的原始输出，将其提纯为极简、精确、零噪声的结构化证据 JSON，供主代理决策。

【必须遵守的提纯原则】
1. **紧扣核心审计焦点**：只保留直接回答主代理疑问的事实（例如被调函数内部是否检查边界/空指针/溢出、结构体字段类型与数组长度、宏常量值、调用方是否已做前置过滤）。
2. **严禁引入旁支噪声**：如果工具输出里附带了其他不相关函数、历史代码风格问题或与当前核心假设无关的外围空指针，必须**主动过滤丢弃**并在 `filtered_noise_note` 一句话说明，绝不把无关瑕疵写进 `verified_facts`。
3. **指导渐进式多跳展开**：如果当前证据已足以确认守卫存在（`guards_present`）或缺失（`guards_missing`），`next_frontier_symbols` 必须为空数组 `[]`；仅当状态为 `inconclusive` 且原始输出中存在更深层的决定性调用方/被调用方时，才列出最多 2 个符号名。
4. **严格引用精确位置**：每条事实尽量以 `path:line` 开头，单条不超过 80 字，总数 1~4 条。

严格只输出一个合法 JSON 对象（不要加额外说明）：
{
  "target": "被查证的符号或文件路径",
  "verified_facts": ["path:line 具体核验事实1", "..."],
  "guard_status": "guards_present" | "guards_missing" | "inconclusive" | "irrelevant",
  "next_frontier_symbols": ["仅在 inconclusive 时填写，最多2个符号名"],
  "filtered_noise_note": "简述丢弃了哪些无关上下文（无则留空）"
}"""


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def _parse_summary_json(
    raw_text: str,
    *,
    tool_name: str,
    default_target: str,
    raw_chars: int,
) -> EvidenceSummary | None:
    if not raw_text.strip():
        return None
    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.extend(reversed(fenced))
    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    decoder = json.JSONDecoder()
    for idx in range(len(raw_text) - 1, -1, -1):
        if raw_text[idx] == "{":
            try:
                obj, _ = decoder.raw_decode(raw_text[idx:])
                if isinstance(obj, dict) and ("verified_facts" in obj or "guard_status" in obj):
                    candidates.append(json.dumps(obj, ensure_ascii=False))
                    break
            except json.JSONDecodeError:
                continue

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("verified_facts"), str):
            data["verified_facts"] = [data["verified_facts"]]
        if isinstance(data.get("next_frontier_symbols"), str):
            data["next_frontier_symbols"] = [data["next_frontier_symbols"]]
        data.setdefault("target", default_target)
        data["tool_name"] = tool_name
        data["raw_chars"] = raw_chars
        try:
            summary = EvidenceSummary.model_validate(data)
            summary.compacted_chars = len(summary.to_compact_message())
            return summary
        except ValidationError:
            continue
    return None


def _target_from_args(tool_args: dict[str, Any]) -> str:
    for key in ("symbol_name", "path", "pattern"):
        val = tool_args.get(key)
        if val:
            return str(val)
    return ""


class EvidenceCompactorSubAgent:
    """运行在隔离短上下文中的证据摘要压缩子代理（固定 ``reasoning_effort="low"``）。"""

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        config: AgentConfig | None = None,
        min_chars: int = 600,
        max_raw_input_chars: int = 10000,
    ) -> None:
        self._model = model
        self._config = config or AgentConfig.improved()
        self.min_chars = max(0, int(min_chars))
        self.max_raw_input_chars = max(1000, int(max_raw_input_chars))

    def _get_model(self) -> BaseChatModel:
        if self._model is not None:
            return self._model
        sub_cfg = self._config.model_copy(
            update={
                "reasoning_effort": "low",
                "max_tokens": 768,
                "temperature": 0.0,
            }
        )
        self._model = build_chat_model(sub_cfg)
        return self._model

    def compact(
        self,
        *,
        function_name: str,
        audit_focus: str,
        tool_name: str,
        tool_args: dict[str, Any],
        raw_output: str,
    ) -> EvidenceSummary:
        target = _target_from_args(tool_args)
        raw_len = len(raw_output)

        # 错误信息或本就很短的输出无需额外消耗 LLM 压缩，直接做零开销结构化透传
        if raw_output.startswith("ERROR:") or raw_len < self.min_chars:
            status = "irrelevant" if raw_output.startswith("ERROR:") else "inconclusive"
            fact = raw_output.strip()[:400]
            summary = EvidenceSummary(
                tool_name=tool_name,
                target=target,
                verified_facts=[fact] if fact else [],
                guard_status=status,  # type: ignore[arg-type]
                next_frontier_symbols=[],
                filtered_noise_note="短输出/错误直通，未触发子代理 LLM 压缩",
                raw_chars=raw_len,
            )
            summary.compacted_chars = len(summary.to_compact_message())
            return summary

        clipped_raw = raw_output[: self.max_raw_input_chars]
        if raw_len > self.max_raw_input_chars:
            clipped_raw += f"\n...[原始输出截断，原长 {raw_len} 字符]"

        user_prompt = (
            f"## 主代理当前审计任务\n"
            f"- 待测目标函数：{function_name or '未知'}\n"
            f"- 当前核验焦点：{(audit_focus or '核验边界/空指针/溢出守卫是否存在')[:500]}\n"
            f"- 本次工具调用：`{tool_name}` args={json.dumps(tool_args, ensure_ascii=False)}\n\n"
            f"## 工具原始输出（{raw_len} 字符）\n"
            f"```\n{clipped_raw}\n```\n\n"
            f"请按规则提纯为单一紧凑 JSON 证据对象。"
        )

        try:
            resp = self._get_model().invoke(
                [
                    SystemMessage(content=COMPACTOR_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            parsed = _parse_summary_json(
                _extract_text(resp.content),
                tool_name=tool_name,
                default_target=target,
                raw_chars=raw_len,
            )
            if parsed is not None:
                return parsed
        except Exception:
            pass

        # 兜底：若子代理异常或 JSON 解析失败，返回安全的确定性截断摘要，绝不中断主流程
        fallback = EvidenceSummary(
            tool_name=tool_name,
            target=target,
            verified_facts=[clipped_raw[:500].replace("\n", " ")],
            guard_status="inconclusive",
            next_frontier_symbols=[],
            filtered_noise_note="子代理摘要解析回退，仅保留前 500 字符摘录",
            raw_chars=raw_len,
        )
        fallback.compacted_chars = len(fallback.to_compact_message())
        return fallback
