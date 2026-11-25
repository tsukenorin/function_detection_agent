"""纯客观证据摘要压缩子代理（EvidenceCompactorSubAgent）。

职责边界（严格中立，不越权代判）：
- 主代理（``VulnerabilityDetectionAgent``）全权负责漏洞 0/1 裁决与防御充分性判断；
- 本子代理运行在隔离短上下文中（固定 ``reasoning_effort="low"``），只做**事实抽取与原代码行摘录**：
  1. 提取 1~4 条带精确 ``path:line`` 的纯客观事实（返回类型、宏值、数组容量、判断条件表达式）；
  2. 原样摘录最关键的 1~5 行 C/C++ 原代码（``key_code_lines``），避免有损压缩丢失细节；
  3. 仅标注信息查全状态（``sufficient`` / ``needs_more_hops`` / ``irrelevant``），
     **严禁输出“安全/已充分防护/存在漏洞”等主观定性结论**；
  4. 主动过滤与当前审计焦点无关的旁支代码，防止旁支瑕疵污染主代理。
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

COMPACTOR_SYSTEM_PROMPT = """你是漏洞检测主代理派遣的【纯客观证据摘录子代理（Neutral Evidence Compactor）】。
你的唯一任务是把单次仓库/调用链工具的冗长原始输出，压缩为带精确行号的**纯客观事实 + 关键原代码行摘录**。

【绝对不可越界的规则】
1. **严禁代做安全定性**：绝对不要替主代理下结论说“因此安全”、“防护完整”、“没有漏洞”或“存在漏洞”。你只陈述代码中客观写了什么（例如：`av_parsers.c:4686 返回类型 u32`、`psd.c:716 row_size = columns * packet_size`、`chain.c:43 包含 if (n < 0 || n >= 64) return -1;`）。
2. **必须保留关键原代码行（`key_code_lines`）**：把与主代理核验焦点直接相关的 1~5 行 C/C++ 代码原样抄录（附带 `path:line:` 前缀），防止摘要丢失 `<` 与 `<=`、类型强转、变量有符号性等关键细节。
3. **主动过滤旁支噪声**：对于与当前核验焦点无关的其他函数、注释、日志或历史代码风格问题，直接忽略并在 `filtered_noise_note` 一句话说明，绝不写入 `verified_facts`。
4. **客观标注信息完备度（`evidence_completeness`）**：
   - `sufficient`：本次输出已包含主代理想查的符号定义/类型/宏值/调用关系；此时 `next_frontier_symbols` 必须为 `[]`。
   - `needs_more_hops`：仅拿到符号名或转发包装层，还需展开下一跳才能看到实际实现；在 `next_frontier_symbols` 中给出最多 2 个精确符号名。
   - `irrelevant`：未命中或返回报错/无关内容；`next_frontier_symbols` 为 `[]`。

严格只输出一个合法 JSON 对象（不要加额外文字）：
{
  "target": "被查证的符号或文件路径",
  "verified_facts": ["path:line 客观事实1（不含安全定性）", "..."],
  "key_code_lines": ["path:line: 原样 C/C++ 关键代码行", "..."],
  "evidence_completeness": "sufficient" | "needs_more_hops" | "irrelevant",
  "next_frontier_symbols": ["仅在 needs_more_hops 时填写，最多2个符号名"],
  "filtered_noise_note": "简述过滤了哪些无关上下文（无则留空）"
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
                if isinstance(obj, dict) and (
                    "verified_facts" in obj
                    or "key_code_lines" in obj
                    or "evidence_completeness" in obj
                ):
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
        for list_field in ("verified_facts", "key_code_lines", "next_frontier_symbols"):
            if isinstance(data.get(list_field), str):
                data[list_field] = [data[list_field]]
        # 强制保持中立，防止模型偶发带出主观 guard_status
        data["guard_status"] = "neutral"
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
    """运行在隔离短上下文中的纯客观证据摘要子代理（固定 ``reasoning_effort="low"``）。"""

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

        if raw_output.startswith("ERROR:") or raw_len < self.min_chars:
            completeness = "irrelevant" if raw_output.startswith("ERROR:") else "sufficient"
            snippet = raw_output.strip()[:500]
            summary = EvidenceSummary(
                tool_name=tool_name,
                target=target,
                verified_facts=[snippet] if snippet else [],
                key_code_lines=[],
                evidence_completeness=completeness,  # type: ignore[arg-type]
                guard_status="neutral",
                next_frontier_symbols=[],
                filtered_noise_note="短输出/错误直通，未触发子代理压缩",
                raw_chars=raw_len,
            )
            summary.compacted_chars = len(summary.to_compact_message())
            return summary

        clipped_raw = raw_output[: self.max_raw_input_chars]
        if raw_len > self.max_raw_input_chars:
            clipped_raw += f"\n...[原始输出截断，原长 {raw_len} 字符]"

        user_prompt = (
            f"## 主代理当前查证需求\n"
            f"- 待测目标函数：{function_name or '未知'}\n"
            f"- 当前核验焦点：{(audit_focus or '核验目标符号定义、类型、常量或前置条件')[:500]}\n"
            f"- 本次工具调用：`{tool_name}` args={json.dumps(tool_args, ensure_ascii=False)}\n\n"
            f"## 工具原始输出（{raw_len} 字符）\n"
            f"```\n{clipped_raw}\n```\n\n"
            f"请严格按纯客观摘录规则输出单一 JSON（禁止替主代理做安全定性）。"
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

        fallback = EvidenceSummary(
            tool_name=tool_name,
            target=target,
            verified_facts=[clipped_raw[:500].replace("\n", " ")],
            key_code_lines=[],
            evidence_completeness="needs_more_hops",
            guard_status="neutral",
            next_frontier_symbols=[],
            filtered_noise_note="子代理摘要解析回退，保留前 500 字符摘录",
            raw_chars=raw_len,
        )
        fallback.compacted_chars = len(fallback.to_compact_message())
        return fallback
