"""Agent 编排层提示词与格式化工具（区分 Baseline 与 Improved 策略）。"""

from __future__ import annotations

from agent.schema import AgentInput, KnowledgeRetrieveRecord

__all__ = [
    "BASELINE_SYSTEM_PROMPT",
    "IMPROVED_SYSTEM_PROMPT",
    "SELF_CHECK_PROMPT",
    "JSON_REPAIR_PROMPT",
    "format_numbered_func",
    "format_knowledge_context",
    "build_baseline_user_prompt",
    "build_improved_user_prompt",
]

BASELINE_SYSTEM_PROMPT = """你是一位 C/C++ 代码安全审计助手。
你的任务是根据提供的单个 C/C++ 函数代码，判断该函数是否存在安全漏洞。

请严格输出一个合法的 JSON 对象，不要包含额外解释文本，字段格式如下：
{
  "prediction": 0 或 1（1 表示存在漏洞，0 表示无漏洞）,
  "confidence": "high" | "medium" | "low",
  "cwe_candidates": ["CWE-xxx"],
  "reason": "简要说明判定理由，尽量引用具体行号或变量",
  "evidence_cited": ["行号或关键表达式"],
  "guard_assessment": "简述关键操作是否存在边界或空指针检查"
}"""

IMPROVED_SYSTEM_PROMPT = """你是一位资深的 C/C++ 软件安全漏洞审计智能体（Vulnerability Detection Agent）。
你运行在一个受控的、单版本只读代码仓库沙箱环境中，遵循“知识检索 -> 初步分析 -> 按需查仓库 -> 严谨核验 -> 结构化判定”的工作流。

【核心审计原则】
1. **区分“真正缺失防御（prediction=1）”与“已有有效防御（prediction=0）”**：
   - 待测函数可能是漏洞版本，也可能是已经打过安全补丁的修复版本！
   - 严禁把所有 C/C++ 指针/数组/算术代码一概判为有漏洞；必须逐条核实危险操作之前是否已存在有效的边界、空指针、溢出、除零、类型/维度或生命周期检查。
   - 仅当存在具体、可达、且缺乏足够防御检查的崩溃或利用路径（如越界读写、空指针解引用、整数溢出/下溢、UAF/Double-Free、除零、未校验张量维度/索引、未初始化内存、无限循环/栈溢出等）时，才判定 `prediction = 1`。
   - 如果危险操作已被显式守卫语句（例如 `if (...) return/break/error`、`OP_REQUIRES`、长度/索引上界与下界校验、非空校验、安全溢出检查函数等）有效拦截，应判定 `prediction = 0`。
2. **善用历史 CVE 成对案例**：
   - 参考“历史相似 CVE 案例”中的修复差异（`fix_pattern`），重点观察同类漏洞通常缺失哪一种检查、以及修复后通常新增了哪一种守卫条件。
3. **证据驱动的按需仓库探索**：
   - 如果仅凭函数本身已经能确凿看到缺失检查或完整检查，请直接停止探索并给出结论，不要做无意义的工具调用。
   - 如果判断取决于**外部被调用函数、宏定义、结构体字段类型/有符号性、数组大小或调用方约束**，请优先调用 `find_symbol`、`grep`、`read_file` 等工具查证，拿到关键证据后立即停止探索。
   - 工具调用次数和字符数均有硬性预算上限，每次调用都要目标明确。

【最终输出格式】
当你收集完足够证据、不再需要调用工具时，请输出一个合法 JSON 对象（可用 ```json 包裹）：
{
  "prediction": 0 或 1,
  "confidence": "high" | "medium" | "low",
  "cwe_candidates": ["CWE-xxx"],
  "reason": "结合具体行号、变量与仓库查证证据，清晰说明为何有洞或为何已安全防护",
  "evidence_cited": ["函数第 X 行 ...", "仓库文件 path:line ..."],
  "guard_assessment": "明确列出关键危险点以及对应检查是否存在、是否覆盖边界条件（如负数、0、上界、溢出、空指针）"
}"""

SELF_CHECK_PROMPT = """在给出最终裁决前，请执行一次严格的反事实自检（Self-Verification），并始终以**待测函数自身代码行与历史 CVE 修复差异**为主锚点：
1. **防误报核验（避免将已修复函数误判为 1）**：你刚才怀疑的漏洞路径，是否已经被待测函数内某条 `if` / `OP_REQUIRES` / 宏 / 或已查证的外部被调用函数所拦截？不要因为“理论上可能出错”或无关代码风格问题而判 1。
2. **防漏报核验（避免将漏洞函数误判为 0）**：请再仔细核对待测函数内最危险的 1~2 处内存访问、索引计算、长度/类型转换、指针解引用、资源释放或除数，是否存在细微遗漏（例如先解引用后判空、只查上界没查负数、`>` 与 `>=` 差一错误、乘法/加法溢出、缺失显式类型转换、返回值未判空）？切勿被仓库中无关上下文冲淡函数本身确凿的缺失守卫。
3. **仅基于已验证事实裁决**：综合上述核验，直接输出最终的单一合法 JSON 对象（字段：`prediction`, `confidence`, `cwe_candidates`, `reason`, `evidence_cited`, `guard_assessment`），不要再调用工具。"""

JSON_REPAIR_PROMPT = """你刚才的回复未能解析为合法的目标 JSON 格式。
请严格仅输出一个 JSON 对象，不要附加任何说明文字，必须包含以下字段：
{
  "prediction": 0 或 1（整数，不可为字符串或布尔值）,
  "confidence": "high" | "medium" | "low",
  "cwe_candidates": ["CWE-xxx"],
  "reason": "简明判定理由",
  "evidence_cited": ["关键证据"],
  "guard_assessment": "防御检查核验说明"
}"""


def format_numbered_func(func_code: str, *, max_chars: int = 18000) -> str:
    """为待测函数代码添加 1-based 行号；超长函数采用头尾保留截断避免丢失末尾逻辑。"""
    lines = func_code.splitlines()
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines, 1))
    if len(numbered) <= max_chars:
        return numbered

    head_budget = int(max_chars * 0.65)
    tail_budget = max_chars - head_budget - 80
    head = numbered[:head_budget]
    tail = numbered[-tail_budget:] if tail_budget > 0 else ""
    omitted = len(numbered) - len(head) - len(tail)
    return f"{head}\n\n... [中间省略 {omitted} 字符以适应上下文窗口] ...\n\n{tail}"


def format_knowledge_context(record: KnowledgeRetrieveRecord | None) -> str:
    """将 RAG 检索出的历史 CVE 成对案例格式化为精炼提示上下文。"""
    if record is None or not record.hits:
        return "（未检索到相关历史 CVE 案例）"

    blocks: list[str] = []
    for i, hit in enumerate(record.hits, 1):
        cwe_str = ", ".join(hit.cwe) if hit.cwe else "未知 CWE"
        header = (
            f"### 历史案例 {i}：{hit.cve or hit.case_id} "
            f"[{cwe_str}]（项目：{hit.project or 'unknown'}，相关度：{hit.score:.2f}）"
        )
        parts = [header]
        if hit.cve_desc:
            parts.append(f"- 漏洞描述：{hit.cve_desc}")
        if hit.commit_message:
            first_line = hit.commit_message.splitlines()[0]
            parts.append(f"- 修复摘要：{first_line}")
        if hit.fix_pattern:
            parts.append(f"- 历史修复差异（- 为漏洞前，+ 为修复后）：\n```diff\n{hit.fix_pattern}\n```")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def build_baseline_user_prompt(
    agent_input: AgentInput,
    *,
    max_func_chars: int = 18000,
) -> str:
    numbered = format_numbered_func(agent_input.func, max_chars=max_func_chars)
    func_title = agent_input.function_name or "未知函数"
    return (
        f"请分析以下 C/C++ 函数是否存在安全漏洞：\n"
        f"- 样本编号：{agent_input.idx}\n"
        f"- 函数名：{func_title}\n\n"
        f"```cpp\n{numbered}\n```\n\n"
        f"请直接输出严格符合要求的 JSON 判定结果。"
    )


def build_improved_user_prompt(
    agent_input: AgentInput,
    *,
    knowledge_record: KnowledgeRetrieveRecord | None = None,
    has_repo_tools: bool = False,
    evidence_gated_tools: bool = True,
    max_func_chars: int = 18000,
) -> str:
    numbered = format_numbered_func(agent_input.func, max_chars=max_func_chars)
    func_title = agent_input.function_name or "未知函数"
    kb_section = format_knowledge_context(knowledge_record)

    if not has_repo_tools:
        tool_hint = "当前未挂载仓库工具，请基于函数代码与历史 CVE 知识审慎分析并输出最终 JSON 判定。"
    elif evidence_gated_tools:
        tool_hint = (
            "当前已挂载单版本只读代码仓库工具（`find_symbol`、`grep`、`read_file` 等），执行**严格证据门控策略**：\n"
            "  - **默认 0 次调用**：先仅看待测函数代码与上方历史 CVE 修复差异。若函数内部已经能直接看到确凿漏洞（如先解引用后判空、返回值未判空、缺失类型转换/边界判断）或完整的局部防御检查，请**不要调用任何仓库工具**，直接输出 JSON 判定！\n"
            "  - **外科手术式按需调用（最多 1~2 次）**：仅当 0/1 裁决真正取决于某个外部符号（被调用函数内部是否做校验、结构体字段有符号性/数组大小、宏定义）时，才定向调用 `find_symbol` 或短窗口 `read_file`（≤60 行）核实该单一符号，拿到答案立即停止探索。"
        )
    else:
        tool_hint = (
            "当前已挂载该版本的代码仓库只读沙箱工具（`find_symbol`、`grep`、`read_file`、`find_files`、`list_dir`、`stat`）。"
            "若需确认被调用函数、宏、结构体或调用方是否进行了边界/空指针/溢出校验，请按需调用工具；若函数内部证据已充分，请直接输出最终 JSON 判定。"
        )

    return (
        f"## 待检测函数信息\n"
        f"- 样本编号：{agent_input.idx}\n"
        f"- 所属项目：{agent_input.repo_name or '未知'}\n"
        f"- 函数名：{func_title}\n"
        f"- 上下文工具状态：{tool_hint}\n\n"
        f"## 历史相似 CVE 领域知识（RAG 检索结果）\n"
        f"{kb_section}\n\n"
        f"## 待测 C/C++ 函数源码（带 1-based 行号）\n"
        f"```cpp\n{numbered}\n```\n\n"
        f"请先结合历史 CVE 修复模式分析该函数是否存在未受保护的漏洞触发路径，"
        f"并核验现有防御检查；如证据不足请调用仓库工具，如证据充分请直接输出最终 JSON 判定。"
    )
