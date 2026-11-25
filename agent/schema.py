"""核心数据契约与领域 Schema 定义。

本模块是整个系统的类型契约中心，显式固化三条架构红线：

1. **隔离边界**：``RawSample``（含 ``target``）仅存在于加载器之前；
   进入 Agent 层的 ``AgentInput`` 使用 ``extra="forbid"``，禁止携带
   ``target`` / ``label`` / ``commit_message`` / ``git_diff`` 等任何泄露字段。
2. **证据可追溯**：``AnalysisTrace`` 中的知识检索记录与仓库读取记录
   均由系统自动采集生成，绝不依赖 LLM 事后自述。
3. **四要素报告与评测兼容**：``DetectionReport`` 严格包含 PPT 要求的
   四项核心要素，并通过 ``to_result_entry()`` 输出与 ``evaluation.py``
   完全兼容的 ``{"idx": int, "prediction": 0 | 1}``。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "RawSample",
    "RepoRef",
    "AgentInput",
    "KnowledgeHit",
    "KnowledgeRetrieveRecord",
    "RepoReadRecord",
    "AnalysisTrace",
    "AgentVerdict",
    "DetectionReport",
    "AgentEvent",
    "AgentConfig",
]


def _ensure_strict_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} 必须是整数，收到 {type(value).__name__}")
    return value


class RawSample(BaseModel):
    """外部输入（单样本表单或 .jsonl 批量上传）的原始样本结构。

    ``target`` 仅在加载器阶段用于确定目标版本快照（1 -> 父提交，0 -> 提交后），
    随后会被消费掉，不会传递到 Agent 层。
    """

    model_config = ConfigDict(extra="ignore")

    idx: int = Field(..., description="样本唯一编号")
    commit_url: str = Field(..., min_length=1, description="代码仓库修复提交链接")
    func: str = Field(..., min_length=1, description="待检测的 C/C++ 函数源代码")
    target: Literal[0, 1] | None = Field(
        default=None,
        description="样本标签：1=漏洞版（取父提交），0=修复版（取提交后），None=无标签单样本",
    )

    @field_validator("idx", mode="before")
    @classmethod
    def _validate_idx(cls, v: Any) -> int:
        return _ensure_strict_int(v, "idx")

    @field_validator("target", mode="before")
    @classmethod
    def _validate_target(cls, v: Any) -> int | None:
        if v is None:
            return None
        val = _ensure_strict_int(v, "target")
        if val not in (0, 1):
            raise ValueError("target 必须为 0、1 或 None")
        return val


class RepoRef(BaseModel):
    """样本加载器解析后的仓库版本引用（仅传给 RepoFS 层，不暴露给 LLM）。"""

    model_config = ConfigDict(frozen=True)

    host: str = Field(default="github.com", description="代码托管主机")
    owner: str = Field(..., min_length=1, description="仓库所有者")
    repo: str = Field(..., min_length=1, description="仓库名称")
    commit_sha: str = Field(..., min_length=6, description="原始修复提交 SHA")
    allowed_sha: str = Field(
        ...,
        min_length=6,
        description="允许访问的唯一版本 SHA（漏洞版为父提交 SHA，修复版为提交后 SHA）",
    )
    version_role: Literal["parent", "commit"] = Field(
        ..., description="版本角色（仅供系统内部审计，严禁暴露给 Agent）"
    )
    original_url: str = Field(..., min_length=1, description="原始 commit_url")

    @property
    def repo_full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


class AgentInput(BaseModel):
    """经 SampleLoader 裁剪后进入 Agent 层的最小化输入。

    ``extra="forbid"`` 从结构上阻止 ``target`` / ``label`` / ``commit_message`` /
    ``git_diff`` 等敏感信息被误传进 Agent。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    idx: int = Field(..., description="样本编号")
    func: str = Field(..., min_length=1, description="待检测 C/C++ 函数代码")
    repo_name: str = Field(default="", description="项目名（如 owner/repo），不含版本标签")
    function_name: str | None = Field(default=None, description="从函数签名提取的函数名")
    snapshot_token: str = Field(
        default="",
        description="不透明快照标识（仅用于日志关联，不暴露 parent/fixed 语义）",
    )

    @field_validator("idx", mode="before")
    @classmethod
    def _validate_idx(cls, v: Any) -> int:
        return _ensure_strict_int(v, "idx")


class KnowledgeHit(BaseModel):
    """领域知识库 RAG 返回的单条历史 CVE 案例。"""

    case_id: str = Field(..., description="知识库案例唯一标识")
    cve: str | None = Field(default=None, description="CVE 编号")
    cwe: list[str] = Field(default_factory=list, description="CWE 类型列表")
    project: str | None = Field(default=None, description="所属开源项目")
    score: float = Field(default=0.0, description="检索相关性分数")
    cve_desc: str = Field(default="", description="漏洞披露描述")
    commit_message: str = Field(default="", description="修复提交说明")
    fix_pattern: str = Field(
        default="",
        description="成对历史案例提炼出的修复差异摘要（展示了漏洞与修复的关键区别）",
    )
    vuln_snippet: str = Field(default="", description="历史漏洞版关键片段")
    fixed_snippet: str = Field(default="", description="历史修复版关键片段")


class KnowledgeRetrieveRecord(BaseModel):
    """一次领域知识库检索的系统日志记录。"""

    query: str = Field(..., description="检索查询摘要")
    top_k: int = Field(default=3, ge=0, description="请求返回的案例数")
    hits: list[KnowledgeHit] = Field(default_factory=list, description="命中的历史案例")
    latency_ms: float = Field(default=0.0, ge=0.0, description="检索耗时（毫秒）")


class RepoReadRecord(BaseModel):
    """一次代码仓库工具调用的系统审计日志。"""

    tool_name: str = Field(..., description="调用的工具名，如 grep / read_file / find_symbol")
    args: dict[str, Any] = Field(default_factory=dict, description="工具入参")
    path: str | None = Field(default=None, description="涉及的相对路径（如有）")
    ok: bool = Field(default=True, description="是否执行成功")
    summary: str = Field(default="", description="返回结果摘要")
    chars_returned: int = Field(default=0, ge=0, description="返回给 Agent 的字符数")
    step: int = Field(default=0, ge=0, description="所属推理步数")
    latency_ms: float = Field(default=0.0, ge=0.0, description="工具执行耗时（毫秒）")


class AnalysisTrace(BaseModel):
    """系统自动采集的完整分析过程记录（对应报告第四要素）。"""

    knowledge_records: list[KnowledgeRetrieveRecord] = Field(
        default_factory=list, description="历史漏洞知识库检索记录"
    )
    repo_records: list[RepoReadRecord] = Field(
        default_factory=list, description="代码仓库按需读取记录"
    )
    tool_calls_count: int = Field(default=0, ge=0, description="实际仓库工具调用总次数")
    total_read_chars: int = Field(default=0, ge=0, description="累计读取字符数")
    steps_taken: int = Field(default=0, ge=0, description="Agent 推理轮次")
    stop_reason: Literal[
        "sufficient_evidence",
        "budget_exhausted",
        "no_repo_available",
        "error",
    ] = Field(default="sufficient_evidence", description="终止原因")


class AgentVerdict(BaseModel):
    """Agent 最终输出的结构化判定结果。"""

    prediction: Literal[0, 1] = Field(..., description="漏洞判定：1=有漏洞，0=无漏洞")
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium", description="判定置信度"
    )
    cwe_candidates: list[str] = Field(
        default_factory=list, description="疑似 CWE 编号（无漏洞时为空列表）"
    )
    reason: str = Field(..., min_length=1, description="详细判定理由")
    evidence_cited: list[str] = Field(
        default_factory=list, description="引用的关键代码行、符号或仓库文件证据"
    )
    guard_assessment: str = Field(
        default="", description="对函数中已有/缺失防御检查（边界、空指针、溢出等）的核验说明"
    )

    @field_validator("prediction", mode="before")
    @classmethod
    def _validate_prediction(cls, v: Any) -> int:
        val = _ensure_strict_int(v, "prediction")
        if val not in (0, 1):
            raise ValueError("prediction 必须为 0 或 1")
        return val


class DetectionReport(BaseModel):
    """面向用户与评测系统的四要素检测报告。

    四项必需要素（PPT 第 7 页）：
    1. ``idx`` —— 检测函数编号
    2. ``prediction`` / ``label_text`` —— 检测结果（有漏洞 / 无漏洞）
    3. ``reason`` —— 判定理由
    4. ``analysis_record`` —— 分析记录（知识检索记录 + 仓库读取记录）
    """

    idx: int = Field(..., description="检测函数 idx")
    prediction: Literal[0, 1] = Field(..., description="数值判定结果：1=有漏洞，0=无漏洞")
    label_text: Literal["有漏洞", "无漏洞"] = Field(..., description="中文判定标签")
    reason: str = Field(..., min_length=1, description="判定理由")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    cwe_candidates: list[str] = Field(default_factory=list)
    evidence_cited: list[str] = Field(default_factory=list)
    guard_assessment: str = Field(default="")
    analysis_record: AnalysisTrace = Field(
        default_factory=AnalysisTrace, description="系统采集的知识检索与仓库读取记录"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="运行元信息")

    @classmethod
    def from_verdict(
        cls,
        idx: int,
        verdict: AgentVerdict,
        trace: AnalysisTrace,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DetectionReport:
        return cls(
            idx=idx,
            prediction=verdict.prediction,
            label_text="有漏洞" if verdict.prediction == 1 else "无漏洞",
            reason=verdict.reason,
            confidence=verdict.confidence,
            cwe_candidates=list(verdict.cwe_candidates),
            evidence_cited=list(verdict.evidence_cited),
            guard_assessment=verdict.guard_assessment,
            analysis_record=trace,
            metadata=dict(metadata or {}),
        )

    def to_result_entry(self) -> dict[str, int]:
        """导出与 ``dataset/evaluation.py`` 兼容的结果条目。"""
        return {"idx": int(self.idx), "prediction": int(self.prediction)}


class AgentEvent(BaseModel):
    """Agent 执行过程中的实时事件（供前端 SSE / CLI 日志订阅）。"""

    event_type: Literal[
        "started",
        "knowledge_retrieved",
        "thought",
        "tool_call",
        "tool_result",
        "self_check",
        "verdict",
        "report_ready",
        "error",
    ]
    idx: int
    step: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class AgentConfig(BaseModel):
    """Agent 编排与实验策略的可配置参数。

    通过切换 ``enable_rag`` / ``enable_repo_tools`` / ``enable_self_check`` / ``mode``，
    可在同一编排框架下公平对比 Baseline 与各项改进措施。
    """

    mode: Literal["baseline", "improved"] = Field(
        default="improved", description="运行策略模式"
    )
    model_name: str = Field(default="deepseek-chat", description="LLM 模型名称")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_steps: int = Field(default=6, ge=1, le=20, description="Agent 最大推理步数")
    max_tool_calls: int = Field(default=6, ge=0, le=30, description="仓库工具调用总预算")
    max_read_chars: int = Field(
        default=24000, ge=1000, description="单样本仓库工具返回总字符预算"
    )
    max_func_chars: int = Field(
        default=18000, ge=1000, description="传入提示词的函数代码最大字符截断上限"
    )
    top_k_knowledge: int = Field(default=3, ge=0, le=10, description="RAG 检索返回案例数")
    enable_rag: bool = Field(default=True, description="是否启用历史 CVE 领域知识库 RAG")
    enable_repo_tools: bool = Field(
        default=True, description="是否启用代码仓库按需上下文挖掘工具"
    )
    enable_self_check: bool = Field(
        default=True, description="是否启用防御检查核验与反事实自检节点"
    )

    @classmethod
    def baseline(cls, **overrides: Any) -> AgentConfig:
        """Baseline 配置：孤立函数直接检测（无 RAG、无仓库工具、无自检节点）。"""
        defaults: dict[str, Any] = {
            "mode": "baseline",
            "max_steps": 2,
            "max_tool_calls": 0,
            "top_k_knowledge": 0,
            "enable_rag": False,
            "enable_repo_tools": False,
            "enable_self_check": False,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def improved(cls, **overrides: Any) -> AgentConfig:
        """Improved 配置：历史 CVE RAG + 按需仓库工具挖掘 + 防御核验自检。"""
        defaults: dict[str, Any] = {
            "mode": "improved",
            "max_steps": 6,
            "max_tool_calls": 6,
            "top_k_knowledge": 3,
            "enable_rag": True,
            "enable_repo_tools": True,
            "enable_self_check": True,
        }
        defaults.update(overrides)
        return cls(**defaults)
