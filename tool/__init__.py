"""工具包。

分层：

- :mod:`tool.workspace` —— 纯 Python 的文件系统原语，无框架依赖，可单测；
- :mod:`tool.file_tools` —— 包装成 LangChain 工具，供 Agent 调用。

后续的 ``grep``、``read_symbol``（Tree-sitter 调用关系）等工具
应当复用 ``Workspace._resolve`` 的沙箱逻辑，不要各自解析路径。
"""

from .workspace import (
    BinaryFile,
    NotFound,
    NotADirectory,
    NotAFile,
    PathViolation,
    ReadOnlyViolation,
    Workspace,
    WorkspaceError,
)

__all__ = [
    "Workspace",
    "WorkspaceError",
    "PathViolation",
    "NotFound",
    "NotAFile",
    "NotADirectory",
    "BinaryFile",
    "ReadOnlyViolation",
]


def __getattr__(name: str):
    """让 LangChain 工具与 trace 适配按需导入，避免没装 langchain 时核心层不可用。"""
    if name in {"build_file_tools", "build_repo_tools", "FILE_TOOL_NAMES", "REPO_TOOL_NAMES"}:
        from . import file_tools

        return getattr(file_tools, name)
    if name in {"ToolBudget", "ToolTraceRecorder", "BudgetExceededError"}:
        from . import trace

        return getattr(trace, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
