"""把 :class:`tool.workspace.Workspace` 包装成 LangChain 工具（v1）。

这一层刻意做得很薄：

- 不做业务判断（要不要读、读了算不算证据，都是 Agent 的事）；
- 不抛异常给 Agent，而是把错误转成 ``ERROR: ...`` 字符串，
  避免一次越界读直接把 loop 打断；
- 返回 JSON 字符串，字段与 ``Workspace`` 的返回值完全一致，
  这样 trace / 报告可以直接复用同一份结构。

用法::

    from tool import Workspace, build_file_tools

    ws = Workspace(".cache/openssl/<sha>")           # 只读快照
    tools = build_file_tools(ws)
    agent = create_react_agent(model, tools)
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain.tools import tool

from .callgraph import CallGraphAnalyzer, MAX_CALLCHAIN_DEPTH
from .trace import BudgetExceededError, ToolBudget, ToolTraceRecorder
from .workspace import (
    DEFAULT_GREP_MAX_RESULTS,
    DEFAULT_LINE_LIMIT,
    DEFAULT_SYMBOL_MAX_RESULTS,
    MAX_LINE_LIMIT,
    Workspace,
    WorkspaceError,
)

__all__ = [
    "build_file_tools",
    "build_repo_tools",
    "FILE_TOOL_NAMES",
    "REPO_TOOL_NAMES",
]

FILE_TOOL_NAMES = (
    "list_dir",
    "read_file",
    "find_files",
    "stat",
    "write_file",
)

REPO_TOOL_NAMES = (
    "list_dir",
    "read_file",
    "find_files",
    "stat",
    "grep",
    "find_symbol",
    "inspect_call_chain",
    "read_symbol_body",
)


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _err(exc: Exception) -> str:
    return f"ERROR: {exc}"


def build_file_tools(workspace: Workspace) -> list:
    """为给定工作区构建一组文件工具。

    每次构建都会闭包捕获 ``workspace``，因此不同样本（不同 sha）之间
    不会共享实例——这正是版本隔离需要的性质。
    """

    @tool
    def list_dir(
        path: str = ".",
        recursive: bool = False,
        max_depth: int = 2,
    ) -> str:
        """列出工作区中某个目录的内容。

        先用它确定仓库结构，再决定读哪个文件。默认只列一层并跳过隐藏目录。

        Args:
            path: 相对工作区根的目录路径，根目录用 "."。
            recursive: 是否递归列出子目录。
            max_depth: 递归深度，仅在 recursive=True 时生效。
        """
        try:
            return _ok(
                workspace.list_dir(path, recursive=recursive, max_depth=max_depth)
            )
        except WorkspaceError as exc:
            return _err(exc)

    @tool
    def read_file(path: str, offset: int = 1, limit: int = DEFAULT_LINE_LIMIT) -> str:
        """读取工作区内一个文本文件，返回带行号的内容。

        只在需要确认具体实现时使用：例如被调用方到底有没有做长度检查。
        文件较长时用 offset / limit 分页，不要一次拉整个大文件。

        Args:
            path: 相对工作区根的文件路径。
            offset: 起始行号，从 1 开始。
            limit: 最多返回多少行（上限 2000）。
        """
        try:
            return _ok(
                workspace.read_file(path, offset=offset, limit=min(limit, MAX_LINE_LIMIT))
            )
        except WorkspaceError as exc:
            return _err(exc)

    @tool
    def find_files(pattern: str, max_results: int = 200) -> str:
        """按文件名或相对路径查找文件，返回匹配的相对路径列表。

        例：`*.c` 查所有 C 源文件；`src/net/*.h` 查某个子目录下的头文件。
        如果只知道符号名、不知道文件名，应当使用后续提供的 grep 工具。

        Args:
            pattern: 通配模式，支持 `*`、`?` 与目录前缀。
            max_results: 最多返回多少个结果。
        """
        try:
            return _ok(workspace.find_files(pattern, max_results=max_results))
        except WorkspaceError as exc:
            return _err(exc)

    @tool
    def stat(path: str) -> str:
        """查看单个路径的类型、大小与修改时间，用于确认文件是否存在。"""
        try:
            return _ok(workspace.stat(path))
        except WorkspaceError as exc:
            return _err(exc)

    @tool
    def write_file(path: str, content: str) -> str:
        """把文本写入工作区（仅用于报告、笔记等产物）。

        检测阶段的工作区通常是只读快照，调用本工具会返回 ERROR。
        不要用它修改待测代码。
        """
        try:
            return _ok(workspace.write_file(path, content))
        except WorkspaceError as exc:
            return _err(exc)

    return [list_dir, read_file, find_files, stat, write_file]


def build_repo_tools(
    workspace: Workspace,
    *,
    recorder: ToolTraceRecorder | None = None,
    budget: ToolBudget | None = None,
) -> list:
    """构建面向漏洞检测 Agent 的只读、可审计、带预算控制的仓库上下文工具集。

    特点：
    - 仅暴露只读检索/阅读与 Tree-sitter 确定性调用链工具（不暴露 ``write_file``）；
    - 每次调用自动经过 ``budget`` 校验与 ``recorder`` 审计；
    - 闭包绑定单个 ``Workspace`` 与 ``CallGraphAnalyzer``，确保严格的成对版本隔离。
    """
    callgraph = CallGraphAnalyzer(workspace)

    def _run_audited(
        tool_name: str,
        args: dict[str, Any],
        path: str | None,
        fn,
    ) -> str:
        t0 = time.perf_counter()
        if recorder is not None:
            recorder.record_call_start(tool_name, args)

        try:
            if budget is not None:
                budget.consume_call()
            payload = fn()
            raw_out = _ok(payload)
            out = budget.trim_and_account(raw_out) if budget is not None else raw_out
            ok = not out.startswith("ERROR:")
            summary = _summarize_payload(tool_name, payload)
        except (WorkspaceError, BudgetExceededError) as exc:
            out = _err(exc)
            ok = False
            summary = str(exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if recorder is not None:
            recorder.record_call_end(
                tool_name=tool_name,
                args=args,
                path=path,
                ok=ok,
                summary=summary,
                chars_returned=len(out),
                latency_ms=elapsed_ms,
            )
        return out

    @tool
    def list_dir(path: str = ".", recursive: bool = False, max_depth: int = 2) -> str:
        """列出代码仓库快照中指定目录的结构，帮助快速定位源码目录。

        Args:
            path: 相对工作区根目录的路径，根目录使用 "."。
            recursive: 是否递归列出子目录。
            max_depth: 递归深度上限（仅 recursive=True 时生效）。
        """
        args = {"path": path, "recursive": recursive, "max_depth": max_depth}
        return _run_audited(
            "list_dir",
            args,
            path,
            lambda: workspace.list_dir(path, recursive=recursive, max_depth=max_depth),
        )

    @tool
    def read_file(path: str, offset: int = 1, limit: int = DEFAULT_LINE_LIMIT) -> str:
        """按行读取代码仓库中的源文件，返回带行号的文本内容。

        用于核实被调用函数实现、结构体定义、宏定义或调用方边界检查逻辑。

        Args:
            path: 相对工作区根目录的文件路径。
            offset: 1-based 起始行号。
            limit: 最多返回行数（建议 80-200 行，避免浪费上下文预算）。
        """
        clamped = min(limit, MAX_LINE_LIMIT)
        args = {"path": path, "offset": offset, "limit": clamped}
        return _run_audited(
            "read_file",
            args,
            path,
            lambda: workspace.read_file(path, offset=offset, limit=clamped),
        )

    @tool
    def find_files(pattern: str, max_results: int = 100) -> str:
        """按文件名或相对路径通配查找仓库中的源码/头文件。

        Args:
            pattern: 通配模式，如 `*.c`、`*exif*`、`src/*.h`。
            max_results: 最多返回多少条路径。
        """
        args = {"pattern": pattern, "max_results": max_results}
        return _run_audited(
            "find_files",
            args,
            None,
            lambda: workspace.find_files(pattern, max_results=max_results),
        )

    @tool
    def stat(path: str) -> str:
        """查看仓库中单个路径是否存在及其类型与大小。"""
        args = {"path": path}
        return _run_audited("stat", args, path, lambda: workspace.stat(path))

    @tool
    def grep(
        pattern: str,
        path: str = ".",
        is_regex: bool = False,
        case_sensitive: bool = True,
        file_glob: str | None = None,
        context_lines: int = 2,
        max_results: int = DEFAULT_GREP_MAX_RESULTS,
    ) -> str:
        """在代码仓库中按关键字或正则搜索内容，返回命中文件路径、行号与上下文。

        适合查找全局变量用法、宏定义、错误检查语句或特定函数调用点。

        Args:
            pattern: 搜索文本或正则表达式。
            path: 搜索起始目录或文件路径（默认 "."）。
            is_regex: 是否按正则表达式解析 pattern。
            case_sensitive: 是否区分大小写。
            file_glob: 可选文件过滤通配，如 `*.c`、`*.h`。
            context_lines: 命中行前后附带的上下文行数。
            max_results: 最多返回条数。
        """
        args = {
            "pattern": pattern,
            "path": path,
            "is_regex": is_regex,
            "case_sensitive": case_sensitive,
            "file_glob": file_glob,
            "context_lines": context_lines,
            "max_results": max_results,
        }
        return _run_audited(
            "grep",
            args,
            path,
            lambda: workspace.grep(
                pattern,
                path,
                is_regex=is_regex,
                case_sensitive=case_sensitive,
                file_glob=file_glob,
                context_lines=context_lines,
                max_results=max_results,
            ),
        )

    @tool
    def find_symbol(
        symbol_name: str,
        path: str = ".",
        max_results: int = DEFAULT_SYMBOL_MAX_RESULTS,
    ) -> str:
        """在 C/C++ 代码仓库中定位函数、结构体、类型或宏的定义与调用位置（定义优先）。

        当待测函数调用了外部辅助函数、宏或结构体成员，且需要确认其是否包含边界/空指针检查时优先调用本工具。

        Args:
            symbol_name: C/C++ 标识符，如 `ReadBlobByte`、`OP_REQUIRES`、`TensorShape`。
            path: 限定搜索目录或文件（默认 "."）。
            max_results: 最多返回结果数。
        """
        args = {"symbol_name": symbol_name, "path": path, "max_results": max_results}
        return _run_audited(
            "find_symbol",
            args,
            path,
            lambda: workspace.find_symbol(symbol_name, path=path, max_results=max_results),
        )

    @tool
    def inspect_call_chain(
        symbol_name: str,
        direction: str = "both",
        depth: int = 1,
        path: str = ".",
        max_nodes: int = 15,
    ) -> str:
        """使用 Tree-sitter AST 确定性解析指定函数的调用链地图（调用方与被调用方，默认 1 跳）。

        推荐采用渐进式展开策略：先用 depth=1 查看 1 跳直接调用方/被调用方，若不足以判定再针对
        返回结果中的 frontier_callees / frontier_callers 展开下一跳（最多 3 跳）。

        Args:
            symbol_name: C/C++ 函数名（支持 Class::Method）。
            direction: 展开方向："both"（双向）、"callees"（向下查被调用方）、"callers"（向上查调用方）。
            depth: 展开跳数（1~3，建议先用 1）。
            path: 限定搜索目录（默认 "."）。
            max_nodes: 最多返回边数（默认 15）。
        """
        norm_dir = direction if direction in {"both", "callees", "callers"} else "both"
        clamped_depth = max(1, min(int(depth), MAX_CALLCHAIN_DEPTH))
        args = {
            "symbol_name": symbol_name,
            "direction": norm_dir,
            "depth": clamped_depth,
            "path": path,
            "max_nodes": max_nodes,
        }
        return _run_audited(
            "inspect_call_chain",
            args,
            path,
            lambda: callgraph.inspect_call_chain(
                symbol_name,
                path=path,
                direction=norm_dir,  # type: ignore[arg-type]
                depth=clamped_depth,
                max_nodes=max_nodes,
            ),
        )

    @tool
    def read_symbol_body(
        symbol_name: str,
        path: str = ".",
        max_lines: int = 120,
    ) -> str:
        """基于 Tree-sitter AST 精确读取函数、结构体、类型别名或宏的完整定义代码（带行号）。

        比手动 grep + read_file 更准确高效，优先用于查看调用链中关键被调用函数或结构体的完整实现。

        Args:
            symbol_name: C/C++ 符号名（函数名、结构体名、宏名）。
            path: 限定搜索目录（默认 "."）。
            max_lines: 最多返回行数（默认 120）。
        """
        args = {"symbol_name": symbol_name, "path": path, "max_lines": max_lines}
        return _run_audited(
            "read_symbol_body",
            args,
            path,
            lambda: callgraph.read_symbol_body(symbol_name, path=path, max_lines=max_lines),
        )

    return [
        list_dir,
        read_file,
        find_files,
        stat,
        grep,
        find_symbol,
        inspect_call_chain,
        read_symbol_body,
    ]


def _summarize_payload(tool_name: str, payload: dict[str, Any]) -> str:
    if tool_name == "inspect_call_chain":
        return (
            f"callchain '{payload.get('symbol')}' (depth={payload.get('depth')}) -> "
            f"{len(payload.get('callers', []))} callers, {len(payload.get('callees', []))} callees"
        )
    if tool_name == "read_symbol_body":
        defs = payload.get("definitions", [])
        if defs:
            d0 = defs[0]
            return f"body '{d0.get('symbol')}' ({d0.get('kind')}) {d0.get('path')}:L{d0.get('start_line')}-{d0.get('end_line')}"
        return f"body '{payload.get('symbol')}' -> not found"
    if tool_name == "read_file":
        return (
            f"read {payload.get('path')} "
            f"L{payload.get('start_line')}-{payload.get('end_line')}"
        )
    if tool_name == "grep":
        return f"grep '{payload.get('pattern')}' -> {payload.get('count', 0)} hits"
    if tool_name == "find_symbol":
        return (
            f"symbol '{payload.get('symbol')}' -> "
            f"{payload.get('definition_count', 0)} defs / {payload.get('count', 0)} total"
        )
    if tool_name == "find_files":
        return f"find '{payload.get('pattern')}' -> {payload.get('count', 0)} files"
    if tool_name == "list_dir":
        return f"list '{payload.get('path')}' -> {payload.get('count', 0)} entries"
    if tool_name == "stat":
        return f"stat '{payload.get('path')}' ({payload.get('type')})"
    return tool_name
