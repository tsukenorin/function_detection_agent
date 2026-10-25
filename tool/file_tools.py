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
from typing import Any

from langchain.tools import tool

from .workspace import (
    DEFAULT_LINE_LIMIT,
    MAX_LINE_LIMIT,
    Workspace,
    WorkspaceError,
)

__all__ = ["build_file_tools", "FILE_TOOL_NAMES"]

FILE_TOOL_NAMES = (
    "list_dir",
    "read_file",
    "find_files",
    "stat",
    "write_file",
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
