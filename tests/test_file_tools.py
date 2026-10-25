"""LangChain 工具适配层的测试。

只验证三件事：
1. 工具能被构建出来、名字符合预期；
2. 正常调用拿到 JSON；
3. 出错时返回 ``ERROR: ...`` 而不是抛异常（agent loop 不该被一次越界读打断）。
"""

from __future__ import annotations

import json

import pytest

from tool import Workspace, build_file_tools
from tool.file_tools import FILE_TOOL_NAMES


@pytest.fixture
def tools(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_text(
        "int main(void) {\n    return 0;\n}\n", encoding="utf-8"
    )
    return {t.name: t for t in build_file_tools(Workspace(tmp_path))}


def test_tool_names(tools):
    assert set(tools) == set(FILE_TOOL_NAMES)


def test_read_file_returns_json(tools):
    payload = json.loads(tools["read_file"].invoke({"path": "src/a.c"}))
    assert payload["path"] == "src/a.c"
    assert payload["content"].splitlines()[0] == "1: int main(void) {"


def test_list_dir_returns_json(tools):
    payload = json.loads(tools["list_dir"].invoke({"path": "src"}))
    assert payload["entries"][0]["path"] == "src/a.c"


def test_find_files_returns_json(tools):
    payload = json.loads(tools["find_files"].invoke({"pattern": "*.c"}))
    assert payload["files"] == ["src/a.c"]


def test_escape_returns_error_string(tools):
    out = tools["read_file"].invoke({"path": "../../secret"})
    assert out.startswith("ERROR:")


def test_write_on_read_only_returns_error(tools):
    out = tools["write_file"].invoke({"path": "note.txt", "content": "x"})
    assert out.startswith("ERROR:")


def test_each_workspace_gets_isolated_tools(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "only_a.c").write_text("int a;\n", encoding="utf-8")

    tools_a = {t.name: t for t in build_file_tools(Workspace(a))}
    tools_b = {t.name: t for t in build_file_tools(Workspace(b))}

    assert json.loads(tools_a["find_files"].invoke({"pattern": "*.c"}))["count"] == 1
    assert json.loads(tools_b["find_files"].invoke({"pattern": "*.c"}))["count"] == 0
