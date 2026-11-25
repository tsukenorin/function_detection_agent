"""grep、find_symbol、预算控制与工具审计日志的单元测试。"""

from __future__ import annotations

import json

import pytest

from tool import (
    PathViolation,
    ToolBudget,
    ToolTraceRecorder,
    Workspace,
    WorkspaceError,
    build_repo_tools,
)
from tool.file_tools import REPO_TOOL_NAMES


@pytest.fixture
def c_repo(tmp_path):
    (tmp_path / "include").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: safe_copy\n", encoding="utf-8")
    (tmp_path / "include" / "safe.h").write_text(
        "#define MAX_BUF 256\n"
        "struct Packet {\n"
        "    int len;\n"
        "    char *data;\n"
        "};\n"
        "int safe_copy(char *dst, const char *src, int n);\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "safe.c").write_text(
        "#include \"safe.h\"\n"
        "int safe_copy(char *dst, const char *src, int n)\n"
        "{\n"
        "    if (n < 0 || n >= MAX_BUF) return -1;\n"
        "    return n;\n"
        "}\n"
        "int caller(struct Packet *p) {\n"
        "    return safe_copy(p->data, \"hi\", p->len);\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "blob.bin").write_bytes(b"\x00safe_copy\x00")
    return tmp_path


def test_grep_literal_and_context(c_repo):
    ws = Workspace(c_repo)
    res = ws.grep("MAX_BUF", context_lines=1)
    assert res["count"] == 2
    paths = {m["path"] for m in res["matches"]}
    assert paths == {"include/safe.h", "src/safe.c"}
    assert any("MAX_BUF" in m["context"] for m in res["matches"])


def test_grep_regex_and_file_glob(c_repo):
    ws = Workspace(c_repo)
    res = ws.grep(r"safe_copy\s*\(", is_regex=True, file_glob="*.c")
    assert res["count"] == 2
    assert all(m["path"].endswith(".c") for m in res["matches"])


def test_grep_skips_git_and_binary(c_repo):
    ws = Workspace(c_repo)
    res = ws.grep("safe_copy")
    paths = {m["path"] for m in res["matches"]}
    assert ".git/HEAD" not in paths
    assert "src/blob.bin" not in paths


def test_grep_rejects_path_escape_and_invalid_regex(c_repo):
    ws = Workspace(c_repo)
    with pytest.raises(PathViolation):
        ws.grep("foo", path="../outside")
    with pytest.raises(WorkspaceError):
        ws.grep("[unclosed", is_regex=True)


def test_find_symbol_prioritizes_definitions(c_repo):
    ws = Workspace(c_repo)
    res = ws.find_symbol("safe_copy")
    assert res["definition_count"] >= 2  # declaration + func_def
    kinds = [r["kind"] for r in res["results"]]
    assert "func_def" in kinds
    assert "reference" in kinds
    # 定义/声明必须排在普通引用前面
    assert kinds.index("func_def") < kinds.index("reference")

    macro_res = ws.find_symbol("MAX_BUF")
    assert macro_res["results"][0]["kind"] == "macro_def"

    struct_res = ws.find_symbol("Packet")
    assert struct_res["results"][0]["kind"] == "type_def"


def test_find_symbol_rejects_escape_and_bad_identifier(c_repo):
    ws = Workspace(c_repo)
    with pytest.raises(PathViolation):
        ws.find_symbol("safe_copy", path="../../etc")
    with pytest.raises(WorkspaceError):
        ws.find_symbol("bad name; rm -rf")


def test_build_repo_tools_is_readonly_audited_and_budgeted(c_repo):
    events = []
    recorder = ToolTraceRecorder(sample_idx=42, on_event=events.append, current_step=1)
    budget = ToolBudget(max_tool_calls=2, max_read_chars=500)
    tools = {
        t.name: t
        for t in build_repo_tools(Workspace(c_repo), recorder=recorder, budget=budget)
    }

    assert set(tools) == set(REPO_TOOL_NAMES)
    assert "write_file" not in tools

    # 第 1 次调用：find_symbol
    out1 = tools["find_symbol"].invoke({"symbol_name": "safe_copy"})
    assert not out1.startswith("ERROR:")
    assert budget.calls_used == 1
    assert len(recorder.records) == 1
    assert recorder.records[0].tool_name == "find_symbol"
    assert recorder.records[0].ok is True

    # 第 2 次调用：grep（可能触发字符预算截断）
    out2 = tools["grep"].invoke({"pattern": "safe_copy"})
    assert budget.calls_used == 2
    assert len(recorder.records) == 2

    # 第 3 次调用：超出调用次数预算，返回 ERROR 而不是崩溃
    out3 = tools["list_dir"].invoke({"path": "."})
    assert out3.startswith("ERROR:")
    assert "上限" in out3
    assert recorder.records[-1].ok is False
    assert any(e.event_type == "tool_call" for e in events)
    assert any(e.event_type == "tool_result" for e in events)
