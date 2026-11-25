"""Tree-sitter 确定性符号定义读取与渐进式多跳调用链分析的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool import (
    CallGraphAnalyzer,
    PathViolation,
    ToolBudget,
    ToolTraceRecorder,
    Workspace,
    build_repo_tools,
    parse_function_dependencies,
)


@pytest.fixture
def multi_hop_repo(tmp_path):
    (tmp_path / "include").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "include" / "types.h").write_text(
        "#define MAX_ITEMS 64\n"
        "struct Buffer {\n"
        "    int len;\n"
        "    char *data;\n"
        "};\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "leaf.c").write_text(
        '#include "types.h"\n'
        "int leaf_sink(struct Buffer *b, int idx) {\n"
        "    return b->data[idx];\n"
        "}\n"
        "int inner_wrapper(struct Buffer *b, int idx) {\n"
        "    if (idx < 0 || idx >= MAX_ITEMS) return -1;\n"
        "    return leaf_sink(b, idx);\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "entry.c").write_text(
        '#include "types.h"\n'
        "int inner_wrapper(struct Buffer *b, int idx);\n"
        "int mid_entry(struct Buffer *b, int idx) {\n"
        "    printf(\"debug\");\n"
        "    return inner_wrapper(b, idx);\n"
        "}\n"
        "int top_caller(struct Buffer *b) {\n"
        "    return mid_entry(b, b->len);\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "engine.cpp").write_text(
        "class Parser {\n"
        "public:\n"
        "    int parse(int n);\n"
        "};\n"
        "int helper_check(int n) { return n > 0; }\n"
        "int Parser::parse(int n) {\n"
        "    if (!helper_check(n)) return -1;\n"
        "    return n;\n"
        "}\n",
        encoding="utf-8",
    )
    return tmp_path


def test_parse_function_dependencies_filters_stdlib():
    code = (
        "int process(struct Buffer *b, int n) {\n"
        "    if (n > MAX_ITEMS) return -1;\n"
        "    memcpy(b->data, src, n);\n"
        "    return inner_wrapper(b, n);\n"
        "}\n"
    )
    deps = parse_function_dependencies(code)
    assert deps["function_name"] == ["process"]
    assert "inner_wrapper" in deps["callees"]
    assert "memcpy" not in deps["callees"]
    assert "MAX_ITEMS" in deps["macros"]


def test_progressive_call_chain_hop1_and_hop2(multi_hop_repo):
    ws = Workspace(multi_hop_repo)
    cg = CallGraphAnalyzer(ws)

    # Hop 1：从 mid_entry 出发，只看 1 跳（应看到 top_caller 和 inner_wrapper，不直接展开 leaf_sink）
    hop1 = cg.inspect_call_chain("mid_entry", direction="both", depth=1)
    assert [e["caller"] for e in hop1["callers"]] == ["top_caller"]
    assert [e["to"] for e in hop1["callees"]] == ["inner_wrapper"]
    assert hop1["frontier_callees"] == ["inner_wrapper"]
    assert hop1["frontier_callers"] == ["top_caller"]

    # Hop 2 渐进展开方式 A：对前沿符号 inner_wrapper 再展开 1 跳 -> 到达 leaf_sink
    hop2_frontier = cg.inspect_call_chain("inner_wrapper", direction="callees", depth=1)
    assert [e["to"] for e in hop2_frontier["callees"]] == ["leaf_sink"]

    # Hop 2 渐进展开方式 B：直接 depth=2 -> 一次看到 mid_entry -> inner_wrapper -> leaf_sink
    hop2_direct = cg.inspect_call_chain("mid_entry", direction="callees", depth=2)
    targets = [(e["hop"], e["from"], e["to"]) for e in hop2_direct["callees"]]
    assert (1, "mid_entry", "inner_wrapper") in targets
    assert (2, "inner_wrapper", "leaf_sink") in targets


def test_read_symbol_body_ast_precision_and_cpp(multi_hop_repo):
    ws = Workspace(multi_hop_repo)
    cg = CallGraphAnalyzer(ws)

    fn_body = cg.read_symbol_body("inner_wrapper")
    assert fn_body["found"] is True
    assert fn_body["definitions"][0]["kind"] == "function"
    assert "leaf_sink" in fn_body["definitions"][0]["callees"]
    assert "if (idx < 0 || idx >= MAX_ITEMS)" in fn_body["definitions"][0]["content"]

    st_body = cg.read_symbol_body("Buffer")
    assert st_body["found"] is True
    assert st_body["definitions"][0]["kind"] == "struct"

    macro_body = cg.read_symbol_body("MAX_ITEMS")
    assert macro_body["found"] is True
    assert macro_body["definitions"][0]["kind"] == "macro"

    cpp_chain = cg.inspect_call_chain("Parser::parse", direction="callees", depth=1)
    assert [e["to"] for e in cpp_chain["callees"]] == ["helper_check"]


def test_callgraph_respects_workspace_sandbox_and_repo_tools(multi_hop_repo):
    ws = Workspace(multi_hop_repo)
    cg = CallGraphAnalyzer(ws)
    with pytest.raises(PathViolation):
        cg.inspect_call_chain("mid_entry", path="../../outside")

    recorder = ToolTraceRecorder(sample_idx=7)
    budget = ToolBudget(max_tool_calls=4, max_read_chars=8000)
    tools = {t.name: t for t in build_repo_tools(ws, recorder=recorder, budget=budget)}
    assert "inspect_call_chain" in tools
    assert "read_symbol_body" in tools

    out1 = json.loads(tools["inspect_call_chain"].invoke({"symbol_name": "mid_entry", "depth": 1}))
    assert out1["frontier_callees"] == ["inner_wrapper"]

    out2 = json.loads(tools["read_symbol_body"].invoke({"symbol_name": "inner_wrapper"}))
    assert out2["found"] is True
    assert len(recorder.records) == 2
    assert recorder.records[0].tool_name == "inspect_call_chain"
    assert recorder.records[1].tool_name == "read_symbol_body"
