"""Workspace 沙箱与文件原语的单元测试。

重点不是"能不能读文件"，而是"能不能越界"：
路径逃逸、软链逃逸、只读写入，这些是后面版本隔离的地基，必须锁死。
"""

from __future__ import annotations

import os

import pytest

from tool.workspace import (
    BinaryFile,
    NotADirectory,
    NotAFile,
    NotFound,
    PathViolation,
    ReadOnlyViolation,
    Workspace,
)

PASSWORD = "".join(f"line{i}\n" for i in range(1, 11))


@pytest.fixture
def root(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_text(PASSWORD, encoding="utf-8")
    (tmp_path / "src" / "b.h").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "src" / "bin.dat").write_bytes(b"\x00\x01\x02hello")
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("secret\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.c").write_text("int deep;\n", encoding="utf-8")
    (tmp_path / "nested" / "sub").mkdir()
    (tmp_path / "nested" / "sub" / "x.c").write_text("int x;\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ws(root):
    return Workspace(root)


@pytest.fixture
def rw_ws(root):
    return Workspace(root, read_only=False)


# ----------------------------------------------------------------------
# 构造
# ----------------------------------------------------------------------
def test_root_must_exist(tmp_path):
    with pytest.raises(NotFound):
        Workspace(tmp_path / "missing")


def test_root_must_be_dir(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectory):
        Workspace(f)


def test_relative_uses_posix_style(ws):
    info = ws.stat("src/a.c")
    assert info["path"] == "src/a.c"
    assert info["type"] == "file"


# ----------------------------------------------------------------------
# 读文件
# ----------------------------------------------------------------------
def test_read_file_has_line_numbers(ws):
    result = ws.read_file("src/a.c")
    assert result["total_lines"] == 10
    assert result["start_line"] == 1
    assert result["end_line"] == 10
    assert result["truncated"] is False
    assert result["content"].splitlines()[0] == "1: line1"
    assert result["content"].splitlines()[-1] == "10: line10"


def test_read_file_offset_and_limit(ws):
    result = ws.read_file("src/a.c", offset=3, limit=2)
    assert result["content"].splitlines() == ["3: line3", "4: line4"]
    assert result["start_line"] == 3
    assert result["end_line"] == 4
    assert result["truncated"] is True


def test_read_file_offset_beyond_eof(ws):
    result = ws.read_file("src/a.c", offset=99)
    assert result["content"] == ""
    assert result["start_line"] == 0
    assert result["truncated"] is False


def test_read_file_empty(tmp_path):
    (tmp_path / "empty.c").write_text("", encoding="utf-8")
    result = Workspace(tmp_path).read_file("empty.c")
    assert result["total_lines"] == 0
    assert result["content"] == ""


def test_read_file_rejects_binary(ws):
    with pytest.raises(BinaryFile):
        ws.read_file("src/bin.dat")


def test_read_file_rejects_dir(ws):
    with pytest.raises(NotAFile):
        ws.read_file("src")


def test_read_file_missing(ws):
    with pytest.raises(NotFound):
        ws.read_file("src/nope.c")


def test_read_file_limit_is_clamped(ws):
    result = ws.read_file("src/a.c", limit=10**9)
    assert result["end_line"] == 10


def test_read_large_file_streams_window(tmp_path):
    big = tmp_path / "big.c"
    big.write_text("".join(f"l{i}\n" for i in range(1, 5001)), encoding="utf-8")
    ws = Workspace(tmp_path, max_read_bytes=100)
    result = ws.read_file("big.c", offset=10, limit=3)
    assert result["total_lines"] is None
    assert result["content"].splitlines() == ["10: l10", "11: l11", "12: l12"]


# ----------------------------------------------------------------------
# 沙箱：这是测试的重点
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["../outside.c", "src/../../outside.c", ".."])
def test_relative_escape_rejected(ws, bad):
    with pytest.raises(PathViolation):
        ws.read_file(bad)


def test_dot_is_valid_root_path(ws):
    assert ws.exists(".") is True


@pytest.mark.parametrize("bad", ["/etc/passwd", "C:/Windows/win.ini", "~/secret"])
def test_absolute_paths_rejected(ws, bad):
    with pytest.raises(PathViolation):
        ws.stat(bad)


def test_empty_path_rejected(ws):
    with pytest.raises(PathViolation):
        ws.stat("   ")


def test_symlink_escape_rejected(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.c").write_text("int secret;\n", encoding="utf-8")
    link = root / "link.c"
    try:
        os.symlink(outside / "secret.c", link)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台无法创建软链接")

    ws = Workspace(root)
    with pytest.raises(PathViolation):
        ws.read_file("link.c")


# ----------------------------------------------------------------------
# 列目录
# ----------------------------------------------------------------------
def test_list_dir_default_is_flat(ws):
    result = ws.list_dir(".")
    paths = {e["path"] for e in result["entries"]}
    assert paths == {"src", "nested", "README.md"}
    assert ".git" not in paths


def test_list_dir_recursive(ws):
    result = ws.list_dir("nested", recursive=True)
    paths = {e["path"] for e in result["entries"]}
    assert paths == {"nested/deep.c", "nested/sub", "nested/sub/x.c"}


def test_list_dir_respects_max_depth(ws):
    result = ws.list_dir("nested", recursive=True, max_depth=1)
    paths = {e["path"] for e in result["entries"]}
    assert "nested/sub" in paths
    assert "nested/sub/x.c" not in paths


def test_list_dir_include_hidden(ws):
    # .git 属于"永远跳过的噪音目录"，即使 include_hidden 也不列出
    result = ws.list_dir(".", include_hidden=True)
    paths = {e["path"] for e in result["entries"]}
    assert ".hidden.txt" in paths
    assert ".git" not in paths


def test_list_dir_on_file(ws):
    with pytest.raises(NotADirectory):
        ws.list_dir("README.md")


# ----------------------------------------------------------------------
# 查找
# ----------------------------------------------------------------------
def test_find_files_by_name(ws):
    result = ws.find_files("*.c")
    assert result["files"] == ["nested/deep.c", "nested/sub/x.c", "src/a.c"]


def test_find_files_by_relative_pattern(ws):
    result = ws.find_files("src/*.c")
    assert result["files"] == ["src/a.c"]


def test_find_files_skips_hidden_and_git(ws):
    result = ws.find_files("*")
    assert all(".git" not in f for f in result["files"])


def test_find_files_truncates(ws):
    result = ws.find_files("*.c", max_results=1)
    assert result["count"] == 1
    assert result["truncated"] is True


# ----------------------------------------------------------------------
# 写
# ----------------------------------------------------------------------
def test_write_rejected_when_read_only(ws):
    with pytest.raises(ReadOnlyViolation):
        ws.write_file("note.txt", "x")


def test_write_then_read(rw_ws):
    info = rw_ws.write_file("out/report.md", "# 报告\n")
    assert info["path"] == "out/report.md"
    assert info["bytes_written"] == len("# 报告\n".encode("utf-8"))
    assert rw_ws.read_file("out/report.md")["content"] == "1: # 报告"


def test_write_no_overwrite(rw_ws):
    rw_ws.write_file("a.txt", "1")
    with pytest.raises(Exception):
        rw_ws.write_file("a.txt", "2", overwrite=False)


def test_write_rejects_escape(rw_ws):
    with pytest.raises(PathViolation):
        rw_ws.write_file("../evil.txt", "x")


def test_write_rejects_nul(rw_ws):
    with pytest.raises(BinaryFile):
        rw_ws.write_file("a.txt", "a\x00b")
