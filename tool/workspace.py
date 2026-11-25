"""工作区文件操作原语（v1）。

设计约束：

1. **根目录沙箱**：所有路径都被限制在一个根目录内。`..`、绝对路径、
   指向外部的软链接一律拒绝。后续的下载/解压逻辑也应该落到这里，
   Agent 永远只能看到"工作区"这一层抽象。
2. **不提供 shell**：不暴露 `subprocess`，避免绕过路径校验。
3. **返回值是 dict**：可直接 JSON 序列化，方便塞进工具结果、trace 和报告。
4. **读文件带行号**：让 Agent 的引用能落到具体行，报告里才可核对。

这一层不关心 CVE、漏洞或 LLM，只负责"安全地看文件"。
`grep`、调用关系分析等后续工具应当复用这里的 `_resolve`，不要再自己拼路径。
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

__all__ = [
    "Workspace",
    "WorkspaceError",
    "PathViolation",
    "NotFound",
    "NotAFile",
    "NotADirectory",
    "BinaryFile",
    "ReadOnlyViolation",
    "DEFAULT_LINE_LIMIT",
    "MAX_LINE_LIMIT",
]

DEFAULT_LINE_LIMIT = 300
MAX_LINE_LIMIT = 2000
DEFAULT_MAX_READ_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_ENTRIES = 500
DEFAULT_MAX_RESULTS = 200
DEFAULT_GREP_MAX_RESULTS = 50
DEFAULT_SYMBOL_MAX_RESULTS = 30

#: C/C++ 常见源码与头文件后缀，用于 find_symbol 优先检索。
_C_CPP_EXTENSIONS = frozenset(
    {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".inc"}
)

#: 遍历目录时跳过的目录名（快照里不该有 .git，但缓存目录可能有）。
_SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "node_modules"})

#: 判定为二进制文件的字节前缀长度。
_SNIFF_BYTES = 8192


class WorkspaceError(Exception):
    """工作区操作失败的基类。"""


class PathViolation(WorkspaceError):
    """路径越出工作区根目录，或使用了不允许的路径形式。"""


class NotFound(WorkspaceError):
    """路径不存在。"""


class NotAFile(WorkspaceError):
    """期望文件，实际是目录或其他类型。"""


class NotADirectory(WorkspaceError):
    """期望目录，实际是文件。"""


class BinaryFile(WorkspaceError):
    """文件包含 NUL 字节，判定为二进制，拒绝按文本读取。"""


class ReadOnlyViolation(WorkspaceError):
    """工作区以只读模式打开，却尝试写入。"""


class Workspace:
    """一个受限的文件系统视图。

    参数
    ----
    root:
        工作区根目录，必须已存在。
    read_only:
        默认 True。检测阶段的工作区就是某个 commit 的快照，只应该被读；
        写能力留给"报告/笔记"这类显式可写的目录，避免 Agent 篡改证据。
    max_read_bytes:
        单文件按文本读取的上限。超过则退化为流式只取窗口，不返回总行数。
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        read_only: bool = True,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> None:
        candidate = Path(root).expanduser()
        if not candidate.exists():
            raise NotFound(f"工作区根目录不存在：{candidate}")
        if not candidate.is_dir():
            raise NotADirectory(f"工作区根路径不是目录：{candidate}")

        self.root: Path = candidate.resolve()
        self.read_only = read_only
        self.max_read_bytes = max(1, int(max_read_bytes))

    # ------------------------------------------------------------------
    # 惰性物化钩子（供 RepoFS 子类覆写）
    # ------------------------------------------------------------------
    def _ensure_local(self) -> None:
        """确保存储目录已在本地物化。普通本地 Workspace 为空操作。"""

    # ------------------------------------------------------------------
    # 路径处理
    # ------------------------------------------------------------------
    def _resolve(self, path: str) -> Path:
        """把"工作区内的相对路径"解析成绝对路径，越界即抛错。"""
        self._ensure_local()
        raw = str(path).strip().replace("\\", "/")
        if not raw:
            raise PathViolation("路径不能为空；表示根目录请用 '.'")

        # 1) 拒绝绝对路径与盘符，避免绕过根目录
        if raw.startswith("/") or raw.startswith("~"):
            raise PathViolation(f"只接受相对于工作区根的路径，收到绝对路径：{path}")
        if len(raw) > 1 and raw[1] == ":":
            raise PathViolation(f"只接受相对于工作区根的路径，收到盘符路径：{path}")

        # 2) resolve() 会展开 .. 与软链接，任何逃逸都会在这里现形
        target = (self.root / raw).resolve()
        if not self._is_inside(target):
            raise PathViolation(f"路径越出工作区：{path}")
        return target

    def _is_inside(self, target: Path) -> bool:
        if target == self.root:
            return True
        root = os.path.normcase(str(self.root))
        child = os.path.normcase(str(target))
        return child.startswith(root + os.sep)

    def _is_safe(self, target: Path) -> bool:
        """遍历得到的条目在真正读取前再做一次软链接校验。"""
        try:
            return self._is_inside(target.resolve())
        except OSError:
            return False

    def relative(self, target: Path) -> str:
        """转成统一的 POSIX 风格相对路径，便于写进报告与跨平台复现。"""
        rel = os.path.relpath(str(target), str(self.root))
        return "." if rel == "." else rel.replace(os.sep, "/")

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def read_file(
        self,
        path: str,
        *,
        offset: int = 1,
        limit: int = DEFAULT_LINE_LIMIT,
    ) -> dict[str, Any]:
        """按行读取文本文件，返回带行号的内容。

        行号是给 Agent 引用用的（"第 42 行"），不要在学生报告里丢掉。
        `offset` 为 1-based，与常见编辑器一致。
        """
        target = self._resolve(path)
        if not target.exists():
            raise NotFound(f"文件不存在：{self.relative(target)}")
        if target.is_dir():
            raise NotAFile(f"{self.relative(target)} 是目录，请用 list_dir")

        offset = max(1, int(offset))
        limit = max(1, min(int(limit), MAX_LINE_LIMIT))

        with target.open("rb") as fh:
            if b"\x00" in fh.read(_SNIFF_BYTES):
                raise BinaryFile(f"{self.relative(target)} 是二进制文件，无法按文本读取")

        size = target.stat().st_size
        total: int | None

        if size <= self.max_read_bytes:
            text = target.read_text(encoding="utf-8", errors="replace")
            all_lines = text.splitlines()
            total = len(all_lines)
            window = all_lines[offset - 1 : offset - 1 + limit]
            truncated = offset - 1 + limit < total
            end = offset - 1 + len(window)
        else:
            # 超大文件：只流式取窗口，不统计总行数
            total = None
            window = []
            with target.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                for lineno, line in enumerate(fh, 1):
                    if lineno < offset:
                        continue
                    if len(window) >= limit:
                        break
                    window.append(line.rstrip("\r\n"))
            truncated = len(window) >= limit
            end = offset - 1 + len(window)

        content = "\n".join(f"{offset + i}: {line}" for i, line in enumerate(window))
        return {
            "path": self.relative(target),
            "start_line": offset if window else 0,
            "end_line": end,
            "total_lines": total,
            "truncated": truncated,
            "size_bytes": size,
            "content": content,
        }

    def list_dir(
        self,
        path: str = ".",
        *,
        recursive: bool = False,
        max_depth: int = 2,
        include_hidden: bool = False,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> dict[str, Any]:
        """列目录。默认只列一层，跳过隐藏项与 .git 之类的噪音目录。"""
        base = self._resolve(path)
        if not base.exists():
            raise NotFound(f"目录不存在：{self.relative(base)}")
        if not base.is_dir():
            raise NotADirectory(f"{self.relative(base)} 是文件，请用 read_file")

        entries: list[dict[str, Any]] = []

        def walk(current: Path, depth: int) -> None:
            if len(entries) >= max_entries:
                return
            try:
                children = sorted(
                    current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
                )
            except OSError:
                return
            for child in children:
                if len(entries) >= max_entries:
                    return
                if child.name in _SKIP_DIRS:
                    continue
                if not include_hidden and child.name.startswith("."):
                    continue
                if not self._is_safe(child):
                    continue
                is_dir = child.is_dir()
                try:
                    size = None if is_dir else child.stat().st_size
                except OSError:
                    size = None
                entries.append(
                    {
                        "path": self.relative(child),
                        "type": "dir" if is_dir else "file",
                        "size_bytes": size,
                    }
                )
                if recursive and is_dir and depth < max_depth:
                    walk(child, depth + 1)

        walk(base, 1)
        return {
            "path": self.relative(base),
            "entries": entries,
            "count": len(entries),
            "truncated": len(entries) >= max_entries,
        }

    def exists(self, path: str) -> bool:
        """路径是否存在。越界仍然抛错，不要把违规当成 False 悄悄吞掉。"""
        return self._resolve(path).exists()

    def stat(self, path: str) -> dict[str, Any]:
        """返回单个路径的类型与元信息。"""
        target = self._resolve(path)
        if not target.exists():
            raise NotFound(f"路径不存在：{self.relative(target)}")
        info = target.stat()
        return {
            "path": self.relative(target),
            "type": "dir" if target.is_dir() else "file",
            "size_bytes": None if target.is_dir() else info.st_size,
            "mtime": info.st_mtime,
        }

    def find_files(
        self,
        pattern: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> dict[str, Any]:
        """按文件名/相对路径通配查找文件（对应 PPT 里的 `find`）。

        支持两种写法：
        - `*.c`       匹配任意目录下的文件名
        - `src/*.c`   匹配相对路径
        """
        pat = (pattern or "*").strip().replace("\\", "/") or "*"
        self._ensure_local()
        match_rel = "/" in pat
        found: list[str] = []

        for child in self.root.rglob("*"):
            if len(found) >= max_results:
                break
            if any(part in _SKIP_DIRS for part in child.parts):
                continue
            if child.name.startswith("."):
                continue
            if not self._is_safe(child) or not child.is_file():
                continue
            rel = self.relative(child)
            if fnmatch.fnmatch(rel if match_rel else child.name, pat):
                found.append(rel)

        return {
            "pattern": pattern,
            "files": sorted(found),
            "count": len(found),
            "truncated": len(found) >= max_results,
        }

    def _iter_text_files(
        self,
        base: Path,
        *,
        file_glob: str | None = None,
        extensions: frozenset[str] | None = None,
    ):
        """安全迭代工作区内的文本文件，跳过隐藏项、噪音目录与二进制文件。"""
        if base.is_file():
            candidates = [base]
        else:
            candidates = sorted(base.rglob("*"))

        norm_glob = file_glob.strip().replace("\\", "/") if file_glob else None
        match_rel = bool(norm_glob and "/" in norm_glob)

        for child in candidates:
            if child.is_dir():
                continue
            rel = self.relative(child)
            rel_parts = rel.split("/")
            if any(part in _SKIP_DIRS or part.startswith(".") for part in rel_parts):
                continue
            if not self._is_safe(child) or not child.is_file():
                continue
            if extensions is not None and child.suffix.lower() not in extensions:
                continue
            if norm_glob is not None:
                target_name = rel if match_rel else child.name
                if not fnmatch.fnmatch(target_name, norm_glob):
                    continue
            try:
                size = child.stat().st_size
                if size > self.max_read_bytes:
                    continue
                with child.open("rb") as fh:
                    if b"\x00" in fh.read(_SNIFF_BYTES):
                        continue
            except OSError:
                continue
            yield child, rel

    def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        is_regex: bool = False,
        case_sensitive: bool = True,
        file_glob: str | None = None,
        context_lines: int = 2,
        max_results: int = DEFAULT_GREP_MAX_RESULTS,
    ) -> dict[str, Any]:
        """在工作区内按内容搜索文本或正则（复用 ``_resolve`` 沙箱校验）。"""
        if not pattern:
            raise WorkspaceError("grep 的 pattern 不能为空")

        base = self._resolve(path)
        if not base.exists():
            raise NotFound(f"路径不存在：{self.relative(base)}")

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern if is_regex else re.escape(pattern), flags)
        except re.error as exc:
            raise WorkspaceError(f"无效的正则表达式：{exc}") from exc

        ctx = max(0, min(int(context_lines), 10))
        limit = max(1, min(int(max_results), 200))
        matches: list[dict[str, Any]] = []
        truncated = False

        for child, rel in self._iter_text_files(base, file_glob=file_glob):
            try:
                lines = child.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for idx, line in enumerate(lines):
                if not compiled.search(line):
                    continue
                if len(matches) >= limit:
                    truncated = True
                    break
                start = max(0, idx - ctx)
                end = min(len(lines), idx + ctx + 1)
                context = [f"{i + 1}: {lines[i]}" for i in range(start, end)]
                matches.append(
                    {
                        "path": rel,
                        "line": idx + 1,
                        "text": line,
                        "context": "\n".join(context),
                    }
                )
            if truncated:
                break

        return {
            "pattern": pattern,
            "path": self.relative(base),
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
        }

    def find_symbol(
        self,
        symbol_name: str,
        *,
        path: str = ".",
        max_results: int = DEFAULT_SYMBOL_MAX_RESULTS,
    ) -> dict[str, Any]:
        """在 C/C++ 代码中定位符号的定义、声明与调用点（优先返回定义）。

        用于按需挖掘被调用方实现、调用方、结构体/类型定义与宏定义。
        """
        sym = symbol_name.strip()
        if not sym or not re.match(r"^[A-Za-z_][A-Za-z0-9_:~]*$", sym):
            raise WorkspaceError(f"非法的 C/C++ 符号名：{symbol_name!r}")

        base = self._resolve(path)
        if not base.exists():
            raise NotFound(f"路径不存在：{self.relative(base)}")

        limit = max(1, min(int(max_results), 100))
        esc = re.escape(sym)
        word_re = re.compile(rf"\b{esc}\b")
        macro_re = re.compile(rf"^\s*#\s*define\s+{esc}\b")
        type_re = re.compile(
            rf"^\s*(?:typedef\s+)?(?:struct|union|enum|class)\s+{esc}\b|^\s*typedef\b.*\b{esc}\s*;"
        )
        func_sig_re = re.compile(rf"\b{esc}\s*\(")

        definitions: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []

        # 优先在 C/C++ 源/头文件中查找；若目录下没有此类文件则回退到所有文本文件
        files = list(self._iter_text_files(base, extensions=_C_CPP_EXTENSIONS))
        if not files:
            files = list(self._iter_text_files(base))

        for child, rel in files:
            try:
                lines = child.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for idx, line in enumerate(lines):
                if not word_re.search(line):
                    continue
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                    continue

                kind = "reference"
                if macro_re.search(line):
                    kind = "macro_def"
                elif type_re.search(line):
                    kind = "type_def"
                elif func_sig_re.search(line):
                    # 判断函数定义（当前行或后续 3 行内出现 '{' 且当前行不是 if/while/for/switch 调用）
                    prefix = line[: line.find(sym)].strip()
                    if (
                        prefix
                        and not prefix.endswith((".", "->", "=", "(", ",", "return"))
                        and not re.match(r"^(?:if|while|for|switch|return|else)\b", stripped)
                    ):
                        lookahead = " ".join(lines[idx : min(len(lines), idx + 4)])
                        if "{" in lookahead and not stripped.endswith(";"):
                            kind = "func_def"
                        elif stripped.endswith(";"):
                            kind = "declaration"

                start = max(0, idx - 1)
                end = min(len(lines), idx + (6 if kind in {"func_def", "type_def", "macro_def"} else 2))
                entry = {
                    "path": rel,
                    "line": idx + 1,
                    "kind": kind,
                    "text": stripped,
                    "snippet": "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end)),
                }
                if kind in {"func_def", "macro_def", "type_def", "declaration"}:
                    definitions.append(entry)
                elif len(references) < limit:
                    references.append(entry)

        combined = definitions + references
        truncated = len(combined) > limit
        results = combined[:limit]
        return {
            "symbol": sym,
            "path": self.relative(base),
            "results": results,
            "definition_count": len(definitions),
            "count": len(results),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def write_file(
        self,
        path: str,
        content: str,
        *,
        create_parents: bool = True,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """写入文本文件。只读工作区下直接拒绝。"""
        if self.read_only:
            raise ReadOnlyViolation(
                f"工作区是只读的，拒绝写入：{path}（检测阶段不应修改代码快照）"
            )
        if "\x00" in content:
            raise BinaryFile("拒绝写入包含 NUL 字节的内容")

        target = self._resolve(path)
        if target.is_dir():
            raise NotAFile(f"{self.relative(target)} 是目录，无法写入")
        if target.exists() and not overwrite:
            raise WorkspaceError(f"文件已存在且 overwrite=False：{self.relative(target)}")
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        elif not target.parent.exists():
            raise NotFound(f"父目录不存在：{self.relative(target.parent)}")

        target.write_text(content, encoding="utf-8", newline="\n")
        return {
            "path": self.relative(target),
            "bytes_written": len(content.encode("utf-8")),
        }
