"""基于 Tree-sitter 的 C/C++ 确定性 AST 符号与渐进式调用链分析器。

设计目标：
1. **严格沙箱复用**：所有文件扫描与读取均复用 ``Workspace._resolve`` 与
   ``Workspace._iter_text_files``，绝不绕过单版本只读隔离；
2. **渐进式前沿展开（Progressive Frontier Expansion）**：
   - ``inspect_call_chain(symbol_name, direction, depth=1)`` 默认先返回 **1 跳**
     轻量关系地图（调用方 / 被调用方 / 宏与结构体定义位置及单行签名），
     不盲目展开整张大调用图；
   - Agent 判定当前证据不足时，可针对前沿候选符号（``frontier_callees`` /
     ``frontier_callers``）再次发起下一跳展开，或直接以 ``depth=2`` 深挖；
3. **AST 精确函数体读取**：
   - ``read_symbol_body(symbol_name)`` 直接按 Tree-sitter 语法树节点起止行号
     切出完整函数、结构体或宏定义，省去反复猜行号翻文件的轮次。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tree_sitter as ts
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

from .workspace import Workspace, WorkspaceError

__all__ = [
    "CallGraphAnalyzer",
    "parse_function_dependencies",
    "MAX_CALLCHAIN_DEPTH",
]

MAX_CALLCHAIN_DEPTH = 3

_C_EXTENSIONS = frozenset({".c", ".h"})
_CPP_EXTENSIONS = frozenset({".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".inl", ".inc"})
_ALL_CC_EXTENSIONS = _C_EXTENSIONS | _CPP_EXTENSIONS

#: 过滤无需追踪的 C/C++ 标准库与日志噪音调用，保持调用链高信噪比。
_STDLIB_NOISE = frozenset(
    {
        "printf",
        "fprintf",
        "sprintf",
        "snprintf",
        "vsnprintf",
        "puts",
        "putchar",
        "scanf",
        "sscanf",
        "malloc",
        "calloc",
        "realloc",
        "free",
        "memcpy",
        "memmove",
        "memset",
        "memcmp",
        "memchr",
        "strlen",
        "strnlen",
        "strcmp",
        "strncmp",
        "strcasecmp",
        "strncasecmp",
        "strcpy",
        "strncpy",
        "strcat",
        "strncat",
        "strchr",
        "strrchr",
        "strstr",
        "strdup",
        "strndup",
        "atoi",
        "atol",
        "atoll",
        "strtol",
        "strtoul",
        "strtoll",
        "strtoull",
        "strtod",
        "abs",
        "labs",
        "fabs",
        "floor",
        "ceil",
        "round",
        "sqrt",
        "min",
        "max",
        "sizeof",
        "typeof",
        "offsetof",
        "assert",
        "abort",
        "exit",
        "perror",
        "strerror",
        "fopen",
        "fclose",
        "fread",
        "fwrite",
        "fseek",
        "ftell",
        "fflush",
        "open",
        "close",
        "read",
        "write",
        "lseek",
        "stat",
        "fstat",
        "lstat",
        "errno",
        "ntohs",
        "ntohl",
        "htons",
        "htonl",
        "likely",
        "unlikely",
    }
)

_C_LANGUAGE = ts.Language(tsc.language())
_CPP_LANGUAGE = ts.Language(tscpp.language())


def _make_parser(is_cpp: bool) -> ts.Parser:
    return ts.Parser(_CPP_LANGUAGE if is_cpp else _C_LANGUAGE)


def _node_text(source: bytes, node: ts.Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_declarator_name(source: bytes, node: ts.Node | None) -> str | None:
    """从函数/变量 declarator AST 子树中递归提取规范符号名（支持 Class::Method）。"""
    if node is None:
        return None
    ntype = node.type
    if ntype in {"identifier", "field_identifier", "type_identifier", "destructor_name"}:
        return _node_text(source, node).strip()
    if ntype == "qualified_identifier":
        return re.sub(r"\s+", "", _node_text(source, node))
    if ntype == "function_declarator":
        return _extract_declarator_name(source, node.child_by_field_name("declarator"))
    if ntype in {"pointer_declarator", "reference_declarator", "parenthesized_declarator", "array_declarator"}:
        decl = node.child_by_field_name("declarator")
        if decl is not None:
            return _extract_declarator_name(source, decl)
    for child in node.children:
        if child.type in {
            "qualified_identifier",
            "function_declarator",
            "pointer_declarator",
            "reference_declarator",
            "parenthesized_declarator",
            "identifier",
            "field_identifier",
            "destructor_name",
        }:
            found = _extract_declarator_name(source, child)
            if found:
                return found
    return None


def _extract_call_target_name(source: bytes, fn_node: ts.Node | None) -> str | None:
    """从 ``call_expression.function`` AST 节点提取被调函数名。"""
    if fn_node is None:
        return None
    ntype = fn_node.type
    if ntype in {"identifier", "field_identifier"}:
        return _node_text(source, fn_node).strip()
    if ntype == "qualified_identifier":
        return re.sub(r"\s+", "", _node_text(source, fn_node))
    if ntype == "field_expression":
        field = fn_node.child_by_field_name("field")
        if field is not None:
            return _node_text(source, field).strip()
    if ntype == "template_function":
        name_node = fn_node.child_by_field_name("name")
        if name_node is not None:
            return _extract_call_target_name(source, name_node)
    return None


def _walk_nodes(root: ts.Node):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        # 逆序入栈保持源码顺序遍历
        for child in reversed(node.children):
            stack.append(child)


def _signature_line(source: bytes, node: ts.Node) -> str:
    text = _node_text(source, node)
    head = text.split("{", 1)[0].strip()
    head = re.sub(r"\s+", " ", head)
    return head[:160]


def _base_symbol(name: str) -> str:
    return name.split("::")[-1].lstrip("~")


def parse_function_dependencies(
    func_code: str,
    *,
    is_cpp: bool = True,
    include_stdlib: bool = False,
) -> dict[str, list[str]]:
    """使用 Tree-sitter 确定性解析单个 C/C++ 函数源码的直接被调用方、类型与宏依赖。"""
    source = func_code.encode("utf-8", errors="replace")
    tree = _make_parser(is_cpp).parse(source)

    self_name: str | None = None
    callees: list[str] = []
    seen_callees: set[str] = set()
    types: list[str] = []
    seen_types: set[str] = set()
    macros: list[str] = []
    seen_macros: set[str] = set()

    for node in _walk_nodes(tree.root_node):
        if node.type == "function_definition" and self_name is None:
            self_name = _extract_declarator_name(source, node.child_by_field_name("declarator"))
        elif node.type == "call_expression":
            callee = _extract_call_target_name(source, node.child_by_field_name("function"))
            if not callee:
                continue
            base = _base_symbol(callee)
            if self_name and base == _base_symbol(self_name):
                continue
            if not include_stdlib and base in _STDLIB_NOISE:
                continue
            if callee not in seen_callees:
                seen_callees.add(callee)
                callees.append(callee)
        elif node.type == "type_identifier":
            tname = _node_text(source, node).strip()
            if (
                len(tname) >= 3
                and tname not in seen_types
                and tname not in {"size_t", "ssize_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t", "bool", "FILE"}
            ):
                seen_types.add(tname)
                types.append(tname)
        elif node.type == "identifier":
            ident = _node_text(source, node).strip()
            # 全大写宏常量（长度 >= 4，至少含一个下划线或字母）
            if (
                len(ident) >= 4
                and ident.isupper()
                and ident not in {"NULL", "TRUE", "FALSE", "EOF"}
                and ident not in seen_macros
            ):
                seen_macros.add(ident)
                macros.append(ident)

    return {
        "function_name": [self_name] if self_name else [],
        "callees": callees,
        "types": types[:20],
        "macros": macros[:20],
    }


@dataclass(slots=True)
class _ParsedFile:
    path: str
    mtime: float
    source: bytes
    lines: list[str]
    tree: ts.Tree


class CallGraphAnalyzer:
    """绑定单个只读 ``Workspace`` 的 Tree-sitter 确定性符号与调用链分析器。"""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._file_cache: dict[str, _ParsedFile] = {}
        self._c_parser = _make_parser(is_cpp=False)
        self._cpp_parser = _make_parser(is_cpp=True)

    def _parse_file(self, file_path: Path, rel_path: str) -> _ParsedFile | None:
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            return None

        cached = self._file_cache.get(rel_path)
        if cached is not None and cached.mtime == mtime:
            return cached

        try:
            source = file_path.read_bytes()
        except OSError:
            return None

        parser = self._cpp_parser if file_path.suffix.lower() in _CPP_EXTENSIONS else self._c_parser
        tree = parser.parse(source)
        lines = source.decode("utf-8", errors="replace").splitlines()
        parsed = _ParsedFile(
            path=rel_path,
            mtime=mtime,
            source=source,
            lines=lines,
            tree=tree,
        )
        self._file_cache[rel_path] = parsed
        return parsed

    def _candidate_files(self, base: Path, needle: str, *, max_files: int = 80) -> list[tuple[Path, str]]:
        """先用快速字节子串预筛候选 C/C++ 文件，再交由 Tree-sitter 精确解析 AST。"""
        needle_bytes = needle.encode("utf-8", errors="ignore")
        hits: list[tuple[Path, str]] = []
        for child, rel in self.workspace._iter_text_files(base, extensions=_ALL_CC_EXTENSIONS):
            try:
                raw = child.read_bytes()
            except OSError:
                continue
            if needle_bytes in raw:
                hits.append((child, rel))
                if len(hits) >= max_files:
                    break
        return hits

    def _find_definitions_in_file(
        self,
        pf: _ParsedFile,
        target_sym: str,
    ) -> list[dict[str, Any]]:
        """在单个已解析 AST 中精确查找函数、结构体/类/类型别名或宏定义。"""
        target_base = _base_symbol(target_sym)
        defs: list[dict[str, Any]] = []

        for node in _walk_nodes(pf.tree.root_node):
            ntype = node.type
            if ntype == "function_definition":
                name = _extract_declarator_name(pf.source, node.child_by_field_name("declarator"))
                if not name:
                    continue
                if name == target_sym or _base_symbol(name) == target_base:
                    body = node.child_by_field_name("body")
                    callees: list[str] = []
                    seen: set[str] = set()
                    if body is not None:
                        for sub in _walk_nodes(body):
                            if sub.type == "call_expression":
                                cn = _extract_call_target_name(
                                    pf.source, sub.child_by_field_name("function")
                                )
                                if not cn:
                                    continue
                                cb = _base_symbol(cn)
                                if cb == target_base or cb in _STDLIB_NOISE:
                                    continue
                                if cn not in seen:
                                    seen.add(cn)
                                    callees.append(cn)
                    defs.append(
                        {
                            "symbol": name,
                            "kind": "function",
                            "path": pf.path,
                            "start_line": node.start_point.row + 1,
                            "end_line": node.end_point.row + 1,
                            "signature": _signature_line(pf.source, node),
                            "callees": callees,
                        }
                    )
            elif ntype in {"struct_specifier", "class_specifier", "enum_specifier", "union_specifier"}:
                # 仅收录带定义的 body（field_declaration_list / enumerator_list）
                body = node.child_by_field_name("body")
                name_node = node.child_by_field_name("name")
                if body is not None and name_node is not None:
                    tname = _node_text(pf.source, name_node).strip()
                    if tname == target_sym or tname == target_base:
                        defs.append(
                            {
                                "symbol": tname,
                                "kind": ntype.split("_")[0],
                                "path": pf.path,
                                "start_line": node.start_point.row + 1,
                                "end_line": node.end_point.row + 1,
                                "signature": _signature_line(pf.source, node),
                                "callees": [],
                            }
                        )
            elif ntype == "type_definition":
                decl = node.child_by_field_name("declarator")
                tname = _extract_declarator_name(pf.source, decl)
                if tname and (tname == target_sym or tname == target_base):
                    defs.append(
                        {
                            "symbol": tname,
                            "kind": "typedef",
                            "path": pf.path,
                            "start_line": node.start_point.row + 1,
                            "end_line": node.end_point.row + 1,
                            "signature": _signature_line(pf.source, node),
                            "callees": [],
                        }
                    )
            elif ntype in {"preproc_def", "preproc_function_def"}:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    mname = _node_text(pf.source, name_node).strip()
                    if mname == target_sym or mname == target_base:
                        defs.append(
                            {
                                "symbol": mname,
                                "kind": "macro",
                                "path": pf.path,
                                "start_line": node.start_point.row + 1,
                                "end_line": node.end_point.row + 1,
                                "signature": _node_text(pf.source, node).strip().splitlines()[0][:160],
                                "callees": [],
                            }
                        )
        return defs

    def _find_callers_in_file(
        self,
        pf: _ParsedFile,
        target_sym: str,
        *,
        max_per_file: int = 10,
    ) -> list[dict[str, Any]]:
        """利用 AST 向上回溯外层 ``function_definition``，精准找出所有调用方。"""
        target_base = _base_symbol(target_sym)
        callers: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int]] = set()

        for node in _walk_nodes(pf.tree.root_node):
            if node.type != "call_expression":
                continue
            callee = _extract_call_target_name(pf.source, node.child_by_field_name("function"))
            if not callee:
                continue
            if callee != target_sym and _base_symbol(callee) != target_base:
                continue

            # 向上找所属函数定义
            parent = node.parent
            enclosing: ts.Node | None = None
            while parent is not None:
                if parent.type == "function_definition":
                    enclosing = parent
                    break
                parent = parent.parent

            caller_name = (
                _extract_declarator_name(pf.source, enclosing.child_by_field_name("declarator"))
                if enclosing is not None
                else "<file_scope>"
            ) or "<anonymous>"
            if _base_symbol(caller_name) == target_base:
                continue

            call_line = node.start_point.row + 1
            key = (caller_name, call_line)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            snippet = pf.lines[call_line - 1].strip()[:160] if 0 < call_line <= len(pf.lines) else ""
            callers.append(
                {
                    "caller": caller_name,
                    "path": pf.path,
                    "call_line": call_line,
                    "caller_start_line": enclosing.start_point.row + 1 if enclosing else call_line,
                    "caller_end_line": enclosing.end_point.row + 1 if enclosing else call_line,
                    "call_site": snippet,
                }
            )
            if len(callers) >= max_per_file:
                break
        return callers

    def _validate_symbol(self, symbol_name: str) -> str:
        sym = symbol_name.strip()
        if not sym or not re.match(r"^[A-Za-z_][A-Za-z0-9_:~]*$", sym):
            raise WorkspaceError(f"非法的 C/C++ 符号名：{symbol_name!r}")
        return sym

    def find_symbol_definitions(
        self,
        symbol_name: str,
        *,
        path: str = ".",
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        sym = self._validate_symbol(symbol_name)
        base = self.workspace._resolve(path)
        needle = _base_symbol(sym)
        found: list[dict[str, Any]] = []
        for child, rel in self._candidate_files(base, needle):
            pf = self._parse_file(child, rel)
            if pf is None:
                continue
            found.extend(self._find_definitions_in_file(pf, sym))
            if len(found) >= max_results:
                break
        return found[:max_results]

    def read_symbol_body(
        self,
        symbol_name: str,
        *,
        path: str = ".",
        max_lines: int = 120,
    ) -> dict[str, Any]:
        """按 Tree-sitter AST 节点边界精确读取函数、结构体或宏的完整定义（带行号）。"""
        sym = self._validate_symbol(symbol_name)
        defs = self.find_symbol_definitions(sym, path=path, max_results=5)
        if not defs:
            return {
                "symbol": sym,
                "found": False,
                "definitions": [],
                "message": f"未在工作区 C/C++ 源码中找到符号 `{sym}` 的 AST 定义",
            }

        # 优先排 function，再 struct/class/typedef/macro
        priority = {"function": 0, "struct": 1, "class": 1, "union": 2, "enum": 2, "typedef": 3, "macro": 4}
        defs.sort(key=lambda d: priority.get(d["kind"], 9))
        limit = max(10, min(int(max_lines), 300))

        bodies: list[dict[str, Any]] = []
        for d in defs[:2]:
            start = d["start_line"]
            span = d["end_line"] - start + 1
            read_res = self.workspace.read_file(d["path"], offset=start, limit=min(span, limit))
            bodies.append(
                {
                    "symbol": d["symbol"],
                    "kind": d["kind"],
                    "path": d["path"],
                    "start_line": start,
                    "end_line": d["end_line"],
                    "truncated": span > limit,
                    "callees": d["callees"][:15],
                    "content": read_res["content"],
                }
            )
        return {
            "symbol": sym,
            "found": True,
            "definitions": bodies,
            "total_definitions": len(defs),
        }

    def inspect_call_chain(
        self,
        symbol_name: str,
        *,
        path: str = ".",
        direction: Literal["both", "callees", "callers"] = "both",
        depth: int = 1,
        max_nodes: int = 15,
    ) -> dict[str, Any]:
        """渐进式确定性调用链展开（默认 1 跳，支持最高 3 跳，返回下一跳前沿符号）。"""
        sym = self._validate_symbol(symbol_name)
        base = self.workspace._resolve(path)
        clamped_depth = max(1, min(int(depth), MAX_CALLCHAIN_DEPTH))
        node_cap = max(3, min(int(max_nodes), 30))

        root_defs = self.find_symbol_definitions(sym, path=path, max_results=3)
        callee_edges: list[dict[str, Any]] = []
        caller_edges: list[dict[str, Any]] = []
        frontier_callees: list[str] = []
        frontier_callers: list[str] = []

        # 1) 向下展开被调用方（Callees）
        if direction in {"both", "callees"}:
            visited_down: set[str] = {_base_symbol(sym)}
            current_layer: list[tuple[str, list[dict[str, Any]]]] = [(sym, root_defs)]
            for hop in range(1, clamped_depth + 1):
                next_layer: list[tuple[str, list[dict[str, Any]]]] = []
                for src_sym, src_defs in current_layer:
                    raw_callees: list[str] = []
                    for d in src_defs:
                        for c in d.get("callees", []):
                            if c not in raw_callees:
                                raw_callees.append(c)
                    for callee in raw_callees:
                        if len(callee_edges) >= node_cap:
                            break
                        cb = _base_symbol(callee)
                        if cb in visited_down:
                            continue
                        visited_down.add(cb)
                        cdefs = self.find_symbol_definitions(callee, path=path, max_results=2)
                        loc = (
                            f"{cdefs[0]['path']}:{cdefs[0]['start_line']}-{cdefs[0]['end_line']}"
                            if cdefs
                            else "external_or_macro"
                        )
                        sig = cdefs[0]["signature"] if cdefs else ""
                        callee_edges.append(
                            {
                                "hop": hop,
                                "from": src_sym,
                                "to": callee,
                                "kind": cdefs[0]["kind"] if cdefs else "unknown",
                                "location": loc,
                                "signature": sig,
                            }
                        )
                        if cdefs and cdefs[0]["kind"] == "function":
                            next_layer.append((callee, cdefs))
                            if hop == clamped_depth:
                                frontier_callees.append(callee)
                current_layer = next_layer
                if not current_layer or len(callee_edges) >= node_cap:
                    break

        # 2) 向上展开调用方（Callers）
        if direction in {"both", "callers"}:
            visited_up: set[str] = {_base_symbol(sym)}
            up_queue: list[str] = [sym]
            for hop in range(1, clamped_depth + 1):
                next_up: list[str] = []
                for target in up_queue:
                    needle = _base_symbol(target)
                    for child, rel in self._candidate_files(base, needle, max_files=50):
                        if len(caller_edges) >= node_cap:
                            break
                        pf = self._parse_file(child, rel)
                        if pf is None:
                            continue
                        for hit in self._find_callers_in_file(pf, target, max_per_file=5):
                            if len(caller_edges) >= node_cap:
                                break
                            cname = hit["caller"]
                            cb = _base_symbol(cname)
                            if cb in visited_up:
                                continue
                            visited_up.add(cb)
                            caller_edges.append(
                                {
                                    "hop": hop,
                                    "caller": cname,
                                    "callee": target,
                                    "location": f"{hit['path']}:{hit['call_line']}",
                                    "caller_span": f"{hit['path']}:{hit['caller_start_line']}-{hit['caller_end_line']}",
                                    "call_site": hit["call_site"],
                                }
                            )
                            if cname != "<file_scope>":
                                next_up.append(cname)
                                if hop == clamped_depth:
                                    frontier_callers.append(cname)
                up_queue = next_up[:6]
                if not up_queue or len(caller_edges) >= node_cap:
                    break

        return {
            "symbol": sym,
            "direction": direction,
            "depth": clamped_depth,
            "root_definitions": [
                {
                    "symbol": d["symbol"],
                    "kind": d["kind"],
                    "location": f"{d['path']}:{d['start_line']}-{d['end_line']}",
                    "signature": d["signature"],
                }
                for d in root_defs
            ],
            "callees": callee_edges,
            "callers": caller_edges,
            "frontier_callees": frontier_callees[:8],
            "frontier_callers": frontier_callers[:8],
            "expand_hint": (
                "若当前调用链仍不足以确定边界/空指针守卫，可对 frontier_callees / frontier_callers "
                "中的关键符号调用 read_symbol_body 读取实现，或再次调用 inspect_call_chain 向下/向上展开下一跳。"
            ),
        }
