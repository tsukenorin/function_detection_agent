"""样本加载器（SampleLoader）—— 成对隔离边界的执行者。

职责：
1. 校验原始输入（``RawSample``）并解析 ``commit_url``；
2. 消费 ``target`` 标签，决定唯一允许访问的仓库快照版本：
   - ``target == 1``（漏洞版）：仅允许访问修复提交的父提交 ``C^``；
   - ``target == 0``（修复版）或 ``None``（无标签单样本）：仅允许访问 ``C``；
3. 裁剪掉 ``target`` / ``commit_message`` / ``git diff`` 等一切可能泄露答案的字段，
   产出最小化的 ``AgentInput`` 与仅供 RepoFS 使用的 ``RepoRef``。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from agent.interfaces import ParentCommitResolver
from agent.schema import AgentInput, RawSample, RepoRef

__all__ = [
    "SampleLoaderError",
    "parse_commit_url",
    "extract_function_name",
    "GitHubParentResolver",
    "SampleLoader",
]

_GITHUB_COMMIT_RE = re.compile(
    r"^/([^/]+)/([^/]+?)(?:\.git)?/commit/([0-9a-fA-F]{7,40})(?:/.*)?$"
)

#: 非 GitHub 官方仓库 URL 到 GitHub 镜像仓库的受控映射（仅用于读取代码快照）。
_MIRROR_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("git.qemu.org", "qemu.git"): ("qemu", "qemu"),
    ("git.openssl.org", "openssl.git"): ("openssl", "openssl"),
}

_C_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "typeof",
        "alignof",
        "catch",
        "defined",
    }
)


class SampleLoaderError(ValueError):
    """样本解析或版本定位失败。"""


def parse_commit_url(commit_url: str) -> tuple[str, str, str, str]:
    """解析修复提交链接，返回 ``(host, owner, repo, commit_sha)``。"""
    url = (commit_url or "").strip()
    if not url:
        raise SampleLoaderError("commit_url 不能为空")

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise SampleLoaderError(f"无法解析 commit_url 主机名：{commit_url}")

    if host in {"github.com", "www.github.com"}:
        m = _GITHUB_COMMIT_RE.match(parsed.path)
        if not m:
            raise SampleLoaderError(f"不支持的 GitHub commit_url 格式：{commit_url}")
        return "github.com", m.group(1), m.group(2), m.group(3).lower()

    # 处理 gitweb 风格链接，例如 https://git.qemu.org/?p=qemu.git;a=commit;h=<sha>
    query_str = parsed.query.replace(";", "&")
    qs = parse_qs(query_str)
    project_param = (qs.get("p") or [""])[0]
    commit_hash = (qs.get("h") or [""])[0].lower()
    if (host, project_param) in _MIRROR_MAP and re.fullmatch(
        r"[0-9a-f]{7,40}", commit_hash
    ):
        owner, repo = _MIRROR_MAP[(host, project_param)]
        return "github.com", owner, repo, commit_hash

    raise SampleLoaderError(f"暂不支持该代码托管平台的 commit_url：{commit_url}")


def extract_function_name(func_code: str) -> str | None:
    """从 C/C++ 函数源码头部启发式提取函数名（支持 ``Class::Method``）。"""
    if not func_code:
        return None
    # 去掉前导注释与预处理指令行，只扫函数签名前 40 行
    header_lines: list[str] = []
    for line in func_code.splitlines()[:40]:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith("/*"):
            continue
        header_lines.append(line)
        if "{" in line:
            break
    header = " ".join(header_lines)
    if "{" in header:
        header = header.split("{", 1)[0]

    # 匹配最后一个在参数列表 '(' 之前的标识符（支持 C++ 作用域运算符 :: 与析构 ~）
    matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[~A-Za-z_][A-Za-z0-9_]*)*)\s*\(", header)
    for name in reversed(matches):
        base_name = name.split("::")[-1].lstrip("~")
        if base_name not in _C_CONTROL_KEYWORDS:
            return name
    return None


class GitHubParentResolver:
    """通过 GitHub Commits API 解析指定 commit 的父提交 SHA，并做本地磁盘缓存。

    严格遵守红线：
    - 仅提取 JSON 响应中的 ``parents[0].sha``；
    - 绝不读取、缓存或向外暴露 commit message、files 或 patch/diff。
    """

    def __init__(
        self,
        cache_path: str | Path = ".cache/meta/parents.json",
        *,
        token: str | None = None,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        self.timeout = timeout
        self._client = client
        self._cache: dict[str, str] = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def resolve_parent(self, owner: str, repo: str, commit_sha: str) -> str:
        key = f"{owner}/{repo}@{commit_sha.lower()}"
        if key in self._cache:
            return self._cache[key]

        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "function-detection-agent",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        close_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout, follow_redirects=True)
        try:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                raise SampleLoaderError(
                    f"获取父提交失败 ({resp.status_code})：{owner}/{repo}@{commit_sha}"
                )
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise SampleLoaderError(
                f"请求 GitHub API 解析父提交失败：{exc}"
            ) from exc
        finally:
            if close_client:
                client.close()

        parents = payload.get("parents") or []
        if not parents or not parents[0].get("sha"):
            raise SampleLoaderError(
                f"提交 {owner}/{repo}@{commit_sha} 没有父提交记录"
            )
        parent_sha = str(parents[0]["sha"]).lower()
        self._cache[key] = parent_sha
        self._save_cache()
        return parent_sha


class SampleLoader:
    """样本加载器：把原始样本裁剪成严格隔离的 ``(AgentInput, RepoRef)``。"""

    def __init__(self, parent_resolver: ParentCommitResolver | None = None) -> None:
        self.parent_resolver: ParentCommitResolver = (
            parent_resolver or GitHubParentResolver()
        )

    def load(self, raw: RawSample) -> tuple[AgentInput, RepoRef]:
        host, owner, repo, commit_sha = parse_commit_url(raw.commit_url)
        if raw.target == 1:
            allowed_sha = self.parent_resolver.resolve_parent(owner, repo, commit_sha)
            version_role = "parent"
        else:
            allowed_sha = commit_sha
            version_role = "commit"

        repo_ref = RepoRef(
            host=host,
            owner=owner,
            repo=repo,
            commit_sha=commit_sha,
            allowed_sha=allowed_sha,
            version_role=version_role,
            original_url=raw.commit_url,
        )

        # 不透明快照标识：避免向 LLM 泄露 parent/commit 语义或原始 commit_sha 关联
        token = hashlib.sha256(
            f"{owner}/{repo}@{allowed_sha}".encode("utf-8")
        ).hexdigest()[:12]

        agent_input = AgentInput(
            idx=raw.idx,
            func=raw.func,
            repo_name=f"{owner}/{repo}",
            function_name=extract_function_name(raw.func),
            snapshot_token=token,
        )
        return agent_input, repo_ref

    @staticmethod
    def load_raw_jsonl(path: str | Path) -> list[RawSample]:
        p = Path(path)
        samples: list[RawSample] = []
        with p.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                if not line.strip():
                    continue
                samples.append(RawSample.model_validate(json.loads(line)))
        return samples

    @staticmethod
    def load_raw_from_zip(
        zip_path: str | Path,
        member: str = "dataset/test.jsonl",
    ) -> list[RawSample]:
        samples: list[RawSample] = []
        with zipfile.ZipFile(zip_path) as zf:
            text = zf.read(member).decode("utf-8-sig")
            for line in text.splitlines():
                if not line.strip():
                    continue
                samples.append(RawSample.model_validate(json.loads(line)))
        return samples
