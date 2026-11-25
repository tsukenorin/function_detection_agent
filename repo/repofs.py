"""RepoFS 仓库快照层：按 SHA 物理隔离与惰性物化。

设计要点：
1. **按 SHA 物理分片**：缓存路径为 ``{cache_dir}/snapshots/{owner}/{repo}/{allowed_sha}``，
   漏洞版（父提交 ``C^``）与修复版（提交后 ``C``）落在完全不同的目录中，绝不共用工作区；
2. **对 Agent 透明的惰性物化**：``LazyWorkspace`` 继承自 ``tool.workspace.Workspace``，
   仅在 Agent 首次发起仓库工具调用时触发 ``ensure_snapshot`` 下载与解压；
   若 Agent 认为无需读仓库，则零网络下载；
3. **安全解压与只读强制**：解压时剥离 ``.git``、拒绝路径穿越与越界软链，
   并以 ``read_only=True`` 暴露给工具层。
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx

from agent.schema import RepoRef
from tool.workspace import DEFAULT_MAX_READ_BYTES, Workspace, WorkspaceError

__all__ = ["RepoFSError", "LazyWorkspace", "RepoFS"]

_READY_MARKER = ".snapshot_ready"
DEFAULT_MAX_TARBALL_BYTES = 250 * 1024 * 1024  # 250 MB


class RepoFSError(WorkspaceError):
    """仓库快照下载或解压失败。"""


class LazyWorkspace(Workspace):
    """惰性物化的只读工作区。

    构造时快照目录尚可不存在；首次执行 ``read_file`` / ``grep`` / ``find_symbol`` 等
    操作时，通过 ``_ensure_local()`` 透明拉取并物化到 ``self.root``。
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        materializer: Callable[[], Path],
        *,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> None:
        self.root: Path = Path(root).expanduser().resolve()
        self.read_only: bool = True
        self.max_read_bytes: int = max(1, int(max_read_bytes))
        self._materializer = materializer
        self._materialized = (self.root / _READY_MARKER).exists()

    @property
    def is_materialized(self) -> bool:
        return self._materialized

    def _ensure_local(self) -> None:
        if self._materialized and self.root.exists():
            return
        actual_root = self._materializer().resolve()
        self.root = actual_root
        self._materialized = True


def _safe_extract_tarball(tar_path: Path, dest_dir: Path) -> None:
    """安全解压 GitHub codeload tarball：剥离顶层目录与 .git，拦截路径穿越与越界软链。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    dest_prefix = os.path.normcase(str(dest_resolved)) + os.sep

    with tarfile.open(tar_path, mode="r:*") as tf:
        for member in tf.getmembers():
            raw_name = member.name.replace("\\", "/").lstrip("/")
            parts = [p for p in raw_name.split("/") if p and p != "."]
            # GitHub codeload tarball 顶层通常有一层 `{repo}-{sha}/`
            if len(parts) <= 1:
                continue
            rel_parts = parts[1:]
            if any(p == ".." or p == ".git" for p in rel_parts):
                continue

            target = (dest_resolved / Path(*rel_parts)).resolve()
            if os.path.normcase(str(target)) != os.path.normcase(str(dest_resolved)) and not os.path.normcase(
                str(target)
            ).startswith(dest_prefix):
                continue

            # 跳过软链与硬链，避免跨目录逃逸或 Windows 权限失败
            if member.issym() or member.islnk():
                continue

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                src_stream = tf.extractfile(member)
                if src_stream is None:
                    continue
                with src_stream, target.open("wb") as out_fh:
                    shutil.copyfileobj(src_stream, out_fh)


class RepoFS:
    """基于 codeload tarball 与本地分片缓存的仓库快照管理器。"""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/snapshots",
        *,
        max_tarball_bytes: int = DEFAULT_MAX_TARBALL_BYTES,
        timeout: float = 120.0,
        downloader: Callable[[RepoRef, Path], None] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_tarball_bytes = max_tarball_bytes
        self.timeout = timeout
        self._downloader = downloader

    def snapshot_dir(self, repo_ref: RepoRef) -> Path:
        safe_owner = repo_ref.owner.replace("/", "_")
        safe_repo = repo_ref.repo.replace("/", "_")
        safe_sha = repo_ref.allowed_sha.lower()
        return self.cache_dir / safe_owner / safe_repo / safe_sha

    def _download_tarball(self, repo_ref: RepoRef, tar_path: Path) -> None:
        if self._downloader is not None:
            self._downloader(repo_ref, tar_path)
            return

        # 严格只按 allowed_sha 拉取该单一版本快照，绝不请求 diff/compare
        url = (
            f"https://codeload.github.com/{repo_ref.owner}/{repo_ref.repo}"
            f"/tar.gz/{repo_ref.allowed_sha}"
        )
        headers = {"User-Agent": "function-detection-agent"}
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        total = 0
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code != 200:
                        raise RepoFSError(
                            f"下载仓库快照失败 ({resp.status_code})："
                            f"{repo_ref.repo_full_name}@{repo_ref.allowed_sha[:10]}"
                        )
                    content_length = resp.headers.get("content-length")
                    if content_length and int(content_length) > self.max_tarball_bytes:
                        raise RepoFSError(
                            f"仓库快照体积 ({int(content_length)}B) 超过上限 "
                            f"({self.max_tarball_bytes}B)，跳过整库下载"
                        )
                    with tar_path.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            total += len(chunk)
                            if total > self.max_tarball_bytes:
                                raise RepoFSError(
                                    f"仓库快照流式体积超过上限 ({self.max_tarball_bytes}B)"
                                )
                            fh.write(chunk)
        except httpx.HTTPError as exc:
            raise RepoFSError(f"下载仓库快照网络异常：{exc}") from exc

    def ensure_snapshot(self, repo_ref: RepoRef) -> Path:
        """确保 ``repo_ref.allowed_sha`` 对应的本地只读快照目录已就绪。"""
        target_dir = self.snapshot_dir(repo_ref)
        marker = target_dir / _READY_MARKER
        if marker.exists():
            return target_dir

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = Path(
            tempfile.mkdtemp(prefix=f"stage_{repo_ref.allowed_sha[:8]}_", dir=target_dir.parent)
        )
        tar_path = staging_parent / "snapshot.tar.gz"
        extract_dir = staging_parent / "tree"
        try:
            self._download_tarball(repo_ref, tar_path)
            _safe_extract_tarball(tar_path, extract_dir)
            (extract_dir / _READY_MARKER).write_text(
                repo_ref.allowed_sha + "\n", encoding="utf-8"
            )
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            extract_dir.rename(target_dir)
            return target_dir
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

    def get_workspace(self, repo_ref: RepoRef, *, lazy: bool = True) -> Workspace:
        """返回严格绑定到 ``repo_ref.allowed_sha`` 的只读 ``Workspace``。"""
        snap_dir = self.snapshot_dir(repo_ref)
        if lazy:
            return LazyWorkspace(snap_dir, lambda: self.ensure_snapshot(repo_ref))
        ready_dir = self.ensure_snapshot(repo_ref)
        return Workspace(ready_dir, read_only=True)
