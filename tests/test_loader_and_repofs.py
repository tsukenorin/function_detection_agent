"""SampleLoader 隔离边界与 RepoFS 惰性快照工作区的单元测试。"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from agent.schema import RawSample
from repo import (
    LazyWorkspace,
    RepoFS,
    SampleLoader,
    SampleLoaderError,
    extract_function_name,
    parse_commit_url,
)
from tool.workspace import ReadOnlyViolation


class FakeParentResolver:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.calls: list[tuple[str, str, str]] = []

    def resolve_parent(self, owner: str, repo: str, commit_sha: str) -> str:
        self.calls.append((owner, repo, commit_sha))
        return self.mapping[commit_sha]


def _make_tarball(dest: Path, root_name: str, files: dict[str, bytes]) -> None:
    with tarfile.open(dest, "w:gz") as tf:
        for rel, content in files.items():
            info = tarfile.TarInfo(name=f"{root_name}/{rel}")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


def test_parse_commit_url_github_and_mirror():
    assert parse_commit_url(
        "https://github.com/ImageMagick/ImageMagick6/commit/dc070da861a015d3c97488fdcca6063b44d47a7b"
    ) == (
        "github.com",
        "ImageMagick",
        "ImageMagick6",
        "dc070da861a015d3c97488fdcca6063b44d47a7b",
    )

    assert parse_commit_url(
        "https://git.qemu.org/?p=qemu.git;a=commit;h=e6908bfe8e07f2b452078e677427702d4e898e9d"
    ) == (
        "github.com",
        "qemu",
        "qemu",
        "e6908bfe8e07f2b452078e677427702d4e898e9d",
    )

    with pytest.raises(SampleLoaderError):
        parse_commit_url("https://unknown.example.com/foo/bar")


def test_extract_function_name():
    code1 = (
        "static MagickBooleanType GetEXIFProperty(const Image *image,\n"
        "  const char *property)\n{\n  if (image == NULL) return MagickFalse;\n}\n"
    )
    assert extract_function_name(code1) == "GetEXIFProperty"

    code2 = "void TensorSlice::Compute(OpKernelContext* ctx) {\n  return;\n}\n"
    assert extract_function_name(code2) == "TensorSlice::Compute"


def test_sample_loader_enforces_pair_isolation():
    commit_sha = "dc070da861a015d3c97488fdcca6063b44d47a7b"
    parent_sha = "1111111111111111111111111111111111111111"
    resolver = FakeParentResolver({commit_sha: parent_sha})
    loader = SampleLoader(parent_resolver=resolver)

    raw_vuln = RawSample(
        idx=1,
        commit_url=f"https://github.com/org/proj/commit/{commit_sha}",
        func="int f(int x) { return buf[x]; }",
        target=1,
    )
    raw_fixed = RawSample(
        idx=2,
        commit_url=f"https://github.com/org/proj/commit/{commit_sha}",
        func="int f(int x) { if (x < 0) return -1; return buf[x]; }",
        target=0,
    )

    in_v, ref_v = loader.load(raw_vuln)
    in_f, ref_f = loader.load(raw_fixed)

    # 版本红线：漏洞版 -> parent_sha；修复版 -> commit_sha
    assert ref_v.allowed_sha == parent_sha
    assert ref_v.version_role == "parent"
    assert ref_f.allowed_sha == commit_sha
    assert ref_f.version_role == "commit"
    assert len(resolver.calls) == 1  # 修复版不应查询父提交

    # 裁剪红线：AgentInput 不含 target / version_role
    assert not hasattr(in_v, "target")
    assert not hasattr(in_v, "version_role")
    assert in_v.snapshot_token != in_f.snapshot_token


def test_repofs_lazy_materialization_and_physical_isolation(tmp_path):
    commit_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    parent_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    downloads: list[str] = []

    def fake_download(repo_ref, tar_path: Path):
        downloads.append(repo_ref.allowed_sha)
        if repo_ref.allowed_sha == parent_sha:
            _make_tarball(
                tar_path,
                f"proj-{parent_sha}",
                {
                    "src/main.c": b"int helper(int n) { return arr[n]; }\n",
                    ".git/config": b"[core]\n",
                    "../evil.txt": b"escape",
                },
            )
        else:
            _make_tarball(
                tar_path,
                f"proj-{commit_sha}",
                {
                    "src/main.c": b"int helper(int n) { if (n < 0) return 0; return arr[n]; }\n",
                },
            )

    repofs = RepoFS(cache_dir=tmp_path / "snaps", downloader=fake_download)
    loader = SampleLoader(parent_resolver=FakeParentResolver({commit_sha: parent_sha}))

    _, ref_v = loader.load(
        RawSample(
            idx=1,
            commit_url=f"https://github.com/org/proj/commit/{commit_sha}",
            func="int f() {}",
            target=1,
        )
    )
    _, ref_f = loader.load(
        RawSample(
            idx=2,
            commit_url=f"https://github.com/org/proj/commit/{commit_sha}",
            func="int f() {}",
            target=0,
        )
    )

    ws_v = repofs.get_workspace(ref_v, lazy=True)
    ws_f = repofs.get_workspace(ref_f, lazy=True)
    assert isinstance(ws_v, LazyWorkspace)
    # 惰性保证：构造后未发生下载
    assert downloads == []
    assert ws_v.is_materialized is False

    # 首次 grep 触发物化，且严格隔离两版内容
    g_v = ws_v.grep("if (n < 0)")
    g_f = ws_f.grep("if (n < 0)")
    assert g_v["count"] == 0
    assert g_f["count"] == 1
    assert downloads == [parent_sha, commit_sha]

    # .git 与路径穿越文件被剥离
    assert ws_v.exists(".git") is False
    assert not (tmp_path / "snaps" / "org" / "proj" / "evil.txt").exists()

    # 再次获取同 SHA 命中缓存，不重复下载
    ws_v2 = repofs.get_workspace(ref_v, lazy=False)
    assert ws_v2.read_file("src/main.c")["total_lines"] == 1
    assert downloads == [parent_sha, commit_sha]

    # 只读强制
    with pytest.raises(ReadOnlyViolation):
        ws_v.write_file("src/main.c", "hack")
