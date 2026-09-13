"""The Dockerfile and its build context have to agree.

Two failures here cost a real build each. The first merged Dockerfile
`COPY`d `relative_work/ascent` while compose set `context: docker/`, so the
path was not in the context at all. The second hazard is subtler and has the
same symptom: a `COPY` of a path that *is* in the tree but is excluded by
`/.dockerignore`. Both fail late, after the expensive conda and torch layers.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
COMPOSE = ROOT / "docker" / "compose.yaml"
DOCKERIGNORE = ROOT / ".dockerignore"

# Everything the image needs at build time. Excluding any of these silently
# breaks the build a long way in.
BUILD_INPUTS = (
    "relative_work/ascent/model_api/dfine_out.py",
    "relative_work/ascent/third_party/GroundingDINO/setup.py",
    "relative_work/ascent/third_party/MobileSAM/setup.py",
    "relative_work/ascent/third_party/habitat-lab/habitat-lab/setup.py",
    "docker/entrypoint.sh",
    "docker/modules/install_claude_code.sh",
    "docker/modules/install_x11_opengl_vulkan.sh",
)
# Directories that must never reach the daemon: together they are ~34 GB.
MUST_EXCLUDE = ("data/x.pth", "outputs/run/episodes.jsonl",
                ".conda-envs/ascent/bin/python",
                "relative_work/ascent/pretrained_weights/mobile_sam.pt",
                "docker/.env")


def _patterns() -> list:
    return [l.strip() for l in DOCKERIGNORE.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def ignored(rel: str) -> bool:
    """Approximate docker's matcher: enough to catch the mistakes above."""
    for pat in _patterns():
        q = pat.rstrip("/")
        if q.startswith("**/"):
            q = q[3:]
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, q):
            return True
        if rel.startswith(q + "/"):
            return True
        if any(fnmatch.fnmatch(seg, q) for seg in rel.split("/")):
            return True
    return False


def copy_sources() -> list:
    # Join line continuations: `COPY --chown=x \<newline>  src dst` is one
    # instruction, and reading it line-wise captures the backslash.
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(?:--chown=\S+\s+)?(\S+)", line)
        if m:
            out.append(m.group(1).lstrip("./"))
    return out


def test_compose_builds_from_the_repo_root():
    """The COPY paths below are written relative to the root; a narrower
    context makes every one of them wrong."""
    text = COMPOSE.read_text()
    assert re.search(r"context:\s*\.\.\s*$", text, re.M), \
        "docker/compose.yaml must set `context: ..` for the COPYs to resolve"
    assert re.search(r"dockerfile:\s*docker/Dockerfile\s*$", text, re.M), \
        "with a root context the dockerfile path must be docker/Dockerfile"


def test_every_copy_source_exists():
    missing = [s for s in copy_sources() if not (ROOT / s).exists()]
    assert not missing, f"COPY sources not in the tree: {missing}"


def test_no_copy_source_is_excluded_by_dockerignore():
    """A path that exists but is ignored fails the build the same way a missing
    one does, and reads as a typo rather than an ignore rule."""
    blocked = [s for s in copy_sources() if ignored(s)]
    assert not blocked, f"COPY sources excluded by /.dockerignore: {blocked}"


def test_the_build_inputs_survive_the_ignore_file():
    blocked = [p for p in BUILD_INPUTS if ignored(p)]
    assert not blocked, f"/.dockerignore excludes build inputs: {blocked}"


def test_the_heavy_directories_are_excluded():
    """Without these the context is 35 GB instead of ~213 MB."""
    leaked = [p for p in MUST_EXCLUDE if not ignored(p)]
    assert not leaked, f"/.dockerignore lets these into the context: {leaked}"


def test_the_submodule_is_a_recorded_pin():
    """The build COPYs a submodule, so the commit it ships is whatever git
    records -- there is no second copy of the pin to drift."""
    gitmodules = (ROOT / ".gitmodules").read_text()
    assert "relative_work/ascent" in gitmodules
    assert "Zeying-Gong/ascent" in gitmodules
