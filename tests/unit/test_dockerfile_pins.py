"""The Dockerfile's ASCENT pin must equal the submodule's.

The image clones the reference itself (the build context is `docker/`, so the
submodule is not available to COPY). That leaves the commit recorded in two
places, and a silent divergence would mean the servers in the image run
different code from the checkout every comparison is scored against.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
SUBMODULE = "relative_work/ascent"


def dockerfile_ref() -> str:
    m = re.search(r"^ARG ASCENT_REF=([0-9a-f]{7,40})\s*$", DOCKERFILE.read_text(), re.M)
    assert m, "docker/Dockerfile no longer declares ARG ASCENT_REF=<sha>"
    return m.group(1)


def submodule_ref() -> str:
    out = subprocess.run(["git", "ls-tree", "HEAD", SUBMODULE],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip(f"{SUBMODULE} is not recorded in this tree")
    mode, kind, rest = out.stdout.split(maxsplit=2)
    assert kind == "commit", f"{SUBMODULE} is not a submodule entry ({kind})"
    return rest.split()[0]


def test_the_image_clones_the_commit_the_repo_pins():
    d, g = dockerfile_ref(), submodule_ref()
    assert g.startswith(d), (
        f"docker/Dockerfile pins ASCENT_REF={d} but the submodule is at {g}. "
        "Bump ARG ASCENT_REF, or the image ships different reference code than "
        "the checkout."
    )


def test_the_clone_url_matches_the_submodule_url():
    url = re.search(r"git clone (\S+) /opt/ascent", DOCKERFILE.read_text())
    assert url, "the Dockerfile no longer clones ASCENT"
    gitmodules = (ROOT / ".gitmodules").read_text()
    assert url.group(1) in gitmodules, (
        f"the Dockerfile clones {url.group(1)}, which is not the URL in .gitmodules"
    )


def test_the_dockerfile_does_not_copy_from_outside_its_build_context():
    """compose sets `context: docker/`. A COPY of anything above it fails the
    build -- which is how the first merged Dockerfile broke."""
    ctx = {p.name for p in (ROOT / "docker").iterdir()}
    # Join line continuations first: `COPY --chown=x \<newline>    src dst` is
    # one instruction, and reading it line-wise captures the backslash.
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    for line in text.splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(?:--chown=\S+\s+)?(\S+)", line)
        if not m:
            continue
        src = m.group(1).lstrip("./")
        assert src.split("/")[0] in ctx, (
            f"COPY {m.group(1)} is outside the docker/ build context "
            f"(contains: {sorted(ctx)})"
        )
