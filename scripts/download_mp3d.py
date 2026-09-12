"""MP3D scene + ObjectNav episode download helper (run inside the nav container).

Three pieces, with different licences:

  --episodes-only   ObjectNav MP3D v1 episodes. PUBLIC, 173 MB. Note the URL
                    path is `objectnav/m3d/` -- habitat's own typo, and
                    `objectnav/mp3d/` returns 403.
  --example-scene   habitat's free single MP3D scene, 17DRP5sb8fy. PUBLIC,
                    ~200 MB. It is a TRAIN scene, so it validates the plumbing
                    but scores nothing on val.
  --mp-script       The 11 val scenes (and the rest). LICENSE-GATED: sign the
                    Matterport Terms of Use at
                    https://niessner.github.io/Matterport/ and they email you
                    `download_mp.py`. Point this flag at that file and it runs
                    `--task habitat`, which is the ~15 GB habitat subset rather
                    than the full 1.3 TB raw release.

Layout produced, which `configs/eval/base_mp3d.yaml` expects:

    data/scene_datasets/mp3d/<scene>/<scene>.glb   (+ .navmesh, .house, _semantic.ply)
    data/datasets/objectnav/mp3d/v1/{train,val,val_mini}/...
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path("data")
SCENES = DATA / "scene_datasets/mp3d"
# habitat's DATASETS.md and ASCENT's README both place these under a `v1`
# level (`data/datasets/objectnav/mp3d/v1/val/...`), and this repo's HM3D tree
# follows the same convention, so native ASCENT's `eval_ascent_mp3d.yaml` reads
# the same files. The zip unpacks flat, so the split dirs are moved under it.
EPISODES = DATA / "datasets/objectnav/mp3d/v1"
EPISODES_URL = "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip"
EXAMPLE_URL = "http://dl.fbaipublicfiles.com/habitat/mp3d/mp3d_example_v1.1.zip"
VAL_SCENES = (
    "2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "QUCTc6BB5sX", "TbHJrupSAjP",
    "X7HyMhZNoso", "Z6MFQCViBuw", "oLBMNvg9in8", "pLe4wQe7qrG", "x8F5xyUWy9e",
    "zsNo4HB9uLZ",
)


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have   {dest.name}")
        return dest
    print(f"  fetch  {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def download_episodes() -> None:
    z = _fetch(EPISODES_URL, EPISODES.parent / "objectnav_mp3d_v1.zip")
    with zipfile.ZipFile(z) as f:
        f.extractall(EPISODES)
    splits = sorted(p.name for p in EPISODES.iterdir() if p.is_dir())
    print(f"episodes under {EPISODES} (splits: {splits})")


def download_example_scene() -> None:
    z = _fetch(EXAMPLE_URL, Path("/tmp/mp3d_example_v1.1.zip"))
    tmp = Path("/tmp/mp3d_example_unpack")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    SCENES.mkdir(parents=True, exist_ok=True)
    # The zip unpacks flat: a scene directory plus the dataset config beside it.
    for item in tmp.iterdir():
        target = SCENES / item.name
        if not target.exists():
            shutil.move(str(item), str(target))
    print(f"example scene under {SCENES} ({sorted(p.name for p in SCENES.iterdir())})")


def download_scenes(mp_script: str) -> None:
    script = Path(mp_script)
    if not script.is_file():
        raise SystemExit(
            f"{script} not found. Sign the Matterport Terms of Use at "
            "https://niessner.github.io/Matterport/ -- they email you download_mp.py."
        )
    SCENES.parent.mkdir(parents=True, exist_ok=True)
    # download_mp.py writes <out>/v1/scans/<scene>/... and prompts for consent.
    cmd = [sys.executable, str(script), "--task", "habitat", "-o", str(SCENES.parent / "mp3d_download")]
    print("  running", " ".join(cmd))
    subprocess.check_call(cmd)
    print(
        "\nDownloaded. The habitat task ships one zip per scene; unpack them so that\n"
        f"  {SCENES}/<scene>/<scene>.glb\n"
        "exists for each, then re-run this script with --verify."
    )


def verify() -> int:
    missing_scene = [s for s in VAL_SCENES if not (SCENES / s / f"{s}.glb").is_file()]
    have = sorted(p.name for p in SCENES.iterdir() if p.is_dir()) if SCENES.is_dir() else []
    val = EPISODES / "val/val.json.gz"
    print(f"scenes dir      {SCENES}  ({len(have)} scene dirs: {have})")
    print(f"val episodes    {val}  {'OK' if val.is_file() else 'MISSING'}")
    if missing_scene:
        print(f"val scenes MISSING ({len(missing_scene)}/11): {missing_scene}")
        print("  -> the licensed download is still needed; "
              "`eval=mp3d_example_smoke` runs without it.")
    else:
        print("all 11 val scenes present")
    return 1 if (missing_scene or not val.is_file()) else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes-only", action="store_true", help="public ObjectNav episodes")
    p.add_argument("--example-scene", action="store_true", help="the free single scene")
    p.add_argument("--mp-script", default="", help="path to the licensed download_mp.py")
    p.add_argument("--verify", action="store_true", help="report what is present")
    a = p.parse_args()
    did = False
    if a.episodes_only:
        download_episodes(); did = True
    if a.example_scene:
        download_example_scene(); did = True
    if a.mp_script:
        download_scenes(a.mp_script); did = True
    if a.verify or not did:
        raise SystemExit(verify())


if __name__ == "__main__":
    main()
