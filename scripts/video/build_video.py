#!/usr/bin/env python3
"""Assemble the ICRA video from the rendered segments, under the attachment limits.

    python scripts/video/build_video.py                         # -> outputs/video/icra2027_rough.mp4
    python scripts/video/build_video.py --narration voice.wav   # mux a recorded voiceover
    python scripts/video/build_video.py --print-storyboard      # the table for docs/VIDEO.md

Robot footage: a shot of kind `robot` with slot R2_5 is replaced by
`outputs/video/robot/R2_5.mp4` when that file exists (scaled and padded to 1280x720,
trimmed or frozen to the slot's duration); otherwise the placeholder slate rendered by
`make_assets.py` stands in. Nothing else about the cut changes, so the timing the narration
was written against holds.

Size: ICRA's limit is 20 MB for at most 3 minutes. The video bitrate is derived from the
total duration so a two-pass H.264 encode lands at ~18.5 MB with the audio, and the result
is checked against both limits before the script reports success.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import FPS, H, OUT_ROOT, ROBOT_DIR, SEGMENTS, W, load_shotlist, probe  # noqa: E402

TARGET_BYTES = 18_500_000     # headroom under the 20 MB limit for container overhead
AUDIO_KBPS = 64


def conform_robot(src: pathlib.Path, dur: float, out: pathlib.Path) -> pathlib.Path:
    """Scale/pad a robot clip to the frame, cut or freeze it to `dur` seconds, 24 fps."""
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x181a1e,fps={FPS},"
          f"tpad=stop_mode=clone:stop_duration={dur},trim=duration={dur},setpts=PTS-STARTPTS,"
          f"fade=t=in:st=0:d=0.3,fade=t=out:st={max(0.0, dur - 0.3)}:d=0.3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", vf,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    return out


def storyboard(data: dict) -> str:
    rows = ["| t | shot | beat | on screen | narration |", "|---|---|---|---|---|"]
    for s in data["shots"]:
        t0 = s["t_start"]
        rows.append(f"| {int(t0 // 60)}:{int(t0 % 60):02d} (+{s['dur']} s) | `{s['id']}` | "
                    f"{s['beat']} | {s.get('on_screen', '')} | {s.get('narration', '')} |")
    rows.append(f"\nTotal: {data['total']:.0f} s of {data['meta']['limits']['max_seconds']} s.")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--narration", type=pathlib.Path, help="voiceover audio file to mux (wav/m4a/mp3)")
    ap.add_argument("--out", type=pathlib.Path, default=OUT_ROOT / "icra2027_rough.mp4")
    ap.add_argument("--print-storyboard", action="store_true")
    args = ap.parse_args()

    data = load_shotlist()
    if args.print_storyboard:
        print(storyboard(data))
        return

    limits = data["meta"]["limits"]
    total = data["total"]
    if total > limits["max_seconds"]:
        sys.exit(f"shot list runs {total:.1f} s, over the {limits['max_seconds']} s limit")

    parts, substituted = [], []
    for s in data["shots"]:
        seg = SEGMENTS / f"{s['id']}.mp4"
        if s["kind"] == "robot":
            robot = ROBOT_DIR / f"{s['slot']}.mp4"
            if robot.exists():
                seg = conform_robot(robot, float(s["dur"]), SEGMENTS / f"{s['id']}_robot.mp4")
                substituted.append(s["slot"])
        if not seg.exists():
            sys.exit(f"missing segment {seg}; run scripts/video/make_assets.py first")
        parts.append(seg)

    concat = SEGMENTS / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))

    v_kbps = min(1000, int((TARGET_BYTES * 8 / 1000 - AUDIO_KBPS * total) / total))
    audio_in = (["-i", str(args.narration)] if args.narration
                else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"])
    common = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
              *audio_in, "-map", "0:v:0", "-map", "1:a:0", "-shortest",
              "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p",
              "-r", str(FPS), "-b:v", f"{v_kbps}k", "-maxrate", f"{int(v_kbps * 1.4)}k",
              "-bufsize", f"{v_kbps * 2}k", "-movflags", "+faststart"]
    log = str(SEGMENTS / "x264pass")
    subprocess.run([*common, "-pass", "1", "-passlogfile", log, "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run([*common, "-pass", "2", "-passlogfile", log, "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k",
                    "-ac", "1", str(args.out)], check=True)

    info = probe(args.out)
    size, dur = int(info["format"]["size"]), float(info["format"]["duration"])
    st = info["streams"][0]
    print(f"{args.out}: {st['width']}x{st['height']} {st['codec_name']} {st['r_frame_rate']} fps, "
          f"{dur:.1f} s, {size / 1e6:.2f} MB  (video {v_kbps} kbps)")
    print(f"robot slots filled: {substituted or 'none (placeholder slates)'}")
    ok = size <= limits["max_bytes"] and dur <= limits["max_seconds"] + 0.5
    print("within limits" if ok else "OVER LIMIT")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
