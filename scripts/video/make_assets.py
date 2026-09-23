#!/usr/bin/env python3
"""Render every shot of `shotlist.json` into a 1280x720 / 24 fps segment.

    python scripts/video/make_assets.py            # all shots -> outputs/video/segments/
    python scripts/video/make_assets.py --only hero problem_clip
    python scripts/video/make_assets.py --frame hero 12.0   # one PNG at t=12 s, for a look

Sources are the per-episode debug videos (`<run>/<scene>/viz/debug/*.mp4`, 8 fps,
RGB+detections beside a costmap panel), the paper figures under `docs/figures/` and
`outputs/figures/`, and `episodes.jsonl`, which supplies every caption number: the floor
log, the FSM state, the presence events. Nothing here re-simulates anything.

Frame design for simulator footage: the first-person RGB fills the frame (the 1280x960
panel centre-cropped to 16:9 so the detection boxes stay large), the costmap crop sits as a
bottom-right inset, the badge at top-left reads `target · step · floor · state`, and the
bottom band carries the one-line explanation the narration is making at that moment.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (ACCENT, ASSETS, BLUE, FPS, GREEN, H, INK, MUTED, PAPER, RED, ROBOT_DIR,  # noqa: E402
                    SEGMENTS, W, WORKSPACE, Canvas, SegmentWriter, debug_clip_path, fade_edges,
                    font, letterbox, load_episode, load_shotlist, paragraph, text_size, to_bgr,
                    to_pil, wrap)

STATE_LABEL = {
    "goto_frontier": "explore to a frontier",
    "explore": "explore",
    "approach": "approach a candidate",
    "close_look": "inspect the surface",
    "climb": "take the stairs",
    "done": "stop",
}

MAP_INSET = (240, 300)   # max size of the costmap inset


# ----------------------------------------------------------------------------- sources

def _map_bbox(path: pathlib.Path, first: int, last: int) -> tuple[int, int, int, int]:
    """Where the map actually is inside the 480-px costmap panel: the non-grey bbox,
    sampled over the step range so the inset does not jump as the map grows."""
    cap = cv2.VideoCapture(str(path))
    xs, ys = [], []
    for idx in range(first, last + 1, max(1, (last - first) // 12)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        panel = f[:, 1280:].astype(int)
        grey = ((abs(panel[:, :, 0] - panel[:, :, 1]) < 6) & (abs(panel[:, :, 1] - panel[:, :, 2]) < 6)
                & (abs(panel[:, :, 0] - 128) < 12))
        nz = np.argwhere(~grey)
        if len(nz):
            (y0, x0), (y1, x1) = nz.min(0), nz.max(0)
            xs += [x0, x1]
            ys += [y0, y1]
    cap.release()
    if not xs:
        return 0, 0, 480, 960
    m = 8
    return max(0, min(xs) - m), max(0, min(ys) - m), min(480, max(xs) + m), min(960, max(ys) + m)


class ClipSource:
    """Sequential access to one debug video, re-laid out as (full RGB, map inset)."""

    def __init__(self, path: pathlib.Path, kind: str, first: int, last: int):
        self.path, self.kind = path, kind
        self.first, self.last = first, last
        self.cap = cv2.VideoCapture(str(path))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if last >= self.n_frames:
            raise ValueError(f"{path.name}: asked for frame {last}, has {self.n_frames}")
        self.bbox = _map_bbox(path, first, last) if kind == "our" else None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        self.idx = first - 1
        self.cur: tuple[np.ndarray, np.ndarray] | None = None

    def at(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx = min(idx, self.last)
        while self.idx < idx:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx += 1
            self.cur = self._relayout(f)
        assert self.cur is not None
        return self.cur

    def _relayout(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "our":
            rgb = f[120:840, :1280]
            x0, y0, x1, y1 = self.bbox
            m = f[y0:y1, 1280 + x0:1280 + x1]
        else:  # ASCENT's own panel: RGB 640x480 | obstacle map 480x480 | value map
            rgb = cv2.resize(f[:, :640], (1280, 960), interpolation=cv2.INTER_CUBIC)[120:840]
            m = f[:, 640:1120]
        s = min(MAP_INSET[0] / m.shape[1], MAP_INSET[1] / m.shape[0])
        m = cv2.resize(m, (max(1, int(m.shape[1] * s)), max(1, int(m.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
        return rgb, m

    def close(self) -> None:
        self.cap.release()


def _lookup(table: list, step: int, default=None):
    """Last entry whose step <= `step`, for step-indexed logs like state_log / floor_log."""
    out = default
    for row in table:
        if row[0] <= step:
            out = row[1]
    return out


# ----------------------------------------------------------------------------- shots

class Renderer:
    def __init__(self, data: dict):
        self.data = data
        self.meta = data["meta"]
        self.runs = {"our": self.meta["our_run"], "ascent": self.meta["ascent_run"]}

    def episode(self, key: str, run: str) -> tuple[dict, dict]:
        ep = self.data["episodes"][key]
        rec = load_episode(self.runs[run], ep["scene"], ep["id"])
        return ep, rec

    def source(self, key: str, run: str, steps: list[int]) -> ClipSource:
        ep = self.data["episodes"][key]
        path = debug_clip_path(self.runs[run], ep["scene"], ep["id"])
        # frame i of the debug video is step i+1 (the caption counts the trajectory)
        return ClipSource(path, run, steps[0] - 1, steps[1] - 1)

    # -- simulator footage -----------------------------------------------------------

    def _decorate(self, spec: dict, rec: dict, step: int, rgb: np.ndarray, m: np.ndarray,
                  scale: float = 1.0, label: str | None = "MD-SG") -> Image.Image:
        """One output frame: RGB base, map inset, badge, captions, events, insets.

        Positions follow the base image's own size, so a composite can decorate a cropped
        panel; `scale` grows the type for panels that are shrunk afterwards.
        """
        cv = Canvas(to_pil(rgb))
        W, H = cv.base.size
        f_badge, f_cap, f_small = font(int(24 * scale), "bold"), font(int(26 * scale)), font(int(19 * scale))
        target = spec.get("target") or rec["target"]
        state = STATE_LABEL.get(_lookup(rec.get("state_log", []), step, "explore"), "")
        floor = _lookup(spec.get("floors", []), step, "")
        head = f"{label} · " if label else ""
        cv.pill((16, 14), f"{head}target: {target} · step {step}", f_badge, bg=INK, alpha=200)
        sub = " · ".join(x for x in (floor, state) if x)
        if sub:
            cv.pill((16, 14 + int(46 * scale)), sub, f_small, fg=PAPER, bg=INK, alpha=170)
        # map inset, bottom-right by default, above the caption band
        mi = to_pil(m)
        if scale != 1.0:
            mi = mi.resize((int(mi.width * scale), int(mi.height * scale)), Image.LANCZOS)
        mx = 16 if spec.get("map_pos") == "bl" else W - mi.width - 16
        my = H - mi.height - int(70 * scale)
        cv.paste(mi, (mx, my), border=MUTED, width=1)
        cv.pill((mx + (0 if spec.get("map_pos") == "bl" else mi.width), my - 6), "costmap", f_small,
                bg=INK, alpha=150, pad=5, anchor="lb" if spec.get("map_pos") == "bl" else "rb")
        # belief bar
        belief = spec.get("belief")
        if belief:
            p = _lookup(belief, step)
            if p is not None:  # a null entry hides the bar: no logged value, nothing shown
                col = GREEN if p >= 0.5 else (ACCENT if p >= 0.3 else RED)
                bw = int(220 * scale)
                x0, y0 = W - 16 - bw - 20, 14
                cv.draw.rounded_rectangle((x0 - 10, y0, x0 + bw + 10, y0 + int(52 * scale)), radius=8,
                                          fill=(*INK, 200))
                cv.draw.text((x0, y0 + 6), f"presence belief p = {p:.2f}", font=f_small, fill=(*PAPER, 255))
                cv.draw.rectangle((x0, y0 + int(32 * scale), x0 + bw, y0 + int(44 * scale)),
                                  fill=(60, 60, 60, 255))
                cv.draw.rectangle((x0, y0 + int(32 * scale), x0 + int(bw * p), y0 + int(44 * scale)),
                                  fill=(*col, 255))
        # picture insets (verifier image, scene-graph tree)
        for ins in spec.get("insets", []):
            if ins["from"] <= step <= ins["to"]:
                im = self._inset_image(ins)
                x, y = ins["box"][0], ins["box"][1]
                cv.paste(im, (x, y), border=PAPER, width=2)
                if ins.get("label"):
                    cv.pill((x + im.width // 2, y + im.height + 8), ins["label"], f_small, bg=INK,
                            alpha=200, anchor="ct")
        # event flash: accent pill under the badge for `hold` steps
        for ev in spec.get("events", []):
            if ev[0] <= step < ev[0] + ev[2]:
                cv.pill((16, int(110 * scale)), ev[1], f_badge, fg=INK, bg=ACCENT, alpha=230)
        # caption band
        cap = None
        for c in spec.get("captions", []):
            if c[0] <= step <= c[1]:
                cap = c[2]
        if cap:
            cv.pill((W // 2, H - 14), cap, f_cap, bg=INK, alpha=200, anchor="cb")
        return cv.finish()

    _inset_cache: dict[str, Image.Image] = {}

    def _inset_image(self, ins: dict) -> Image.Image:
        key = json.dumps(ins, sort_keys=True)
        if key not in self._inset_cache:
            im = Image.open(WORKSPACE / ins["image"])
            w, h = ins["box"][2], ins["box"][3]
            self._inset_cache[key] = letterbox(im, (w, h), bg=INK, crop=ins.get("crop"))
        return self._inset_cache[key]

    def render_clip(self, spec: dict) -> Iterator[np.ndarray]:
        ep, rec = self.episode(spec["episode"], spec["run"])
        src = self.source(spec["episode"], spec["run"], spec["steps"])
        a, b = spec["steps"]
        n_out = int(round(spec["dur"] * FPS))
        n_hold = int(round(spec.get("hold", 0) * FPS))
        sps = (b - a + 1) / ((n_out - n_hold) / FPS)
        spec = {**spec, "target": ep["target"]}
        for k in range(n_out):
            step = min(b, a + int((k / FPS) * sps))
            rgb, m = src.at(step - 1)
            yield to_bgr(self._decorate(spec, rec, step, rgb, m))
        src.close()

    def render_composite(self, spec: dict) -> Iterator[np.ndarray]:
        sides = []
        for side in (spec["left"], spec["right"]):
            ep, rec = self.episode(side["episode"], side["run"])
            src = self.source(side["episode"], side["run"], side["steps"])
            sides.append((side, {**side, "target": ep["target"]}, rec, src))
        n_out = int(round(spec["dur"] * FPS))
        f_cap, f_end = font(26), font(30, "bold")
        for k in range(n_out):
            canvas = Image.new("RGB", (W, H), INK)
            for i, (side, sspec, rec, src) in enumerate(sides):
                a, b = side["steps"]
                step = min(b, a + int((k / FPS) * side["sps"]))
                rgb, m = src.at(step - 1)
                # centre 4:3 of the 16:9 panel: two of them fill the width at 640x480
                im = self._decorate(sspec, rec, step, rgb[:, 160:1120], m, scale=1.5, label=None)
                if step >= b:  # frozen: this side has ended
                    cv = Canvas(im)
                    col = GREEN if side.get("end_color") == "green" else RED
                    cv.pill((im.width // 2, im.height // 2 - 60), side["end"], font(46, "bold"),
                            fg=PAPER, bg=col, alpha=225, pad=20, anchor="ct")
                    im = cv.finish()
                im = im.resize((W // 2, 480), Image.LANCZOS)
                canvas.paste(im, (i * (W // 2), 90))
            cv = Canvas(canvas)
            for i, (side, *_) in enumerate(sides):  # method labels in the band above each panel
                cv.pill((i * (W // 2) + W // 4, 42), side["label"], font(26, "bold"), fg=INK,
                        bg=ACCENT if i == 0 else BLUE, alpha=255, anchor="ct")
            cv.pill((W // 2, H - 24), spec["caption"], f_cap, bg=INK, alpha=200, anchor="cb")
            yield to_bgr(cv.finish())
        for *_, src in sides:
            src.close()

    # -- stills and cards ----------------------------------------------------------------

    def render_still(self, spec: dict) -> Iterator[np.ndarray]:
        im = letterbox(Image.open(WORKSPACE / spec["src"]), (W, H - 90), bg=INK, crop=spec.get("crop"))
        canvas = Image.new("RGB", (W, H), INK)
        canvas.paste(im, (0, 0))
        cv = Canvas(canvas)
        if spec.get("caption"):
            cv.pill((W // 2, H - 18), spec["caption"], font(26), bg=(40, 40, 44), alpha=255, anchor="cb")
        frame = to_bgr(cv.finish())
        for _ in range(int(round(spec["dur"] * FPS))):
            yield frame

    def _bg_frame(self, bg: dict | None) -> Image.Image:
        if not bg:
            return Image.new("RGB", (W, H), INK)
        src = self.source(bg["episode"], "our", [bg["step"], bg["step"]])
        rgb, _ = src.at(bg["step"] - 1)
        src.close()
        im = to_pil(rgb)
        if bg.get("blur"):
            im = im.filter(ImageFilter.GaussianBlur(bg["blur"]))
        return Image.blend(im, Image.new("RGB", (W, H), INK), 0.55)

    def render_slate(self, spec: dict) -> Iterator[np.ndarray]:
        card = getattr(self, f"card_{spec['card']}")
        frame = to_bgr(card(spec))
        for _ in range(int(round(spec["dur"] * FPS))):
            yield frame

    def card_title(self, spec: dict) -> Image.Image:
        cv = Canvas(self._bg_frame(spec.get("bg")))
        d = cv.draw
        d.text((80, 200), self.meta["title"], font=font(110, "bold"), fill=(*PAPER, 255))
        y = paragraph(d, (84, 340), self.meta["subtitle"], font(34), 1100, fill=(*PAPER, 255))
        d.text((84, y + 30), self.meta["venue"], font=font(26), fill=(*MUTED, 255))
        if self.meta.get("authors"):
            d.text((84, y + 70), self.meta["authors"], font=font(26), fill=(*PAPER, 255))
        return cv.finish()

    def card_contributions(self, spec: dict) -> Image.Image:
        canvas = Image.new("RGB", (W, H), INK)
        fig = letterbox(Image.open(WORKSPACE / "docs/figures/full_pipeline.png"), (W - 80, 400), bg=INK)
        canvas.paste(fig, (40, 56))
        cv = Canvas(canvas)
        d = cv.draw
        d.text((40, 14), "Contributions", font=font(30, "bold"), fill=(*ACCENT, 255))
        bullets = [
            "Dynamic multi-floor ObjectNav: objects move within or across floors, so mapped locations can no longer be treated as fixed.",
            "MD-SG: an online, open-vocabulary 3D scene-graph pipeline that revises object-location beliefs and reasons over hypotheses across floors.",
            "The first ObjectNav benchmark with controlled same-floor and cross-floor relocation; improved reacquisition over DualMap.",
        ]
        colw = (W - 80 - 2 * 30) // 3
        for i, b in enumerate(bullets):
            x = 40 + i * (colw + 30)
            d.text((x, 478), f"{i + 1}", font=font(40, "bold"), fill=(*ACCENT, 255))
            paragraph(d, (x + 40, 486), b, font(21), colw - 40, fill=(*PAPER, 255), spacing=6)
        return cv.finish()

    def card_eq3(self, spec: dict) -> Image.Image:
        cv = Canvas(Image.new("RGB", (W, H), INK))
        d = cv.draw
        d.text((60, 50), "Presence belief  (Eq. 3)", font=font(38, "bold"), fill=(*ACCENT, 255))
        d.text((60, 105), "log-odds L per mapped object,  p = σ(L),  clipped to [−6, +3]",
               font=font(26), fill=(*MUTED, 255))
        rows = [
            ("detected, visible", "ΔL = log( r / q )", "positive", PAPER, False),
            ("missed, visible", "ΔL = log( (1−r) / (1−q) )", "negative — the channel a static map lacks", PAPER, True),
            ("not visible", "ΔL = 0", "unobserved is not observed-absent", MUTED, False),
        ]
        y = 190
        for a, b, c, col, hi in rows:
            if hi:
                d.rounded_rectangle((44, y - 14, W - 44, y + 66), radius=10, fill=(*ACCENT, 60))
            d.text((60, y), a, font=font(30, "bold"), fill=(*col, 255))
            d.text((420, y), b, font=font(30, mono=True), fill=(*col, 255))
            d.text((60, y + 36), c, font=font(22), fill=(*(ACCENT if hi else MUTED), 255))
            y += 110
        paragraph(d, (60, 540),
                  "visible = a large, readable projection in range with nothing nearer in the depth map. "
                  "Nearer depth means occlusion and no update; the low +3 ceiling is what lets a few honest misses overturn a long history.",
                  font(22), W - 120, fill=(*PAPER, 255), spacing=6)
        return cv.finish()

    def _table_card(self, spec: dict, title: str, rows: list[tuple], header: tuple,
                    note: str, y0: int = 150, ours_idx: int | None = None) -> Canvas:
        cv = Canvas(self._bg_frame(spec.get("bg")))
        d = cv.draw
        d.text((60, 50), title, font=font(30, "bold"), fill=(*ACCENT, 255))
        cols = [60, 520, 760, 1000]
        for x, h_ in zip(cols, header):
            d.text((x, y0), h_, font=font(22), fill=(*MUTED, 255))
        y = y0 + 44
        for i, row in enumerate(rows):
            ours = i == ours_idx
            fnt = font(34, "bold" if ours else "regular")
            col = ACCENT if ours else PAPER
            for x, cell in zip(cols, row):
                d.text((x, y), str(cell), font=fnt, fill=(*col, 255))
            y += 60
        if note:
            paragraph(d, (60, H - 80), note, font(20), W - 120, fill=(*MUTED, 255))
        return cv

    def card_table2(self, spec: dict) -> Image.Image:
        cv = self._table_card(
            spec, "Multi-floor cross-anchor relocation · HM3D v0.2 · 5 scenes · 25 episodes  (Table II)",
            [("ASCENT", "25", "0.0 %", "0.000"), ("MD-SG (ours)", "25", "40.0 %", "0.162")],
            ("method", "episodes", "success", "SPL"),
            "1000-step budget. Every episode starts from a map built on the static layout; the target has been moved to another storey.",
            y0=200, ours_idx=1)
        return cv.finish()

    def card_table1(self, spec: dict) -> Image.Image:
        cv = self._table_card(
            spec, "DualMap's benchmark · single floor · cross-anchor · 53 episodes  (Table I)",
            [("DualMap (re-evaluated)", "", "30.2 %", ""), ("MD-SG (ours)", "", "52.8 %", "")],
            ("method", "", "success", ""),
            "", y0=120, ours_idx=1)
        d = cv.draw
        d.text((60, 330), "Ablation, same 53 episodes  (Table III)", font=font(26, "bold"), fill=(*ACCENT, 255))
        bars = [("w/o presence-belief revision", 41.5), ("w/o belief-guided surface scoring", 35.8), ("full system", 52.8)]
        y = 380
        for name, v in bars:
            d.text((60, y), name, font=font(24), fill=(*PAPER, 255))
            d.rectangle((520, y + 4, 520 + int(v * 9), y + 30), fill=(*(ACCENT if v == 52.8 else MUTED), 255))
            d.text((530 + int(v * 9), y), f"{v:.1f} %", font=font(24, "bold"), fill=(*PAPER, 255))
            y += 56
        paragraph(d, (60, H - 80), "All seven episodes lost without presence revision commit to a stale prior-map track that never decays.",
                  font(20), W - 120, fill=(*MUTED, 255))
        return cv.finish()

    def card_topology(self, spec: dict) -> Image.Image:
        cv = Canvas(Image.new("RGB", (W, H), INK))
        d = cv.draw
        d.text((60, 40), "On the robot", font=font(34, "bold"), fill=(*ACCENT, 255))
        boxes = [
            ((60, 130, 560, 330), "osg-thor  (Jetson AGX Thor, L4T/CUDA)",
             "the pipeline, the scene graph, the perception models, ollama.  No ROS, no Habitat."),
            ((60, 380, 560, 520), "osg-bridge  (ros:humble)", "rclpy and nothing else; talks to osg-thor over a local socket."),
            ((760, 130, 1220, 520), "Stretch 3", "Nav2, stretch_driver, RealSense, TF.  Keeps its own low-level control."),
        ]
        for (x0, y0, x1, y1), head, body in boxes:
            d.rounded_rectangle((x0, y0, x1, y1), radius=14, outline=(*PAPER, 255), width=2)
            d.text((x0 + 20, y0 + 18), head, font=font(24, "bold"), fill=(*PAPER, 255))
            paragraph(d, (x0 + 20, y0 + 60), body, font(21), x1 - x0 - 40, fill=(*MUTED, 255))
        d.line((310, 330, 310, 380), fill=(*ACCENT, 255), width=4)
        d.text((325, 340), "socket", font=font(19), fill=(*ACCENT, 255))
        d.line((560, 450, 760, 450), fill=(*ACCENT, 255), width=4)
        d.text((620, 415), "DDS", font=font(19), fill=(*ACCENT, 255))
        d.text((580, 250), "goal pose → NavigateToPose", font=font(19), fill=(*PAPER, 255))
        d.text((580, 280), "← RGB-D, TF", font=font(19), fill=(*PAPER, 255))
        paragraph(d, (60, 570),
                  "The pipeline keeps doing all of the thinking and stops steering. Nav2's map is 2-D, so the storey is declared by the operator on /osg/floor.",
                  font(24), W - 120, fill=(*PAPER, 255))
        return cv.finish()

    def card_close(self, spec: dict) -> Image.Image:
        bg = Image.new("RGB", (W, H), INK)
        robot = ROBOT_DIR / f"{spec.get('robot_bg', 'R3')}.mp4"
        if robot.exists():
            cap = cv2.VideoCapture(str(robot))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
            ok, f = cap.read()
            cap.release()
            if ok:
                bg = Image.blend(letterbox(to_pil(f), (W, H), bg=INK), bg, 0.55)
        cv = Canvas(bg)
        d = cv.draw
        paragraph(d, (80, 200), "Revisable, floor-aware scene memory is a first-class design axis for object navigation in buildings that change.",
                  font(40, "bold"), W - 160, fill=(*PAPER, 255), spacing=10)
        d.text((80, 470), f"{self.meta['title']} · {self.meta['venue']}", font=font(26), fill=(*MUTED, 255))
        if self.meta.get("code_url"):
            d.text((80, 510), self.meta["code_url"], font=font(26), fill=(*ACCENT, 255))
        return cv.finish()

    def render_robot(self, spec: dict) -> Iterator[np.ndarray]:
        cv = Canvas(Image.new("RGB", (W, H), (30, 22, 18)))
        d = cv.draw
        d.text((80, 160), f"[ ROBOT FOOTAGE  {spec['slot'].replace('_', '.')}  ·  {spec['dur']} s ]",
               font=font(48, "bold"), fill=(*ACCENT, 255))
        paragraph(d, (80, 260), spec["desc"], font(32), W - 160, fill=(*PAPER, 255), spacing=10)
        d.text((80, H - 90), f"drop the clip at outputs/video/robot/{spec['slot']}.mp4 and rebuild",
               font=font(22), fill=(*MUTED, 255))
        frame = to_bgr(cv.finish())
        for _ in range(int(round(spec["dur"] * FPS))):
            yield frame

    # -- driver --------------------------------------------------------------------------

    def frames(self, spec: dict) -> Iterator[np.ndarray]:
        return getattr(self, f"render_{spec['kind']}")(spec)

    def render(self, spec: dict) -> pathlib.Path:
        out = SEGMENTS / f"{spec['id']}.mp4"
        n = int(round(spec["dur"] * FPS))
        w = SegmentWriter(out)
        for f in fade_edges(self.frames(spec), n):
            w.write(f)
        w.close()
        if w.n != n:
            raise RuntimeError(f"{spec['id']}: wrote {w.n} frames, expected {n}")
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="shot ids to render (default: all)")
    ap.add_argument("--frame", nargs=2, metavar=("SHOT", "T"), help="write one PNG at T seconds into SHOT")
    args = ap.parse_args()

    data = load_shotlist()
    r = Renderer(data)
    by_id = {s["id"]: s for s in data["shots"]}
    if args.frame:
        spec, t = by_id[args.frame[0]], float(args.frame[1])
        k = int(t * FPS)
        for i, f in enumerate(r.frames(spec)):
            if i == k:
                ASSETS.mkdir(parents=True, exist_ok=True)
                out = ASSETS / f"frame_{spec['id']}_{t:.1f}s.png"
                cv2.imwrite(str(out), f)
                print(out)
                break
        return

    SEGMENTS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for spec in data["shots"]:
        if args.only and spec["id"] not in args.only:
            continue
        out = r.render(spec)
        manifest[spec["id"]] = {"segment": str(out.relative_to(WORKSPACE)), "dur": spec["dur"],
                                "kind": spec["kind"], "t_start": spec["t_start"]}
        print(f"{spec['t_start']:6.1f}s  {spec['id']:<16} {spec['dur']:>4}s  -> {out.name}")
    with (SEGMENTS / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"total {data['total']:.1f} s of {data['meta']['limits']['max_seconds']} s")


if __name__ == "__main__":
    main()
