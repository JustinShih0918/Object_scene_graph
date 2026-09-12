#!/usr/bin/env python3
"""Can ANY encoder read the crops FastSAM finds, for the objects we fail?

`probe_region_ranking.py` showed the proposal is not the bottleneck for
scissors: FastSAM puts a tight region on it on 72% of frames and MobileCLIP-S2
then scores that region 0.149, BELOW a typical distractor, ranking it first on
1 of 34 frames. The proposal stage is only worth building if some encoder or
crop treatment can read those regions.

Four factors, swept together on cached regions so FastSAM runs once per frame:

  backbone   mobileclip_s2 (DualMap's own) vs vitb32 (OpenAI CLIP)
  pad        context around the tight box, as a fraction of its longer side.
             A thin object in a tight box is mostly background; too much
             context and the crop becomes the furniture.
  shape      `raw` crops the box and lets the preprocess squash it to square,
             which distorts a long thin object; `square` takes a square window
             about the box centre so the aspect ratio survives.
  prompt     CLIP is prompt-sensitive and `scissors` is an awkward noun --
             "a pair of scissors" is the phrase the caption corpus would use.

Reports rank-1 of the true region among all of a frame's regions, per query.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_perception_gap import frames, covers, area  # noqa: E402

PROMPTS = {
    "plain": "{q}",
    "photo": "a photo of a {q}",
    "natural": None,   # a per-query phrasing, below
}
NATURAL = {
    "scissors": "a pair of scissors",
    "soup_can": "a tin can of soup",
    "cracker_box": "a cardboard box of crackers",
    "banana": "a ripe banana",
    "mug": "a coffee mug",
    "pitcher": "a plastic water pitcher",
}


def crop_of(img: Image.Image, box, pad_frac: float, shape: str) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = pad_frac * max(w, h)
    if shape == "square":
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(w, h) / 2.0 + pad
        box2 = (cx - side, cy - side, cx + side, cy + side)
    else:
        box2 = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    W, H = img.size
    box2 = (max(0, int(box2[0])), max(0, int(box2[1])),
            min(W, int(np.ceil(box2[2]))), min(H, int(np.ceil(box2[3]))))
    if box2[2] - box2[0] < 2 or box2[3] - box2[1] < 2:
        box2 = (max(0, int(x1)), max(0, int(y1)), min(W, int(x2) + 2), min(H, int(y2) + 2))
    return img.crop(box2).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=Path("outputs/osg_dualmap_tightring/gt_dump"))
    ap.add_argument("--queries", nargs="+", default=["scissors", "soup_can", "cracker_box", "banana"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--backbones", nargs="+", default=["mobileclip_s2", "vitb32"])
    ap.add_argument("--pads", nargs="+", type=float, default=[0.0, 0.25, 0.6])
    ap.add_argument("--shapes", nargs="+", default=["raw", "square"])
    ap.add_argument("--prompts", nargs="+", default=["photo", "natural"])
    ap.add_argument("--tight-frac", type=float, default=0.02)
    ap.add_argument("--tight-ratio", type=float, default=0.25)
    ap.add_argument("--max-regions", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/CROP_ENCODING.md"))
    args = ap.parse_args()

    from ultralytics import FastSAM
    sam = FastSAM("outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt")

    # Stage 1: regions once per frame, with the true ones flagged.
    cache: List[dict] = []
    work = frames(args.dump, args.queries, args.limit)
    for query, items in work.items():
        for it in items:
            img = Image.open(it["path"]).convert("RGB")
            W, H = img.size
            r = sam.predict(np.asarray(img), imgsz=1024, conf=0.25, iou=0.7,
                            retina_masks=True, verbose=False, device=args.device)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = [[float(x) for x in b] for b in r.boxes.xyxy.detach().cpu().numpy()][: args.max_regions]
            cov = [i for i, b in enumerate(boxes) if covers(b, it["u"], it["v"])]
            if not cov:
                continue
            big = max(area(boxes[i]) for i in cov)
            tight = [i for i in cov
                     if area(boxes[i]) <= args.tight_frac * W * H
                     and area(boxes[i]) <= args.tight_ratio * big]
            if not tight:
                continue
            cache.append({"query": query, "path": it["path"], "boxes": boxes,
                          "truth": set(tight), "range_m": it["range_m"]})
    print(f"cached {len(cache)} frames with a tight region")

    results: Dict[tuple, Dict[str, List[int]]] = {}
    from osg.perception.feature_encoder import FeatureEncoder
    for backbone in args.backbones:
        ck = "data/clip/ViT-B-32.pt" if backbone == "vitb32" else ""
        enc = FeatureEncoder(backbone, ck, args.device, half=True)
        for pad, shape, pname in itertools.product(args.pads, args.shapes, args.prompts):
            ranks: Dict[str, List[int]] = {}
            for fr in cache:
                q = fr["query"]
                text_s = NATURAL[q] if pname == "natural" else PROMPTS[pname].format(q=q.replace("_", " "))
                text = enc.text_feature(text_s)
                img = Image.open(fr["path"]).convert("RGB")
                crops = [crop_of(img, b, pad, shape) for b in fr["boxes"]]
                feats = enc.encode_images([np.asarray(c) for c in crops])
                sims = feats @ text
                idx = [i for i in fr["truth"] if i < len(sims)]
                if not idx:
                    continue
                bt = max(float(sims[i]) for i in idx)
                rank = 1 + sum(1 for i in range(len(sims)) if i not in fr["truth"] and float(sims[i]) > bt)
                ranks.setdefault(q, []).append(rank)
            results[(backbone, pad, shape, pname)] = ranks
            tot = [r for v in ranks.values() for r in v]
            print(f"  {backbone:<14} pad={pad:<4} {shape:<6} {pname:<8} "
                  f"rank1 {sum(1 for r in tot if r==1)}/{len(tot)}")

    L: List[str] = []
    def say(s: str = "") -> None:
        print(s); L.append(s)
    say("# Can any encoder read the crops FastSAM finds?")
    say()
    say(f"{len(cache)} frames that HAVE a tight region on the object. rank-1 = appearance")
    say("picked that region over every other region in the frame.")
    say()
    header = "| backbone | pad | shape | prompt | " + " | ".join(args.queries) + " | all |"
    say(header)
    say("|---|---:|---|---|" + "---:|" * (len(args.queries) + 1))
    best = None
    for key, ranks in results.items():
        backbone, pad, shape, pname = key
        cells = []
        for q in args.queries:
            v = ranks.get(q, [])
            cells.append(f"{sum(1 for r in v if r==1)}/{len(v)}" if v else "-")
        tot = [r for v in ranks.values() for r in v]
        n1 = sum(1 for r in tot if r == 1)
        cells.append(f"**{n1}/{len(tot)}**")
        say(f"| {backbone} | {pad} | {shape} | {pname} | " + " | ".join(cells) + " |")
        sc = ranks.get("scissors", [])
        s1 = sum(1 for r in sc if r == 1)
        if best is None or (s1, n1) > best[0]:
            best = ((s1, n1), key)
    say()
    if best:
        say(f"Best for scissors: `{best[1]}` -- {best[0][0]} rank-1 scissors frames, "
            f"{best[0][1]} overall.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
