"""Class-agnostic region proposal: find the object the detector will not name."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegionProposalConfig:
    """Off by default; every field is inert while `enabled` is False.

    Measured justification for each number is in
    `src/osg/perception/region_proposer.py` and `docs/PERCEPTION_GAP.md`.
    """

    enabled: bool = False
    weights: str = "outputs/dualmap_comparison/vendor/DualMap/model/FastSAM-s.pt"
    backbone: str = "mobileclip_s2"     # beat ViT-B/32 at every setting probed
    checkpoint: str = ""                # mobileclip resolves from the HF cache
    device: str = "cuda"
    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.7
    # Admission. 0.24 is the knee of the precision curve measured over 166
    # frames: 40 correct against 7 wrong, 85% precision, 24% recall. At 0.22
    # precision falls to 64%, at 0.28 recall falls to 9 frames.
    tau: float = 0.24
    # A detection's score is what every downstream gate is calibrated against,
    # so an admitted region is handed the detector's own "good enough" value
    # rather than its cosine, which would read as almost-certainly-false.
    admit_score: float = 0.50
    # Padding the box by a quarter of its longer side, with the natural phrasing
    # in region_proposer.NATURAL, moved overall rank-1 from 52% to 70%.
    pad_frac: float = 0.25
    # The size band a proposal must fall in to be a candidate for the object
    # rather than the furniture it rests on.
    min_area_px: float = 200.0
    max_area_frac: float = 0.02
    max_regions: int = 64
    # Run the stage only when the label path found nothing for the target on
    # this keyframe, and at most this many times per episode. FastSAM is 13 ms
    # a frame but encoding ~60 crops is not, and an unbounded stage would admit
    # on 15% of frames at 85% precision, which is a lot of wrong tracks.
    only_when_unnamed: bool = True
    max_per_episode: int = 40
