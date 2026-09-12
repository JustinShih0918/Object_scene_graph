"""Stair detection: where can the agent change storey?

Two signals, and the ordering between them is set by measurement rather than
preference. On the 100-episode v1 run the YOLOE `stairs` class fired in only
**15% of multi-floor episodes** (12 tracks / 100 episodes) -- high quality when
it fired (median score 0.60, median 26 observations) but far too sparse to
gate a floor transition on. So:

* **Geometry leads.** ZONDA's criterion: on a coarse grid, a cell is steppable
  when the largest height difference to its 8 neighbours is below the agent's
  climb limit. Flat floor gives ~0, a wall gives metres, a staircase sits in
  between at roughly its riser height. No model, no weights, works on any
  observed surface.
* **Semantics confirm.** A `stairs` track in the object layer raises confidence
  for cells near it, but is never required.

The output is deliberately conservative. Confirmed cells get written FREE in
the costmap and exempted from the obstacle stamp, and
`docs/INVESTIGATION.md` records that *every* previous attempt to soften the
costmap's obstacle writes (hit-count corroboration, speckle filtering) cost
more real geometry than it gained. So a cell must clear the geometric test AND
belong to a component large enough to be an actual staircase before it is
touched, and the total relabelled area is capped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.labels import normalize_label
from .costmap import FREE, HEIGHT_AXIS, PLANE, Costmap2D, block_min, grow_aligned
from ..core.geometry import backproject
from ..core.types import Detection, FrameData

STAIR_LABELS = ("stairs", "staircase", "stair")


@dataclass
class StairRegion:
    """A connected run of steppable-but-not-flat cells."""

    cells_rc: np.ndarray          # (N, 2) indices into the FINE costmap grid
    centroid_xy: np.ndarray       # world (x, z)
    n_cells: int
    mean_dh: float
    low_y: float
    high_y: float
    semantic: bool = False        # a `stairs` track corroborates it

    @property
    def rise_m(self) -> float:
        return float(self.high_y - self.low_y)


def stair_tracks(object_layer, min_obs: int = 2, min_evidence: float = 1.0) -> List[np.ndarray]:
    """3D centres of `stairs` tracks that clear the existing corroboration bar.

    Reuses the object layer's own multi-frame accumulation (`evidence`, already
    tuned over 1343 tracks) rather than inventing a second one.
    """
    out = []
    for t in object_layer.tracks(include_blacklisted=True):
        if normalize_label(t.label) not in STAIR_LABELS:
            continue
        if t.n_obs >= min_obs and float(getattr(t, "evidence", 0.0)) >= min_evidence:
            out.append(np.asarray(object_layer.center_of(t), dtype=float))
    return out


def detect_stairs(
    costmap: Costmap2D,
    climb_limit_m: float = 0.2,
    min_dh_m: float = 0.03,
    cell_m: float = 0.1,
    min_cells: int = 12,
    min_rise_m: float = 1.0,
    semantic_centers: Optional[Sequence[np.ndarray]] = None,
    semantic_radius_m: float = 1.5,
    require_semantic: bool = False,
) -> List[StairRegion]:
    """Steppable, non-flat regions of the costmap.

    `min_dh_m` excludes flat floor (which would otherwise make the whole map a
    "staircase"); `climb_limit_m` excludes walls and should match Habitat's
    navmesh `max_climb` so the costmap and the navmesh agree on what the
    embodiment can step over.

    `min_rise_m` is the one that does the real work, and 1.0 is measured rather
    than guessed. The per-cell test alone is satisfied by thresholds, ramps,
    sloped floor and low furniture: on the 35-episode SINGLE-floor gate it fired
    in 28/35 episodes, relabelling 43k cells in scenes with no storey to reach.
    Every one of those false regions rose 0.33-0.75 m, while a staircase that
    actually connects HM3D storeys must climb 2.5-3.4 m. Demanding a metre of
    rise separates them cleanly. The cost is that a staircase glimpsed only in
    part stays unconfirmed until enough of its run is observed -- the safe
    direction, since the relabel is permanent.
    """
    from scipy import ndimage

    dh, valid = costmap.height_gradient(cell_m=cell_m)
    steppable = valid & np.isfinite(dh) & (dh > min_dh_m) & (dh < climb_limit_m)
    if not steppable.any():
        return []

    block = max(1, int(round(cell_m / costmap.resolution)))
    labels, n = ndimage.label(steppable, structure=np.ones((3, 3)))
    coarse_h = block_min(costmap.height, block)

    regions: List[StairRegion] = []
    for lbl in range(1, n + 1):
        coarse_rc = np.argwhere(labels == lbl)
        if coarse_rc.shape[0] * (block ** 2) < min_cells:
            continue
        ys = coarse_h[labels == lbl]
        ys = ys[np.isfinite(ys)]
        if ys.size == 0 or float(ys.max() - ys.min()) < min_rise_m:
            continue

        fine_rc = _expand_to_fine(coarse_rc, block, costmap.grid.shape)
        if fine_rc.shape[0] == 0:
            continue
        centroid = costmap.grid_to_world(fine_rc.mean(axis=0))

        semantic = False
        if semantic_centers is not None and len(semantic_centers):
            xy = np.stack([np.asarray(c, float)[list(PLANE)] for c in semantic_centers])
            semantic = bool(
                (np.linalg.norm(xy - centroid[None, :], axis=1) <= semantic_radius_m).any()
            )
        if require_semantic and not semantic:
            continue

        regions.append(
            StairRegion(
                cells_rc=fine_rc,
                centroid_xy=centroid,
                n_cells=int(fine_rc.shape[0]),
                mean_dh=float(np.nanmean(dh[labels == lbl])),
                low_y=float(ys.min()),
                high_y=float(ys.max()),
                semantic=semantic,
            )
        )
    regions.sort(key=lambda r: -r.n_cells)
    return regions


@dataclass
class Flight:
    """A run of treads between this storey and the next, read off the height
    layer rather than a detector mask.

    `cells_rc` index the costmap grid; `heights` are absolute world y per cell.
    `foot_xy` is the lowest tread (the mouth on this floor for an ascent),
    `top_xy` the highest. `kind` is "up" or "down" relative to `floor_y`.
    """

    kind: str
    cells_rc: np.ndarray
    heights: np.ndarray
    foot_xy: np.ndarray
    top_xy: np.ndarray
    span_m: float

    @property
    def n_cells(self) -> int:
        return int(self.cells_rc.shape[0])


def find_flights(
    costmap: Costmap2D,
    floor_y: float,
    new_level_m: float = 1.8,
    min_span_m: float = 1.0,
    min_cells: int = 150,
    max_cells: int = 4000,
    margin_m: float = 0.2,
) -> List[Flight]:
    """Staircases as connected runs of INTERMEDIATE-height cells.

    Stage 4's detector looked for steppable cells by their local height step
    and failed structurally: a tread's interior is flat, fell below its
    `min_dh`, and the flight fragmented into riser edges. This asks a different
    question. The height layer keeps the lowest surface seen per cell within a
    storey of the floor, so every tread of a flight is a cell whose height sits
    strictly BETWEEN this floor and the next -- and those cells are contiguous,
    because treads touch. Connect them, and keep a component only if its
    heights SPAN at least `min_span_m`: a table is a plateau and spans nothing,
    a sloped floor is large and is capped by `max_cells`, and a flight that
    joins two storeys spans a metre or more by construction.

    The detector's `stairs` mask, seen from below, stamps cells under the upper
    flight: on 00821 its target sat 2.7 m beside the true foot. The lowest cell
    of an intermediate-height component IS the foot, whatever the mask thinks.
    """
    from scipy import ndimage

    if costmap.height is None:
        return []
    rel = costmap.height - float(floor_y)
    seen = np.isfinite(rel)
    lo, hi = float(margin_m), float(new_level_m) - float(margin_m)
    out: List[Flight] = []
    for kind, mask in (
        ("up", seen & (rel > lo) & (rel < hi)),
        ("down", seen & (rel < -lo) & (rel > -hi)),
    ):
        if not mask.any():
            continue
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        for label in range(1, n + 1):
            rc = np.argwhere(labels == label)
            if rc.shape[0] < min_cells or rc.shape[0] > max_cells:
                continue
            h = costmap.height[rc[:, 0], rc[:, 1]]
            span = float(h.max() - h.min())
            if span < min_span_m:
                continue
            lowest, highest = int(np.argmin(h)), int(np.argmax(h))
            if kind == "up":
                foot, top = rc[lowest], rc[highest]
            else:
                # Descending, the mouth on THIS floor is the highest tread.
                foot, top = rc[highest], rc[lowest]
            out.append(Flight(
                kind=kind, cells_rc=rc, heights=h,
                foot_xy=costmap.grid_to_world(foot.astype(float)),
                top_xy=costmap.grid_to_world(top.astype(float)),
                span_m=span,
            ))
    return out


def apply_stair_mask(
    costmap: Costmap2D, regions: Sequence[StairRegion], max_area_frac: float = 0.05
) -> int:
    """Mark confirmed stair cells traversable. Returns the number of cells set.

    `max_area_frac` caps the relabelled area as a fraction of the KNOWN map: a
    runaway mask would carve free space through real obstacles and, because the
    exemption is permanent, could never be undone.
    """
    if not regions:
        return 0
    if costmap.stair_mask is None:
        costmap.stair_mask = np.zeros(costmap.grid.shape, dtype=bool)

    known = int((costmap.grid != -1).sum())
    # A zero budget means "relabel nothing", not "no limit" -- this is a safety
    # cap on a permanent, irreversible write, so it must fail closed.
    budget = int(max_area_frac * known)
    added = 0
    for region in regions:
        rc = region.cells_rc
        if added + rc.shape[0] > budget:
            break
        costmap.stair_mask[rc[:, 0], rc[:, 1]] = True
        added += rc.shape[0]
    if added:
        costmap.grid[costmap.stair_mask] = FREE
    return added


# ---------------------------------------------------------------------- helpers


def _expand_to_fine(coarse_rc: np.ndarray, block: int, shape: Tuple[int, int]) -> np.ndarray:
    """Coarse cell indices -> every fine cell they cover."""
    if coarse_rc.shape[0] == 0:
        return np.zeros((0, 2), dtype=int)
    offs = np.array([(i, j) for i in range(block) for j in range(block)])
    fine = (coarse_rc[:, None, :] * block + offs[None, :, :]).reshape(-1, 2)
    h, w = shape
    keep = (fine[:, 0] >= 0) & (fine[:, 0] < h) & (fine[:, 1] >= 0) & (fine[:, 1] < w)
    return fine[keep]


@dataclass
class StairDetection:
    kind: str
    centroid_xy: np.ndarray
    cells: np.ndarray
    n_cells: int
    support: int


class StairDetector:
    """Accumulated per-floor up/down stair evidence used by ASCENT policies."""

    def __init__(
        self,
        resolution_m: float = 0.05,
        min_hits: int = 3,
        min_cells: int = 25,
        max_range_m: float = 5.0,
        up_mode: str = "detector",
        close_iters: int = 2,
        disable_margin_m: float = 0.5,
        **_kwargs,
    ) -> None:
        self.resolution_m = float(resolution_m)
        self.min_hits = int(min_hits)
        self.min_cells = int(min_cells)
        self.max_range_m = float(max_range_m)
        self.up_mode = str(up_mode)
        self.close_iters = int(close_iters)
        self.disable_margin_m = float(disable_margin_m)

    @staticmethod
    def _ensure_grids(layer) -> None:
        if layer.up_stair_hits is not None:
            return
        shape = layer.costmap.grid.shape
        layer.up_stair_hits = np.zeros(shape, dtype=np.int16)
        layer.down_stair_hits = np.zeros(shape, dtype=np.int16)
        layer.disabled_stair = np.zeros(shape, dtype=bool)

        def on_grow(h, w, off_r, off_c, current=layer):
            current.up_stair_hits = grow_aligned(
                current.up_stair_hits, h, w, off_r, off_c
            )
            current.down_stair_hits = grow_aligned(
                current.down_stair_hits, h, w, off_r, off_c
            )
            current.disabled_stair = grow_aligned(
                current.disabled_stair, h, w, off_r, off_c, fill=False
            )

        layer.costmap.add_grow_listener(on_grow)

    @staticmethod
    def _stamp(hits: np.ndarray, costmap: Costmap2D, pts: np.ndarray) -> None:
        if pts.shape[0] == 0:
            return
        rc = costmap.world_to_grid(pts[:, list(PLANE)])
        h, w = hits.shape
        ok = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
        rc = np.unique(rc[ok], axis=0)
        if len(rc):
            hits[rc[:, 0], rc[:, 1]] += 1

    def accumulate(
        self,
        frame: FrameData,
        layer,
        dets: Optional[List[Detection]] = None,
        seg_stair_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._ensure_grids(layer)
        pts = backproject(
            frame.depth, frame.intrinsics, frame.T_wc,
            stride=2, max_depth=self.max_range_m,
        )
        if len(pts):
            drop = layer.floor_y - pts[:, HEIGHT_AXIS]
            self._stamp(layer.down_stair_hits, layer.costmap, pts[(drop > 0.35) & (drop < 4.0)])
        masks = []
        if self.up_mode == "rednet" and seg_stair_mask is not None:
            masks = [seg_stair_mask]
        else:
            masks = [d.mask for d in (dets or []) if normalize_label(d.label) in STAIR_LABELS]
        for mask in masks:
            up = backproject(
                frame.depth, frame.intrinsics, frame.T_wc,
                mask=mask, stride=2, max_depth=self.max_range_m,
            )
            self._stamp(layer.up_stair_hits, layer.costmap, up)

    def disable(self, layer, cells: np.ndarray) -> None:
        from scipy import ndimage

        self._ensure_grids(layer)
        if not len(cells):
            return
        mask = np.zeros(layer.disabled_stair.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        radius = max(1, int(round(self.disable_margin_m / self.resolution_m)))
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=radius)
        layer.disabled_stair |= mask
        layer.up_stair_hits[mask] = 0
        layer.down_stair_hits[mask] = 0

    def extract(self, layer) -> List[StairDetection]:
        from scipy import ndimage

        out = []
        for kind, hits in (("up", layer.up_stair_hits), ("down", layer.down_stair_hits)):
            if hits is None:
                continue
            mask = hits >= self.min_hits
            if layer.disabled_stair is not None:
                mask &= ~layer.disabled_stair
            labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
            free = layer.costmap.grid == FREE
            if not free.any():
                continue
            _, (fr, fc) = ndimage.distance_transform_edt(~free, return_indices=True)
            for label in range(1, count + 1):
                cells = np.argwhere(labels == label)
                if len(cells) < self.min_cells:
                    continue
                r, c = np.rint(cells.mean(axis=0)).astype(int)
                r, c = int(fr[r, c]), int(fc[r, c])
                out.append(StairDetection(
                    kind, layer.costmap.grid_to_world(np.array([r, c], float)),
                    cells, len(cells), int(hits[cells[:, 0], cells[:, 1]].sum()),
                ))
        return out
