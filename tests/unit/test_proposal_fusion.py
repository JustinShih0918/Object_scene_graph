"""Proposals are fused as appearance hypotheses, not as namings.

Measured on the full 107 (docs/REGION_FALSE_ADMISSION.md): a region the stage
admitted became an ordinary same-label track, won the candidate gate and spent
an attempt -- 18 trials lost, because the per-frame threshold admits on 13% of
frames with no object in view. The fix is not when the stage runs but how its
output is judged. This file pins that judgement.
"""
from __future__ import annotations

import numpy as np

from osg.core.types import CameraIntrinsics, Detection, FrameData
from osg.objects.object_layer import ObjectLayer

K = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)


def _det(label, box, score, source="detector", ft=None):
    x1, y1, x2, y2 = box
    mask = np.zeros((480, 640), dtype=bool)
    mask[int(y1):int(y2), int(x1):int(x2)] = True
    return Detection(label=label, score=score,
                     bbox_xyxy=np.array([float(x1), float(y1), float(x2), float(y2)]),
                     mask=mask, source=source, clip_ft=ft)


def _frame(frame_id, depth=2.0, x=0.0):
    T = np.eye(4); T[0, 3] = x
    return FrameData(frame_id=frame_id, rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                     depth=np.full((480, 640), depth, dtype=np.float32),
                     T_wc=T, intrinsics=K)


def _unit(*v):
    a = np.asarray(v, dtype=np.float32); return a / np.linalg.norm(a)


BOX = (280, 200, 360, 280)


def _layer():
    return ObjectLayer(min_det_score=0.0, min_det_bbox_px=0.0, confirm_baseline_m=0.0)


# ------------------------------------------------------------ the object layer

def test_a_proposal_counts_apart_from_a_naming_and_keeps_a_mean_feature():
    layer = _layer()
    layer.set_proposal_text(_unit(1, 0, 0))
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    layer.update(_frame(2), [_det("bowl", BOX, 0.5, "proposal", _unit(0, 1, 0))])
    (t,) = layer.tracks()
    assert t.n_obs == 2 and t.n_proposal_obs == 2 and t.proposal_only
    # mean of (1,0,0) and (0,1,0), renormalised: cosine to (1,0,0) is 1/sqrt 2
    assert abs(t.proposal_sim - 0.7071) < 1e-3
    assert layer.funnel["proposal_obs"] == 2


def test_one_naming_by_the_detector_ends_proposal_only():
    layer = _layer()
    layer.set_proposal_text(_unit(1, 0, 0))
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    layer.update(_frame(2), [_det("bowl", BOX, 0.9)])
    (t,) = layer.tracks()
    assert t.n_obs == 2 and t.n_proposal_obs == 1 and not t.proposal_only


def test_without_a_query_text_the_cosine_stays_unset():
    layer = _layer()
    layer.update(_frame(1), [_det("bowl", BOX, 0.5, "proposal", _unit(1, 0, 0))])
    (t,) = layer.tracks()
    assert t.n_proposal_obs == 1 and t.proposal_sim == -1.0


def test_a_detector_only_map_is_untouched():
    layer = _layer()
    layer.update(_frame(1), [_det("bowl", BOX, 0.9)])
    layer.update(_frame(2), [_det("bowl", BOX, 0.9)])
    (t,) = layer.tracks()
    assert t.n_proposal_obs == 0 and not t.proposal_only and t.clip_ft is None
    assert layer.funnel["proposal_obs"] == 0


# ------------------------------------------------------------ the candidate gate

def _proposal_track(layer, n, sim_ft, text=_unit(1, 0, 0)):
    layer.set_proposal_text(text)
    for i in range(n):
        layer.update(_frame(i + 1), [_det("bowl", BOX, 0.5, "proposal", sim_ft)])


def test_a_proposal_only_track_needs_both_observations_and_the_cosine():
    layer = _layer()
    _proposal_track(layer, 3, _unit(1, 0.5, 0))          # cosine 0.894
    assert len(layer.candidates("bowl", min_obs=1)) == 1  # defaults: no bar
    assert layer.candidates("bowl", min_obs=1, proposal_min_obs=4) == []
    assert layer.candidates("bowl", min_obs=1, proposal_tau=0.95) == []
    assert len(layer.candidates("bowl", min_obs=1, proposal_min_obs=3, proposal_tau=0.85)) == 1


def test_commits_off_keeps_every_proposal_only_track_out():
    layer = _layer()
    _proposal_track(layer, 5, _unit(1, 0, 0))            # cosine 1.0
    assert layer.candidates("bowl", min_obs=1, proposal_commits=False) == []
    # ...and does not touch a named one
    layer.update(_frame(9), [_det("bowl", (10, 10, 90, 90), 0.9)])
    layer.update(_frame(10), [_det("bowl", (10, 10, 90, 90), 0.9)])
    out = layer.candidates("bowl", min_obs=1, proposal_commits=False)
    assert [t.proposal_only for t in out] == [False]


def test_a_named_track_always_ranks_ahead_of_a_proposal_only_one():
    layer = _layer()
    _proposal_track(layer, 6, _unit(1, 0, 0))            # strong: 6 obs, cosine 1.0
    layer.update(_frame(20), [_det("bowl", (10, 10, 90, 90), 0.4)])
    layer.update(_frame(21), [_det("bowl", (10, 10, 90, 90), 0.4)])
    for rank_by_presence in (False, True):
        out = layer.candidates("bowl", min_obs=1, rank_by_presence=rank_by_presence)
        assert [t.proposal_only for t in out] == [False, True]


def test_a_label_the_detector_never_used_is_still_not_a_candidate():
    layer = _layer()
    _proposal_track(layer, 3, _unit(1, 0, 0))
    assert layer.candidates("mug", min_obs=1) == []


# ------------------------------------------------------------ the stage's output

def test_the_proposal_detection_carries_its_feature_and_provenance():
    from osg.perception.region_proposer import RegionProposer

    class Cfg:
        imgsz = 64; conf = 0.1; iou = 0.5; tau = 0.0; admit_score = 0.5
        pad_frac = 0.25; min_area_px = 10.0; max_area_frac = 0.5; max_regions = 8
        device = "cpu"

    class T:
        def __init__(self, a): self.a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a

    class Boxes:
        def __init__(self, a): self.xyxy = T(a)
        def __len__(self): return len(self.xyxy.a)

    class Masks:
        def __init__(self, a): self.data = T(a)

    class Result:
        def __init__(self, b, m): self.boxes = Boxes(b); self.masks = Masks(m)

    class Model:
        def predict(self, rgb, **kw):
            b = np.array([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=float)
            m = np.zeros((2, 100, 100), dtype=bool); m[0, 10:30, 10:30] = True; m[1, 40:60, 40:60] = True
            return [Result(b, m)]

    class Enc:
        def text_feature(self, s): return _unit(1, 0, 0)
        def encode_images(self, crops): return np.stack([_unit(0, 1, 0), _unit(1, 0, 0)])

    rp = RegionProposer(Model(), Enc(), Cfg())
    rp.set_target("bowl")
    det = rp.propose(np.zeros((100, 100, 3), dtype=np.uint8))
    assert det is not None and det.source == "proposal"
    assert np.allclose(det.clip_ft, _unit(1, 0, 0))
    assert rp.last_score == 1.0
    assert det.bbox_xyxy[0] == 40.0     # the second region won


def test_every_keyframe_bypasses_the_episode_gates_but_not_the_cap():
    from osg.agent.nav_agent import NavAgent

    class Cfg:
        every_keyframe = True; require_never_named = True; unnamed_keyframes = 20
        max_per_episode = 2

    class RP:
        cfg = Cfg()

    a = NavAgent.__new__(NavAgent)
    a.region_proposer = RP(); a._region_admits = 0; a._region_kf = 0; a._region_named = True
    assert a._region_active()
    a._region_admits = 2
    assert not a._region_active()
