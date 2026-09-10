"""The appearance channel's arithmetic, and its promise to stay out of the way.

The last of these is the one that matters most: every scored result on the
DualMap benchmark was produced with this code absent, and runs there are exactly
reproducible, so a candidate list or a surface order that shifts by a hair when
the feature term is off would silently invalidate the comparison it exists to
serve.
"""
from __future__ import annotations

import numpy as np

from osg.exploration.search_belief import InspectionLog, build_container_candidates
from osg.objects.feature_memory import (
    admits,
    blend_label_text,
    container_children,
    container_feature_scores,
    cosine,
    feature_term,
    merge_running_mean,
    refuted,
)


def unit(*values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Track:
    def __init__(self, tid, label="thing", ft=None, **kw):
        self.id = tid
        self.label = label
        self.clip_ft = ft
        self.feature_sim = -1.0
        self.feature_admitted = False
        self.from_prior = kw.get("from_prior", False)
        self._seen_live = kw.get("seen_live", False)
        self.absence_arrivals = kw.get("absence_arrivals", 0)
        self.failed_attempts = kw.get("failed_attempts", 0)
        self.blacklisted = kw.get("blacklisted", False)
        self.disabled = kw.get("disabled", False)

    @property
    def seen_live(self):
        return self._seen_live


class _Layer:
    def __init__(self, tracks):
        self._by_id = {t.id: t for t in tracks}

    def get(self, tid):
        return self._by_id.get(int(tid))


class _Node:
    def __init__(self, cid, label, centre, top_h=0.75, area=1.0, track_ids=()):
        self.id = cid
        self.label = label
        self.center = np.asarray(centre, dtype=float)
        self.top_h = top_h
        self.area_m2 = area
        self.room_id = 0
        self.floor = 0
        self.floor_id = 0
        self.track_ids = list(track_ids)
        self.object_ids = []


class _Obj:
    def __init__(self, tid, centre, container_id=None):
        self.track_id = tid
        self.center = np.asarray(centre, dtype=float)
        self.container_id = container_id
        self.floor_id = 0


class _Graph:
    def __init__(self, containers, objects):
        self.containers = {c.id: c for c in containers}
        self.objects = list(objects)
        self.floors = {}
        self.rooms = {}


# ------------------------------------------------------------------ the term

def test_the_best_looking_surface_is_unweighted_and_the_rest_decay():
    assert feature_term(0.30, 0.30, beta=20.0, floor=0.25) == 1.0
    near = feature_term(0.29, 0.30, beta=20.0, floor=0.25)
    far = feature_term(0.20, 0.30, beta=20.0, floor=0.25)
    assert 0.25 < far < near < 1.0


def test_a_surface_with_no_feature_is_ordinary_not_impossible():
    """`floor`, never zero: half the objects in a house were never mapped, so
    silence about a surface is not evidence against it."""
    assert feature_term(None, 0.30, beta=20.0, floor=0.25) == 0.25
    assert feature_term(None, 0.30, beta=20.0, floor=0.0) == 0.0


def test_beta_zero_is_the_off_switch():
    assert feature_term(0.10, 0.30, beta=0.0, floor=0.25) == 1.0


def test_cosine_is_rounded_and_missing_features_score_minus_one():
    a, b = unit(1, 0, 0), unit(1, 0, 0)
    assert cosine(a, b) == 1.0
    assert cosine(None, b) == -1.0
    assert cosine(a, None) == -1.0
    assert cosine(a, unit(0, 1, 0)) == 0.0
    # Rounded, so a kernel that differs in the seventh decimal cannot flip an
    # admission and break a reproducible run.
    assert cosine(np.float32([1, 1e-9, 0]), unit(1, 0, 0)) == 1.0


# ------------------------------------------------------------------- merging

def test_merged_track_features_stay_unit_and_average():
    ft, n = merge_running_mean(None, 0, unit(1, 0, 0))
    assert n == 1 and np.isclose(np.linalg.norm(ft), 1.0)
    ft, n = merge_running_mean(ft, n, unit(0, 1, 0))
    assert n == 2 and np.isclose(np.linalg.norm(ft), 1.0)
    assert np.allclose(ft, unit(1, 1, 0), atol=1e-6)
    same, n_same = merge_running_mean(ft, n, None)
    assert n_same == 2 and np.allclose(same, ft)


def test_label_text_blending_is_off_at_weight_zero():
    image = unit(1, 0, 0)
    assert np.allclose(blend_label_text(image, unit(0, 1, 0), 0.0), image)
    mixed = blend_label_text(image, unit(0, 1, 0), 0.3)
    assert np.isclose(np.linalg.norm(mixed), 1.0) and mixed[1] > 0


# ---------------------------------------------------------------- admission

def test_admission_needs_the_threshold_and_holds_back_unseen_prior_tracks():
    live = _Track(1, ft=unit(1, 0, 0))
    live.feature_sim = 0.31
    assert admits(live, 0.30) is True
    live.feature_sim = 0.29
    assert admits(live, 0.30) is False

    ghost = _Track(2, from_prior=True, seen_live=False)
    ghost.feature_sim = 0.40
    assert admits(ghost, 0.30) is False
    assert admits(ghost, 0.30, admit_prior_tracks=True) is True

    seen = _Track(3, from_prior=True, seen_live=True)
    seen.feature_sim = 0.40
    assert admits(seen, 0.30) is True


def test_refutation_is_an_arrival_or_a_failed_attempt_not_a_low_belief():
    assert refuted(_Track(1)) is False
    assert refuted(_Track(2, absence_arrivals=1)) is True
    assert refuted(_Track(3, failed_attempts=1)) is True
    assert refuted(_Track(4, blacklisted=True)) is True
    assert refuted(None) is True


# ---------------------------------------------------------------- containers

def _scene():
    desk = _Node(10, "desk", (0.0, 0.0, 0.0), track_ids=[10])
    bed = _Node(20, "bed", (5.0, 0.0, 0.0), track_ids=[20])
    objects = [
        _Obj(1, (0.1, 0.8, 0.0), container_id=10),   # resting on the desk
        _Obj(2, (5.4, 0.8, 0.0), container_id=None),  # beside the bed, unbound
    ]
    return _Graph([desk, bed], objects)


def test_a_surface_inherits_the_appearance_of_what_it_holds():
    text = unit(1, 0, 0)
    tracks = [
        _Track(1, ft=unit(1, 0, 0)), _Track(2, ft=unit(0, 1, 0)),
        _Track(10, ft=unit(0, 0, 1)), _Track(20, ft=unit(0, 0, 1)),
    ]
    scores = container_feature_scores(_scene(), _Layer(tracks), text)
    assert scores[10] == 1.0          # the desk, through its child
    assert scores[20] == 0.0          # the bed, on its own feature only


def test_the_child_set_widens_by_proximity_because_support_binds_few_objects():
    graph = _scene()
    bound = container_children(graph, radius_m=0.0)
    assert 2 not in bound[20]
    widened = container_children(graph, radius_m=1.0)
    assert 2 in widened[20]

    text = unit(0, 1, 0)
    tracks = [_Track(1, ft=unit(1, 0, 0)), _Track(2, ft=unit(0, 1, 0)),
              _Track(10, ft=unit(0, 0, 1)), _Track(20, ft=unit(0, 0, 1))]
    assert container_feature_scores(graph, _Layer(tracks), text)[20] == 0.0
    widened_scores = container_feature_scores(graph, _Layer(tracks), text, radius_m=1.0)
    assert widened_scores[20] == 1.0


def test_a_refuted_child_stops_recommending_its_surface():
    text = unit(1, 0, 0)
    tracks = [_Track(1, ft=unit(1, 0, 0), absence_arrivals=1), _Track(2, ft=unit(0, 1, 0)),
              _Track(10, ft=unit(0, 0, 1)), _Track(20, ft=unit(0, 0, 1))]
    scores = container_feature_scores(_scene(), _Layer(tracks), text)
    assert scores[10] == 0.0
    kept = container_feature_scores(_scene(), _Layer(tracks), text, skip_refuted=False)
    assert kept[10] == 1.0


def test_no_text_feature_means_no_scores_at_all():
    assert container_feature_scores(_scene(), _Layer([]), None) == {}


# ------------------------------------------------- the promise to stay away

def test_the_candidate_builder_is_untouched_when_no_features_are_supplied():
    graph = _scene()
    plain = build_container_candidates(graph, "bowl", InspectionLog())
    again = build_container_candidates(
        graph, "bowl", InspectionLog(), feature_scores=None, feature_beta=9.0,
        feature_floor=0.0,
    )
    assert [(c.ref_id, c.prior, c.goal_xy.tolist()) for c in plain] == \
           [(c.ref_id, c.prior, c.goal_xy.tolist()) for c in again]


def test_the_feature_term_can_reorder_surfaces_without_changing_the_scale():
    graph = _scene()
    plain = {c.ref_id: c.prior for c in build_container_candidates(graph, "bowl", InspectionLog())}
    fused = build_container_candidates(
        graph, "bowl", InspectionLog(), feature_scores={20: 0.31, 10: 0.20},
        feature_beta=20.0, feature_floor=0.1,
    )
    priors = {c.ref_id: c.prior for c in fused}
    assert priors[20] > priors[10]
    # The best candidate still carries the full surface mass, so the
    # search-versus-explore comparison sees the same magnitude it always did.
    assert max(priors.values()) == max(plain.values())
