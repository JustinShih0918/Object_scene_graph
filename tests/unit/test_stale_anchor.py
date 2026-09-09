"""The stale-anchor policy (core/config/verification.py): a prior-map track
that this episode has never seen gets one STOP instead of an absence verdict,
and once a prior-map track of the label is refuted, its unseen twins from the
same static pass stop being candidates."""
from __future__ import annotations

import numpy as np

from osg.agent.nav_agent import State
from osg.objects.association import Observation, ObjectTrack
from osg.objects.ellipsoid import Ellipsoid

from .test_nav_agent import _frame, make_agent, make_cfg


def _obs(frame_id: int) -> Observation:
    return Observation(frame_id=frame_id, mu=np.zeros(2), cov=np.eye(2), K=np.eye(3),
                       T_cw=np.eye(4), mean_depth=1.0)


def _track(tid, centre, *, prior: bool, live: bool, label="chair", score=0.9):
    t = ObjectTrack(
        id=tid, label=label,
        ellipsoid=Ellipsoid(center=np.asarray(centre, dtype=float),
                            axes=np.array([0.2, 0.2, 0.2]), R=np.eye(3)),
    )
    t.from_prior = prior
    t.observations = [_obs(-1_000_000 + 5)] if prior else []
    if live:
        t.observations.append(_obs(12))
    t.best_score = score
    return t


def test_seen_live_reads_the_frame_ids():
    assert _track(1, [1, 0, 0], prior=True, live=False).seen_live is False
    assert _track(2, [1, 0, 0], prior=True, live=True).seen_live is True
    assert _track(3, [1, 0, 0], prior=False, live=True).seen_live is True


def _silent_arrival(track, **over):
    cfg = make_cfg(approach_to_viewpoint=False, approach_navigable_goal=False)
    for k, v in over.items():
        setattr(cfg.verification, k, v)
    agent = make_agent(cfg, target="chair")
    agent.object_layer._tracks[track.id] = track
    agent._candidate_id = track.id
    agent.approach.start(np.array([3.0, 0.0]), agent_xy=np.zeros(2))
    agent.approach.steps_left = 0
    agent._goto_deadline = 10_000
    return agent


def test_off_by_default():
    cfg = make_cfg()
    assert cfg.verification.stop_at_stale_anchor_once is False
    assert cfg.verification.retire_stale_twins_after_absence is False


def test_first_silent_arrival_at_a_stale_anchor_stops_once():
    agent = _silent_arrival(_track(1, [3.0, 0.5, 0.0], prior=True, live=False),
                            stop_at_stale_anchor_once=True)
    action = agent.approach.step(_frame([0.0, 0.0]))
    assert action == "stop" and agent.state is State.DONE
    assert agent.stats["stale_anchor_stop"] == 1
    # The second silent arrival goes to the absence sensor as before.
    agent.rearm(100)
    agent._candidate_id = 1
    agent.approach.start(np.array([3.0, 0.0]), agent_xy=np.zeros(2))
    agent.approach.steps_left = 0
    agent.approach.step(_frame([0.0, 0.0]))
    assert agent.stats["stale_anchor_stop"] == 1


def test_a_track_seen_live_gets_no_free_stop():
    agent = _silent_arrival(_track(1, [3.0, 0.5, 0.0], prior=True, live=True),
                            stop_at_stale_anchor_once=True)
    agent.approach.step(_frame([0.0, 0.0]))
    assert "stale_anchor_stop" not in agent.stats


def test_a_live_track_gets_no_free_stop():
    agent = _silent_arrival(_track(1, [3.0, 0.5, 0.0], prior=False, live=True),
                            stop_at_stale_anchor_once=True)
    agent.approach.step(_frame([0.0, 0.0]))
    assert "stale_anchor_stop" not in agent.stats


def _agent_with_twins(refuted: bool, **over):
    cfg = make_cfg()
    cfg.verification.min_obs = 0
    for k, v in over.items():
        setattr(cfg.verification, k, v)
    agent = make_agent(cfg, target="chair")
    anchor = _track(1, [3.0, 0.5, 0.0], prior=True, live=False)
    if refuted:
        anchor.absence_arrivals = 1
        anchor.presence.log_odds = -3.0  # under the bar, as after a refutation
    twin = _track(2, [8.0, 0.5, 2.0], prior=True, live=False, score=0.8)
    live = _track(3, [5.0, 0.5, -2.0], prior=False, live=True, score=0.7)
    for t in (anchor, twin, live):
        agent.object_layer._tracks[t.id] = t
    return agent


def test_unseen_prior_twins_are_retired_once_the_anchor_is_refuted():
    agent = _agent_with_twins(refuted=True, retire_stale_twins_after_absence=True)
    cands = agent.object_layer.candidates("chair", min_obs=0, min_score=0.0, min_bbox_px=0,
                                          min_evidence=0.0, min_presence=0.0)
    kept = agent.candidates._without_stale_twins(cands)
    assert [t.id for t in kept] == [3]
    assert agent.stats["stale_twins_retired"] >= 1


def test_nothing_is_retired_before_a_refutation_or_with_the_flag_off():
    agent = _agent_with_twins(refuted=False, retire_stale_twins_after_absence=True)
    cands = agent.object_layer.candidates("chair", min_obs=0, min_score=0.0, min_bbox_px=0,
                                          min_evidence=0.0, min_presence=0.0)
    assert len(agent.candidates._without_stale_twins(cands)) == len(cands)
    agent = _agent_with_twins(refuted=True)
    agent.candidates.check(np.zeros(2))
    assert "stale_twins_retired" not in agent.stats
