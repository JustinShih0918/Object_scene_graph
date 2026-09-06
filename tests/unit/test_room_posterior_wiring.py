"""The room posterior inside the strategy: the trigger, and the guards that are
supposed to keep in_anchor at 0.822 while cross_anchor moves.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from osg.core.config import OSGConfig
from osg.exploration.strategy import ExplorationStrategy
from osg.graph.scene_graph import ContainerNode, RoomNode


class StubPrior:
    """Records what it was asked; answers whatever it was told to."""

    def __init__(self, answer=None):
        self.answer = answer
        self.asks = []

    def request(self, target, rooms, anchor=None, n_disbelieved=0, block_s=0.0):
        self.asks.append({"target": target, "rooms": rooms, "anchor": anchor,
                          "n_disbelieved": n_disbelieved, "block_s": block_s})
        return True

    def ranking(self):
        return list(self.answer) if self.answer else None

    def reset(self):
        pass

    def shutdown(self):
        pass


def _strategy(room_prior=None, **over):
    cfg = OSGConfig()
    cfg.exploration.search_posterior = True
    for k, v in over.items():
        setattr(cfg.exploration, k, v)
    return ExplorationStrategy(cfg, planner=None, scorer=None, viewpoint_planner=None,
                               affinity=None, stats={}, profiler=types.SimpleNamespace(),
                               room_prior=room_prior)


def _sg():
    """Two rooms: 1 holds bedroom furniture, 5 holds the kitchen."""
    sg = types.SimpleNamespace()
    sg.rooms = {
        1: RoomNode(id=1, label="bedroom", centroid_xy=np.array([0.0, 0.0])),
        5: RoomNode(id=5, label="kitchen", centroid_xy=np.array([8.0, 0.0])),
    }
    nodes = {
        10: ContainerNode(id=10, label="bed", track_ids=[10],
                          center=np.array([0.0, 0.5, 0.0]), top_h=0.5, area_m2=2.0),
        11: ContainerNode(id=11, label="desk", track_ids=[11],
                          center=np.array([1.0, 0.7, 0.0]), top_h=0.7, area_m2=1.0),
        50: ContainerNode(id=50, label="refrigerator", track_ids=[50],
                          center=np.array([8.0, 1.0, 0.0]), top_h=1.0, area_m2=1.0),
    }
    nodes[10].room_id, nodes[11].room_id, nodes[50].room_id = 1, 1, 5
    sg.containers = nodes
    sg.containers_in_room = lambda rid: [n for n in nodes.values() if n.room_id == rid]
    sg.room_of_point = lambda xy: sg.rooms[1] if float(np.asarray(xy)[0]) < 4.0 else sg.rooms[5]
    return sg


def _world(sg, target="tin can", tracks=()):
    layer = types.SimpleNamespace(tracks=lambda include_blacklisted=False: list(tracks))
    return types.SimpleNamespace(scene_graph=sg, object_layer=layer, target=target,
                                 agent_xy=np.zeros(2), step=100)


def _track(label, xy, p, n_expected, n_missed=0):
    """A real ObjectTrack, not a namespace: a track's pose is on its ELLIPSOID,
    and a stub with a bare `.center` is what let an AttributeError reach a live
    run (only ContainerNode has that field)."""
    from osg.objects.association import ObjectTrack
    from osg.objects.ellipsoid import Ellipsoid

    t = ObjectTrack(id=abs(hash((label, xy))) % 10000, label=label,
                    ellipsoid=Ellipsoid(center=np.array([xy[0], 0.5, xy[1]]),
                                        axes=np.array([0.1, 0.1, 0.1]), R=np.eye(3)))
    t.presence.log_odds = math.log(p / (1.0 - p))
    t.presence.n_expected = n_expected
    t.presence.n_missed = n_missed
    return t


# ------------------------------------------------------------------- trigger


def test_no_ask_before_the_room_has_actually_failed():
    """The trigger is fruitless ARRIVALS. Asking on entry would fire on the
    in_anchor half, where the room the agent is in is usually the right one."""
    p = StubPrior()
    s = _strategy(p, room_posterior_llm=True, room_posterior_after=2)
    s._maybe_ask_rooms(_world(_sg()), here_id=1)
    assert p.asks == []
    s._room_fruitless[1] = 1
    s._maybe_ask_rooms(_world(_sg()), here_id=1)
    assert p.asks == []


def test_asks_once_the_room_is_refused_and_only_once():
    p = StubPrior()
    s = _strategy(p, room_posterior_llm=True, room_posterior_after=2)
    s._room_fruitless[1] = 2
    for _ in range(5):
        s._maybe_ask_rooms(_world(_sg()), here_id=1)
    assert len(p.asks) == 1, "one refused room is one question, not one per round"
    assert s.stats["room_prior_requests"] == 1


def test_a_second_room_asks_again():
    p = StubPrior()
    s = _strategy(p, room_posterior_llm=True, room_posterior_after=2)
    s._room_fruitless[1] = 2
    s._maybe_ask_rooms(_world(_sg()), here_id=1)
    s._room_fruitless[5] = 2
    s._maybe_ask_rooms(_world(_sg()), here_id=5)
    assert len(p.asks) == 2


# ------------------------------------------------------------------- payload


def test_payload_reports_rooms_surfaces_and_search_effort():
    p = StubPrior()
    s = _strategy(p, room_posterior_llm=True, room_posterior_after=1)
    s.search_log.survived[10] = 0.2  # the bed has been looked at
    s._room_fruitless[1] = 1
    s._maybe_ask_rooms(_world(_sg()), here_id=1)
    rooms = {r["id"]: r for r in p.asks[0]["rooms"]}
    assert rooms[1]["surfaces"] == ["bed", "desk"] and rooms[1]["n_looked"] == 1
    assert rooms[5]["surfaces"] == ["refrigerator"] and rooms[5]["n_looked"] == 0
    assert rooms[1]["label"] == "bedroom"


def test_payload_carries_the_anchor_and_the_disbelief_count():
    p = StubPrior()
    s = _strategy(p, room_posterior_llm=True, room_posterior_after=1)
    tracks = [
        _track("tin can", (0.5, 0.0), p=0.07, n_expected=23, n_missed=20),
        _track("mug", (1.0, 0.0), p=0.05, n_expected=9),      # disbelieved
        _track("chair", (1.0, 0.0), p=0.9, n_expected=9),     # still believed
        _track("lamp", (1.0, 0.0), p=0.2, n_expected=0),      # never expected
    ]
    s._room_fruitless[1] = 1
    s._maybe_ask_rooms(_world(_sg(), tracks=tracks), here_id=1)
    ask = p.asks[0]
    assert ask["anchor"]["room"] == 1
    assert ask["anchor"]["n_expected"] == 23 and ask["anchor"]["n_missed"] == 20
    assert ask["anchor"]["p"] == pytest.approx(0.07, abs=1e-6)
    # the target itself is disbelieved too, so 2: the can and the mug
    assert ask["n_disbelieved"] == 2, "unobserved tracks are not evidence of absence"


# -------------------------------------------------------------------- guards


def _priors(s, world):
    """Apply whatever room term the strategy would apply, to one candidate per
    room, and return {room_id: prior}."""
    cands = [types.SimpleNamespace(ref_id=11, prior=1.0),
             types.SimpleNamespace(ref_id=50, prior=1.0)]
    sg = world.scene_graph
    mult = s._room_multipliers(world)
    room_bonus = float(s.cfg.search_same_room_bonus)
    here = sg.room_of_point(world.agent_xy)
    use_bonus = room_bonus > 1.0 and here is not None and (
        not mult or bool(s.cfg.room_posterior_keep_bonus))
    for c in cands:
        node = sg.containers[c.ref_id]
        if mult:
            c.prior *= mult.get(int(node.room_id), 1.0)
        if use_bonus and node.room_id == here.id:
            c.prior *= s._room_bonus(room_bonus, node.room_id)
    return {sg.containers[c.ref_id].room_id: c.prior for c in cands}


def test_flag_off_reproduces_the_positional_bonus_exactly():
    s = _strategy(None)
    got = _priors(s, _world(_sg()))
    assert got == {1: 4.0, 5: 1.0}


def test_no_answer_yet_reproduces_the_positional_bonus_exactly():
    """A query in flight must cost latency and nothing else."""
    s = _strategy(StubPrior(answer=None), room_posterior_llm=True)
    assert _priors(s, _world(_sg())) == {1: 4.0, 5: 1.0}


def test_model_agreeing_with_the_agent_reproduces_the_bonus():
    """The top room gets exactly x4, so an episode where the agent is already
    in the right room computes the shipped posterior -- which is what protects
    00829, where the room bonus is load-bearing."""
    s = _strategy(StubPrior(answer=[1, 5]), room_posterior_llm=True)
    assert _priors(s, _world(_sg())) == {1: 4.0, 5: 0.25}


def test_model_disagreeing_flips_the_ranking():
    """The cross_anchor case: the object left the room the agent is standing in
    and the kitchen must outrank the bedroom."""
    s = _strategy(StubPrior(answer=[5, 1]), room_posterior_llm=True)
    got = _priors(s, _world(_sg()))
    assert got[5] == pytest.approx(4.0) and got[1] == pytest.approx(0.25)
    assert got[5] > got[1] * 15


def test_keep_bonus_stacks_both_terms():
    s = _strategy(StubPrior(answer=[5, 1]), room_posterior_llm=True,
                  room_posterior_keep_bonus=True)
    got = _priors(s, _world(_sg()))
    assert got[1] == pytest.approx(0.25 * 4.0)
    assert got[5] == pytest.approx(4.0)


# ------------------------------------------------------------- frontier arm


def test_frontier_scores_are_inert_until_an_answer_lands():
    s = _strategy(StubPrior(answer=None), room_posterior_llm=True)
    s.scorer = types.SimpleNamespace(latest=lambda: {})
    f = types.SimpleNamespace(id=7, centroid_xy=np.array([8.0, 0.0]))
    assert s._frontier_scores(_world(_sg()), [f]) == {}


def test_frontier_into_the_favoured_room_outranks_the_other():
    s = _strategy(StubPrior(answer=[5, 1]), room_posterior_llm=True)
    s.scorer = types.SimpleNamespace(latest=lambda: {})
    world = _world(_sg())
    s._room_multipliers(world)  # land the answer
    near = types.SimpleNamespace(id=1, centroid_xy=np.array([0.0, 0.0]))
    far = types.SimpleNamespace(id=2, centroid_xy=np.array([8.0, 0.0]))
    scores = s._frontier_scores(world, [near, far])
    assert scores[2] > scores[1]
    # the best room leaves unscored_prior untouched: this must not re-price the
    # frontier arm against the surface arm (that is condition R, a null).
    assert scores[2] == pytest.approx(float(s.cfg.unscored_prior))


def test_the_block_budget_reaches_the_provider():
    """Pure asynchrony delivered nothing: 36 episodes, 10 queries, 8 answers,
    0 applied, X identical to Y. The blocking budget is how the treatment is
    actually delivered on a cold cache, so it must not be silently dropped."""
    p = StubPrior()
    s = _strategy(p, room_posterior_llm=True, room_posterior_after=1,
                  room_posterior_block_s=90.0)
    s._room_fruitless[1] = 1
    s._maybe_ask_rooms(_world(_sg()), here_id=1)
    assert p.asks[0]["block_s"] == 90.0
