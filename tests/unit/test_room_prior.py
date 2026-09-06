"""The room posterior: what it asks, what it does with the answer, and the two
ways it must fail safely.
"""
from __future__ import annotations

import json

import pytest

from osg.llm.room_prior import RoomPriorProvider, rank_multipliers


ROOMS = [
    {"id": 1, "label": "bedroom", "surfaces": ["bed", "desk", "shelf"],
     "n_surfaces": 4, "n_looked": 4},
    {"id": 3, "label": "office", "surfaces": ["cabinet", "desk"],
     "n_surfaces": 2, "n_looked": 0},
    {"id": 5, "label": "kitchen", "surfaces": ["cabinet", "refrigerator"],
     "n_surfaces": 3, "n_looked": 0},
]
ANCHOR = {"room": 1, "label": "desk", "p": 0.07, "n_expected": 23, "n_missed": 20}


class FakeClient:
    def __init__(self, reply, record=None):
        self.reply = reply
        self.calls = 0
        self.record = record if record is not None else []

    def chat(self, system, user, **kw):
        self.calls += 1
        self.record.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _wait(p):
    """The provider answers on a worker thread; join it rather than sleeping."""
    ex = p._executor
    if ex is not None:
        ex.shutdown(wait=True)
        p._executor = None


# ----------------------------------------------------------------- multipliers


def test_multiplier_is_geometric_and_symmetric():
    m = rank_multipliers([5, 3, 1], [1, 3, 5], spread=4.0)
    assert m[5] == pytest.approx(4.0)
    assert m[3] == pytest.approx(1.0)
    assert m[1] == pytest.approx(0.25)


def test_top_room_gets_exactly_the_same_room_bonus():
    """The multiplier REPLACES `search_same_room_bonus`, so at the default
    spread the room the model puts first must be worth what the positional
    bonus asserts today -- otherwise swapping the term silently re-scales the
    whole surface arm."""
    assert rank_multipliers([2], [2, 7], spread=4.0)[2] == pytest.approx(4.0)


def test_rooms_the_model_dropped_rank_last():
    m = rank_multipliers([5], [1, 3, 5], spread=4.0)
    assert m[5] > m[3] and m[5] > m[1]
    assert m[5] == pytest.approx(4.0)


def test_floor_makes_the_term_promote_only():
    """Symmetric was measured harmful where the positional prior it replaces is
    right: on 00829 the control took 5/9 of the episodes the posterior reached
    and the symmetric version 2/9, because a correct room fell from x4 to
    x0.25. Floored at 1.0 the model can lift a room and never bury one."""
    m = rank_multipliers([5, 3, 1], [1, 3, 5], spread=4.0, floor=1.0)
    assert m[5] == pytest.approx(4.0)
    assert m[3] == pytest.approx(1.0)
    assert m[1] == pytest.approx(1.0), "a room the model dislikes must not go below neutral"
    assert min(m.values()) >= 1.0


def test_floor_defaults_off_so_earlier_conditions_reproduce():
    m = rank_multipliers([5, 3, 1], [1, 3, 5], spread=4.0)
    assert m[1] == pytest.approx(0.25)


def test_unit_spread_is_a_no_op():
    m = rank_multipliers([5, 3, 1], [1, 3, 5], spread=1.0)
    assert set(m.values()) == {1.0}


# --------------------------------------------------------------------- prompt


def test_prompt_names_every_room_and_what_it_holds():
    p = RoomPriorProvider(client=None)
    text = p._build("tin can", ROOMS, ANCHOR, n_disbelieved=47)
    assert "room 1 (bedroom)" in text and "room 5 (kitchen)" in text
    assert "cabinet, refrigerator" in text
    assert "tin can" in text


@pytest.mark.parametrize("p_val,n_exp,bucket,phrase", [
    (0.07, 23, "gone", "many times"),
    (0.30, 2, "probably gone", "a few times"),
    (0.82, 0, "unchecked", "has not been"),
    (0.90, 12, "still there", "still seems"),
])
def test_presence_enters_as_a_bucket_that_keeps_the_count_distinction(
        p_val, n_exp, bucket, phrase):
    """p=0.07-over-23 and p=0.82-over-0 must not collapse to the same sentence.
    A bare `p` threshold is what fired on 56% of in_anchor episodes against 30%
    predicted -- p decays from ordinary missed expectations, and `n_expected` is
    what separates that drift from real absence."""
    prov = RoomPriorProvider(client=None)
    anchor = {"room": 1, "label": "desk", "p": p_val, "n_expected": n_exp,
              "n_missed": max(0, n_exp - 3)}
    assert prov._anchor_bucket(anchor) == bucket
    assert phrase in prov._build("tin can", ROOMS, anchor, 0)


def test_prompt_and_cache_key_carry_the_same_facts():
    """A prompt richer than its key means two questions share one entry and the
    answer a run gets depends on which fired first -- nondeterminism inside an
    A/B. Anything that changes the prompt must change the key."""
    prov = RoomPriorProvider(client=None)
    gone = {"room": 1, "label": "desk", "p": 0.07, "n_expected": 23, "n_missed": 20}
    unchecked = {"room": 1, "label": "desk", "p": 0.82, "n_expected": 0, "n_missed": 0}
    assert prov._build("tin can", ROOMS, gone, 0) != prov._build("tin can", ROOMS, unchecked, 0)
    assert prov._key("tin can", ROOMS, gone) != prov._key("tin can", ROOMS, unchecked)
    # and the reverse: same key => same prompt
    same = dict(gone, n_missed=21)   # n_missed is in neither
    assert prov._key("tin can", ROOMS, gone) == prov._key("tin can", ROOMS, same)
    assert prov._build("tin can", ROOMS, gone, 0) == prov._build("tin can", ROOMS, same, 0)

    # and everything the prompt DOES say must move the key
    for changed in (dict(gone, room=3), dict(gone, label="shelf")):
        assert prov._build("tin can", ROOMS, gone, 0) != prov._build("tin can", ROOMS, changed, 0)
        assert prov._key("tin can", ROOMS, gone) != prov._key("tin can", ROOMS, changed)


def test_cache_key_separates_questions_the_prompt_can_tell_apart():
    """The prompt names rooms by id and the answer is a list of ids, so a
    renumbered house is a DIFFERENT question -- reusing its answer would send
    the agent to a room picked for its number. Same for where the object was
    last seen, which the prompt states."""
    p = RoomPriorProvider(client=None)
    shifted = [dict(r, id=r["id"] + 100) for r in ROOMS]
    assert p._key("tin can", ROOMS, ANCHOR) != p._key("tin can", shifted, ANCHOR)
    other = [dict(r) for r in ROOMS]
    other[2] = dict(other[2], surfaces=["bathtub", "toilet"])
    assert p._key("tin can", ROOMS, ANCHOR) != p._key("tin can", other, ANCHOR)
    assert p._key("tin can", ROOMS, ANCHOR) != p._key(
        "tin can", ROOMS, dict(ANCHOR, room=3))
    assert p._key("tin can", ROOMS, ANCHOR) != p._key(
        "tin can", ROOMS, dict(ANCHOR, label="shelf"))
    # identical question, identical key
    assert p._key("tin can", ROOMS, ANCHOR) == p._key("tin can", list(ROOMS), dict(ANCHOR))


# -------------------------------------------------------------------- answers


def test_answer_becomes_a_ranking():
    c = FakeClient({"rooms": [5, 3, 1]})
    p = RoomPriorProvider(client=c)
    assert p.request("tin can", ROOMS, ANCHOR, 47)
    _wait(p)
    assert p.ranking() == [5, 3, 1]
    assert c.calls == 1


def test_second_ask_is_served_from_cache_without_a_call(tmp_path):
    path = tmp_path / "rooms.json"
    c = FakeClient({"rooms": [5, 3, 1]})
    p = RoomPriorProvider(client=c, cache_path=str(path))
    p.request("tin can", ROOMS, ANCHOR, 47)
    _wait(p)
    assert json.loads(path.read_text())

    c2 = FakeClient({"rooms": [1, 3, 5]})  # would answer differently if asked
    p2 = RoomPriorProvider(client=c2, cache_path=str(path))
    assert p2.request("tin can", ROOMS, ANCHOR, 47)
    assert p2.ranking() == [5, 3, 1]
    assert c2.calls == 0, "a cached A/B must not pay for the model"


@pytest.mark.parametrize("raw", [
    [5, 3], ["5", "3"], ["room 5", "room 3"], ["Room 5", "Room 3"],
])
def test_every_shape_the_model_uses_for_a_room_id_parses(raw):
    """Observed live: the same model returns 2, "2", "room 2" and "Room 2"
    interchangeably. Trimming a lowercase prefix drops the capitalised one."""
    p = RoomPriorProvider(client=FakeClient({"rooms": raw}))
    p.request("tin can", ROOMS, ANCHOR, 47)
    _wait(p)
    assert p.ranking() == [5, 3]


def test_invented_rooms_are_dropped_and_partial_answers_kept():
    c = FakeClient({"rooms": ["room 5", 99, 3]})
    p = RoomPriorProvider(client=c)
    p.request("tin can", ROOMS, ANCHOR, 47)
    _wait(p)
    assert p.ranking() == [5, 3]


def test_a_failed_call_leaves_no_ranking():
    """No answer must mean 'keep the behaviour you already had', not a random
    one: the caller falls back to the positional bonus when ranking() is None."""
    p = RoomPriorProvider(client=FakeClient(RuntimeError("nim down")))
    p.request("tin can", ROOMS, ANCHOR, 47)
    _wait(p)
    assert p.ranking() is None
    assert p.n_errors == 1


def test_garbage_json_leaves_no_ranking():
    p = RoomPriorProvider(client=FakeClient({"surfaces": ["counter"]}))
    p.request("tin can", ROOMS, ANCHOR, 47)
    _wait(p)
    assert p.ranking() is None


def test_one_room_is_not_a_question():
    c = FakeClient({"rooms": [1]})
    p = RoomPriorProvider(client=c)
    assert not p.request("tin can", ROOMS[:1], ANCHOR, 0)
    assert c.calls == 0


def test_reset_clears_the_answer_but_not_the_disk_cache(tmp_path):
    """room.id restarts each episode, so a surviving answer would re-rank an
    unrelated scene the moment an id collided (f6f4ebd)."""
    path = tmp_path / "rooms.json"
    p = RoomPriorProvider(client=FakeClient({"rooms": [5, 3, 1]}), cache_path=str(path))
    p.request("tin can", ROOMS, ANCHOR, 47)
    _wait(p)
    assert p.ranking() == [5, 3, 1]
    p.reset()
    assert p.ranking() is None
    assert p.request("tin can", ROOMS, ANCHOR, 47)  # cache hit, synchronous
    assert p.ranking() == [5, 3, 1]


def test_no_client_is_inert():
    p = RoomPriorProvider(client=None)
    assert not p.request("tin can", ROOMS, ANCHOR, 47)
    assert p.ranking() is None


def test_a_late_answer_does_not_clobber_what_other_episodes_learned(tmp_path):
    """A provider is built per episode and a query can outlive its episode.
    Writing the in-memory dict wholesale would let a straggler holding an old
    snapshot erase every entry written since."""
    path = tmp_path / "rooms.json"
    stale = RoomPriorProvider(client=FakeClient({"rooms": [5, 3, 1]}), cache_path=str(path))

    # a later episode learns something and writes it
    later = RoomPriorProvider(client=FakeClient({"rooms": [1, 3, 5]}), cache_path=str(path))
    later._cache["someone-elses-question"] = [9]
    later._save()

    # now the straggler from the earlier episode finally answers
    stale.request("tin can", ROOMS, ANCHOR, 47)
    _wait(stale)

    on_disk = json.loads(path.read_text())
    assert "someone-elses-question" in on_disk, "a straggler must not erase newer entries"
    assert any(v == [5, 3, 1] for v in on_disk.values())


# --------------------------------------------------- which question is asked


def test_placement_framing_asks_about_the_person_not_the_object():
    """The evaluated layouts are the collector's, and they put relocated objects
    where someone set them down -- 00848's tomato soup can is on a bed in all
    three cross_anchor layouts. Measured on that scene's five rooms, "where does
    it belong" ranks the one bed-less room FIRST 6/6 and "where would someone
    have put it down" ranks it LAST."""
    belongs = RoomPriorProvider(client=None)._build("tin can", ROOMS, ANCHOR, 0)
    place = RoomPriorProvider(client=None, placement=True)._build("tin can", ROOMS, ANCHOR, 0)
    assert belongs != place
    assert "belongs" in place and "set it down" in place
    assert "carried" in place


def test_the_two_framings_never_share_a_cache_entry():
    """They rank the same rooms in opposite orders; sharing an entry would make
    an A/B between them compare nothing."""
    b = RoomPriorProvider(client=None)
    p = RoomPriorProvider(client=None, placement=True)
    assert b._key("tin can", ROOMS, ANCHOR) != p._key("tin can", ROOMS, ANCHOR)


def test_placement_provider_sends_its_own_system_prompt():
    rec = []
    c = FakeClient({"rooms": [5, 3, 1]}, record=rec)

    class Recording(FakeClient):
        def chat(self, system, user, **kw):
            rec.append(system)
            return self.reply

    p = RoomPriorProvider(client=Recording({"rooms": [5, 3, 1]}), placement=True)
    p.request("tin can", ROOMS, ANCHOR, 0)
    _wait(p)
    assert any("person left" in s for s in rec)
