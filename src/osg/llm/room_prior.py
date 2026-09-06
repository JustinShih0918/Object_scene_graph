"""Which ROOM did it go to? Asked when a room has been searched and refused.

The search posterior ranks surfaces under one index, `b*d/c`, and two of its
three terms are positional: proximity to where the object was last seen
(`exp(-d/1.0)`) and a flat x4 for the room the agent is standing in. On the
`in_anchor` half of the benchmark, where the median relocation is 0.72 m, both
are right. On the `cross_anchor` half, where it is 6.06 m, both are wrong
together -- `exp(-6.06)` is 0.0023 against `exp(-0.72)` = 0.487, a 209x handicap
on the true surface, and the room bonus holds the agent in the room the object
just left. Measured over the M2..W campaign, cross_anchor sat at 0.44-0.53 for
eleven consecutive conditions while in_anchor climbed to 0.822.

What is NOT wrong is the map. Every episode loads the static prior map first --
343 tracks in 00848 -- so the kitchen, its refrigerator and its cabinets are all
in `sg.containers` before the agent's first step. The failing episodes do not
fail to find the escape route; they rank bedroom furniture above it. One episode
spent steps 47-188 inspecting a desk, a shelf and two beds for a TIN CAN, and
reached the room with the refrigerator at step 370 with 130 steps left.

That is what this asks about, and the level matters. `author_semantic_layouts`
draws BOTH layout types from the same home categories, so the two halves differ
only by destination REGION -- which makes the room the only level at which an
answer can be selective for the half that is failing. An earlier text scorer
asked the same model to rate individual frontiers and produced 0.3/0.35/0.4,
because a frontier's local subgraph reads the same anywhere in a house; the
answer was flat because the question had no content.

Two properties are load-bearing and both are copied from `llm/affinity.py`:

- **A permutation, never a score.** The caller assigns the spread. A model's
  0.3-vs-0.4 is uncalibrated and gets divided by a path cost that spans 7x; a
  rank the caller turns into a chosen multiplier does not.
- **Cached to disk.** A run is deterministic after the first, an A/B is not
  paying for the model, and the priors the agent used are inspectable
  afterwards -- which matters, because docs/INVESTIGATION.md records an earlier
  LLM scorer that produced byte-identical trajectories to a geometric heuristic
  while adding a network dependency.

The evidence fed in is the presence filter's, because it is the only channel in
the system carrying NEGATIVE information -- `log((1-r)/(1-q))` when an object is
expected and absent. It goes in as counts, never as a threshold:
`strategy._last_known_target_xy` records what happened when `presence.p` was
used as a trigger, "fired in 56% of in_anchor episodes against the 30%
predicted, because presence also decays from ordinary missed expectations while
the agent merely walks past". A scalar that noisy cannot carry a threshold, but
`p=0.07 over 23 observations` and `p=0.82 over 0` are different sentences and a
reader can tell them apart. So: presence is what the model reads, a permutation
is what it writes, and the trigger stays an event the caller owns.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import prompts

logger = logging.getLogger(__name__)


class RoomPriorProvider:
    """Callable-ish: `request(...)` submits, `ranking()` returns the latest
    room order, or None while nothing has been answered yet.

    Asynchronous for the same reason `AsyncScorer` is -- the real-time claim
    concerns the mapping/control loop -- but the staleness that made the old
    frontier scorer useless does not apply here. That scorer's answers named
    frontier ids, which are reassigned on every extraction, so a reply arriving
    seven seconds later described frontiers that no longer existed. Room ids are
    stable for the whole episode, so an answer stays valid for hundreds of steps
    and it does not matter which step it lands on.
    """

    def __init__(
        self,
        client=None,
        cache_path: Optional[str] = None,
        max_rooms: int = 12,
        max_surfaces_per_room: int = 8,
        placement: bool = False,
    ) -> None:
        self.client = client
        # Which question to ask: where the object BELONGS, or where a person
        # would have PUT IT DOWN. On these layouts the two framings rank the
        # rooms in opposite orders and only one matches where the objects are --
        # see prompts.ROOM_PRIOR_PLACEMENT_USER.
        self.placement = bool(placement)
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_rooms = int(max_rooms)
        self.max_surfaces_per_room = int(max_surfaces_per_room)
        self.n_calls = 0
        self.n_errors = 0
        self.last_error: Optional[str] = None
        self.last_prompt: Optional[str] = None
        self._cache: Dict[str, List[int]] = {}
        # Created on the first real call. A NavAgent -- and so a provider -- is
        # rebuilt per episode, and most episodes never trigger a query at all;
        # an eagerly built pool would be one idle thread per episode for the
        # whole run.
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._latest: Optional[List[int]] = None
        self._in_flight = False
        if self.cache_path and self.cache_path.is_file():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Room ids restart from 0/1 each episode (a fresh RoomSegmenter per
        NavAgent), so a surviving answer from a previous, unrelated scene would
        re-rank this one the moment an id collided. The same cross-episode leak
        f6f4ebd fixed in the scorer caches.

        The DISK cache is keyed by content, not by id, so it survives -- that is
        what makes the second run of an A/B free and deterministic.
        """
        with self._lock:
            self._latest = None

    def shutdown(self) -> None:
        with self._lock:
            ex, self._executor = self._executor, None
        if ex is not None:
            ex.shutdown(wait=False)

    def ranking(self) -> Optional[List[int]]:
        with self._lock:
            return list(self._latest) if self._latest is not None else None

    # --------------------------------------------------------------- prompt

    @staticmethod
    def _anchor_bucket(anchor: Optional[dict]) -> str:
        """How strongly the map has refuted the object's last known pose, in
        three buckets rather than as a probability.

        Presence is fed in because it is the only NEGATIVE channel in the
        system, but it must not enter as a bare threshold on `p`:
        `strategy._last_known_target_xy` records a `presence.p < min_presence`
        trigger firing on 56% of in_anchor episodes against the 30% predicted,
        because p decays from ordinary missed expectations while the agent walks
        past. What separates real absence from that drift is `n_expected` -- p =
        0.07 over 23 checks and p = 0.82 over 0 are different claims and a
        reader can tell them apart, which is the whole reason to hand this to a
        model rather than to a comparison.

        Buckets rather than the raw counts so that the prompt says exactly what
        the cache key says (see `_key`); the distinction that carries the
        information is looked-and-gone vs never-rechecked, not 23 versus 24.
        """
        if not anchor:
            return "unknown"
        n = int(anchor.get("n_expected", 0))
        if n <= 0:
            return "unchecked"
        if float(anchor.get("p", 1.0)) >= 0.5:
            return "still there"
        return "gone" if n >= 5 else "probably gone"

    def _build(self, target: str, rooms: Sequence[dict], anchor: Optional[dict],
               n_disbelieved: int) -> str:
        """`rooms` is a list of dicts the caller assembles from the scene graph:
        {id, label, surfaces: [str], n_surfaces, n_looked}."""
        # Room LABELS are usually absent: `LLMTextScorer.label_rooms` is the only
        # thing that sets `RoomNode.label`, and it is unreachable under the
        # NullScorer every experiment preset selects. Rather than restore a
        # second call and a second cache to produce a word the surface list
        # already implies -- "bed, desk, shelf" is a bedroom -- the prompt asks
        # the model to infer the type. A label is used when one happens to exist.
        lines = []
        for r in rooms[: self.max_rooms]:
            surf = ", ".join(r["surfaces"][: self.max_surfaces_per_room]) or "nothing mapped"
            label = f" ({r['label']})" if r.get("label") else ""
            lines.append(f"- room {r['id']}{label}: {surf}")
        # The anchor sentence is the single most informative line in the prompt:
        # it is the only one that says the object has actually MOVED, and it is
        # the presence filter's negative channel stated in words.
        # The anchor sentence is the only line that says the object has MOVED,
        # and it is the presence filter's negative channel stated in words. It is
        # binary -- was it refuted or not -- rather than a count, for the same
        # reason the cache key is: see `_key`.
        bucket = self._anchor_bucket(anchor)
        anchor_text = "The robot does not know where it used to be."
        if anchor:
            where = f"room {anchor['room']}" + (
                f"'s {anchor['label']}" if anchor.get("label") else "")
            anchor_text = {
                "unchecked": f"It was mapped on {where}, and the robot has not been"
                             " back to check whether it is still there.",
                "still there": f"It was mapped on {where} and still seems to be there.",
                "probably gone": f"It was mapped on {where}. The robot has looked"
                                 " there a few times since and not seen it.",
                "gone": f"It was mapped on {where}. The robot has looked there many"
                        " times since and it is not there any more.",
            }.get(bucket, anchor_text)
        template = (prompts.ROOM_PRIOR_PLACEMENT_USER if self.placement
                    else prompts.ROOM_PRIOR_USER)
        return template.format(
            target=str(target).strip(),
            anchor_text=anchor_text,
            rooms_text="\n".join(lines),
        )

    def _key(self, target: str, rooms: Sequence[dict], anchor: Optional[dict]) -> str:
        """Content-addressed, so the cache is portable across episodes and runs.

        Room ids ARE in the key, even though they are per-episode integers and
        it costs hit rate, because the PROMPT names rooms by id -- "- room 4:
        cabinet, refrigerator" -- and the answer is a list of those ids. Two
        episodes whose segmentation assigned the same furniture to different
        numbers are not the same question, and reusing one's answer for the
        other silently sends the agent to a room chosen for its number. Ids
        are stable in practice, since every episode segments the same prior
        map; when they are not, re-asking is the correct cost.

        The key and the PROMPT carry exactly the same facts, which is a
        requirement rather than a tidiness: a prompt richer than its key means
        two different questions share one cache entry, and which answer a run
        gets depends on which fired first. That is a hidden nondeterminism in
        the middle of an A/B. So per-room search progress is in neither -- the
        agent's effort is already expressed by `search_room_saturation` and by
        InspectionLog on the individual surfaces, one level down, and asking the
        model to re-encode it cost ~3x the distinct questions (5 rooms of
        searched/unsearched flags) for a term the posterior already has.
        """
        parts = []
        for r in sorted(rooms, key=lambda x: int(x["id"])):
            surf = ",".join(sorted(str(s).lower() for s in r["surfaces"]))
            parts.append(f"{int(r['id'])}:{str(r.get('label') or '?').lower()}[{surf}]")
        where = ""
        if anchor:
            where = f"@{int(anchor.get('room', -1))}:{str(anchor.get('label') or '?').lower()}"
        # The framing is part of the question. Without it the two prompts --
        # measured to rank the same rooms in OPPOSITE orders -- would share a
        # cache entry, and an A/B between them would compare nothing.
        framing = "place" if self.placement else "belong"
        return (f"{framing}|{str(target).lower().strip()}|"
                f"{RoomPriorProvider._anchor_bucket(anchor)}{where}|" + "|".join(parts))

    # -------------------------------------------------------------- request

    def request(self, target: str, rooms: Sequence[dict],
                anchor: Optional[dict] = None, n_disbelieved: int = 0,
                block_s: float = 0.0) -> bool:
        """Submit a query. Returns True if this call produced or started an
        answer, False if one was already in flight.

        A cache hit resolves synchronously, so the second run of an A/B pays no
        latency at all and reproduces the first exactly.

        `block_s > 0` waits that long for a MISS to answer. Purely asynchronous
        delivery was measured not to work: over 36 episodes on 00829 the query
        was requested 10 times and answered 8, and the posterior was applied
        ZERO times -- an answer costs ~180 s against an ~80 s episode, and the
        provider is rebuilt per episode, so every answer landed after the run
        that asked for it had ended. X and Y came out identical, episode for
        episode. That is the same shape as the frontier scorer
        docs/INVESTIGATION.md retired: 63 calls that never changed a selection.
        Blocking costs the control loop one stall per episode on a cold cache
        and nothing at all on a warm one, and it is the only way the treatment
        is actually delivered. Report it as decision latency, which is what
        `AsyncScorer` says that number is for.
        """
        rooms = [r for r in rooms if r.get("surfaces")]
        if len(rooms) < 2:
            return False  # nothing to order
        key = self._key(target, rooms, anchor)
        if key in self._cache:
            with self._lock:
                self._latest = self._remap(self._cache[key], rooms)
            return True
        if self.client is None:
            return False
        with self._lock:
            if self._in_flight:
                return False
            self._in_flight = True
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="room-prior"
                )
            ex = self._executor
        fut = ex.submit(self._run, key, str(target), list(rooms), anchor,
                        int(n_disbelieved))
        if block_s > 0.0:
            try:
                fut.result(timeout=float(block_s))
            except Exception:
                pass  # timed out or failed; the caller keeps its current prior
        return True

    @staticmethod
    def _remap(ranked: Sequence[int], rooms: Sequence[dict]) -> List[int]:
        """Keep only ids this episode actually has. A cached answer is keyed by
        room COMPOSITION, so the ids it carries are whatever the episode that
        first asked happened to use; an id that no longer exists must not
        silently rank a different room."""
        valid = {int(r["id"]) for r in rooms}
        return [int(i) for i in ranked if int(i) in valid]

    def _run(self, key: str, target: str, rooms: List[dict],
             anchor: Optional[dict], n_disbelieved: int) -> None:
        try:
            user = self._build(target, rooms, anchor, n_disbelieved)
            self.last_prompt = user
            self.n_calls += 1
            system = (prompts.ROOM_PRIOR_PLACEMENT_SYSTEM if self.placement
                      else prompts.ROOM_PRIOR_SYSTEM)
            reply = self.client.chat(system, user, json_response=True)
            ranked = self._parse(reply, rooms)
            if ranked is None:
                self.n_errors += 1
                return
            self._cache[key] = ranked
            self._save()
            with self._lock:
                self._latest = list(ranked)
                self.last_error = None
        except Exception as exc:  # a missing prior is not worth failing a run over
            self.n_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("room prior call failed for %r: %s", key, exc)
        finally:
            with self._lock:
                self._in_flight = False

    @staticmethod
    def _parse(reply, rooms: Sequence[dict]) -> Optional[List[int]]:
        """A partial answer is still an answer: rooms the model omits are
        ranked last by the caller, which is what dropping them means. A reply
        that names nothing real is not, and returns None so the caller keeps the
        behaviour it already had."""
        raw = reply.get("rooms") if isinstance(reply, dict) else None
        if not isinstance(raw, list):
            return None
        valid = {int(r["id"]) for r in rooms}
        out: List[int] = []
        for item in raw:
            # The model returns 2, "2", "room 2" and "Room 2" interchangeably.
            # `lstrip("room ")` handles the first three and silently drops the
            # fourth, so pull the digits out instead of trimming a prefix.
            digits = re.search(r"-?\d+", str(item))
            if digits is None:
                continue
            try:
                rid = int(digits.group(0))
            except ValueError:
                continue
            if rid in valid and rid not in out:
                out.append(rid)
        return out or None

    def _save(self) -> None:
        """Merge, then write. A provider is built per EPISODE and a query can
        outlive the episode that asked it -- `shutdown(wait=False)` lets the
        worker finish, which is what makes a late answer still worth having.
        Writing `self._cache` wholesale would then let episode 3's straggler,
        holding the file as it looked at episode 3's start, overwrite everything
        episodes 4 and 5 have since learned. Re-reading first is what makes the
        cache accumulate across a 36-episode run instead of thrashing.
        """
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            merged: Dict[str, List[int]] = {}
            if self.cache_path.is_file():
                try:
                    on_disk = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    if isinstance(on_disk, dict):
                        merged.update(on_disk)
                except (OSError, json.JSONDecodeError):
                    pass
            merged.update(self._cache)
            self._cache = merged
            # Write-and-rename, so a reader in another episode never sees a
            # half-written file and silently starts from an empty cache.
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(merged, indent=1, sort_keys=True), encoding="utf-8")
            tmp.replace(self.cache_path)
        except OSError as exc:
            logger.warning("could not write room prior cache: %s", exc)


def rank_multipliers(ranked: Sequence[int], all_ids: Sequence[int],
                     spread: float, floor: float = 0.0) -> Dict[int, float]:
    """Turn a permutation into the multiplier the search posterior can use.

    Geometric from `spread` at the top to `1/spread` at the bottom, so the term
    is symmetric in log space and its dynamic range is a number the caller
    chose rather than one the model happened to emit. Rooms the model dropped
    take the floor -- being left out of the ordering IS the answer that they are
    the least likely.

    With spread = 4.0 the top room gets exactly the x4 `search_same_room_bonus`
    asserts today, which is what lets this REPLACE that term rather than fight
    it: when the model agrees the agent is already in the right room, the
    posterior it produces is the one the shipped code already computes.

    `floor` clamps the bottom, and 1.0 makes the term PROMOTE ONLY -- the room
    the model likes is lifted, and one it dislikes becomes ordinary rather than
    actively repellent. That guard exists because the symmetric version was
    measured to be dangerous where the positional prior it replaces is right:
    on 00829, over the nine episodes the posterior reached, the control scored
    5/9 and the symmetric posterior 2/9, every loss running to the 500-step cap,
    one of them with the target in view 64 times against the control's 26. The
    model named a room other than the agent's in seven of those nine, so the
    correct room went from x4 to x0.25 -- a 16x swing against a room that was
    right. `search_room_saturation_floor` carries the identical argument one
    level down, in its own words: "a room that has disappointed becomes
    ordinary, not worse than one never visited".
    """
    ids = [int(i) for i in all_ids]
    if spread <= 1.0 or not ids:
        return {i: 1.0 for i in ids}
    ranked = [int(i) for i in ranked if int(i) in set(ids)]
    missing = [i for i in ids if i not in set(ranked)]
    order = ranked + missing
    n = len(order)
    if n == 1:
        return {order[0]: 1.0}
    out: Dict[int, float] = {}
    for pos, rid in enumerate(order):
        # +1 at the top, -1 at the bottom
        t = 1.0 - 2.0 * pos / (n - 1)
        out[rid] = max(float(floor), float(spread ** t))
    return out
