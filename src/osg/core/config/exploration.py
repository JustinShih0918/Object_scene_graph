"""`exploration` group: frontier selection and the search posterior.

Both halves are scored under ONE index, b*d/c, so exploring and re-searching are
not separate subsystems. The `search_*` block is the dynamic-scene half: where an
object could have been moved to, and what a look at a surface is worth. See
`osg/exploration/search_belief.py`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExplorationConfig:
    scorer: str = "vlm"  # vlm | llm_text | nearest | random
    top_n_frontiers: int = 5
    # HybridVoronoiPlanner navigates the medial axis and stops at the graph node
    # nearest the goal, within this radius -- it stops NEAR a goal, not on it.
    # Read by NavAgent both to build the planner and to derive
    # `_frontier_reach_m`; those two must agree, or an ordinary arrival at a
    # frontier is misread as a degenerate stub and the frontier is blocked for
    # having been reached.
    voronoi_goal_near_m: float = 0.7
    frontier_dedup_m: float = 1.0
    frontier_min_cells: int = 8
    subgraph_radius_m: float = 3.0
    images_per_frontier: int = 1  # each image costs ~1-2k ctx tokens
    max_frontiers_per_call: int = 4
    unscored_prior: float = 0.3
    min_path_cost_m: float = 0.5
    # Search posterior (docs/DYNAMIC_SCENES.md, C3). Off by default: it changes
    # where the agent goes, so it must be an explicit A/B. When on, mapped
    # surfaces compete with frontiers under ONE index, b*d/c, so exploring and
    # re-searching stop being separate subsystems.
    search_posterior: bool = False
    # d(x): chance a visit to a surface would find the target if it is there.
    # Taken from the same measured detection rate as the presence filter's
    # absence recall (0.812 over 6790 logged expectations).
    search_detect_prob: float = 0.8
    # Length scale for "things are moved short distances": b decays as
    # exp(-d/L) from where the object was last believed to be.
    #
    # 1.0 m, not the 4.0 m this started at, because the benchmark's own
    # displacements say so: in_anchor relocations move a median 0.72 m and
    # cross_anchor ones 6.06 m, which is a short mode plus a long tail rather
    # than one exponential with a 4 m scale. Scored offline against the true
    # destination over 114 (scene, layout, target) combinations
    # (scripts/rank_search_surfaces.py), share of cases where the true surface
    # lands in the top 5 -- what one episode can afford to inspect:
    #
    #                                    overall   in_anchor  cross_anchor
    #   proximity dropped after absence   12/114      5/57        7/57
    #   L=4.0 with a 0.2 floor            20/114     19/57        1/57
    #   L=1.0, no floor                   36/114     29/57        7/57
    search_proximity_len_m: float = 1.0
    # The floor used to be 0.2, and it was doing damage. `max(exp(-d/L), floor)`
    # clips every candidate past ~6 m to the SAME value, so all of them tie and
    # their order falls to whatever `sorted` does with equal keys -- track id.
    # For a cross-anchor move, where the destination is 6 m away by
    # construction, that discards the only signal left. Without the floor the
    # far candidates stay ordered by distance and cross_anchor recovers from
    # 1/57 to 7/57. Keep it at 0.0 unless something needs a genuine mixture.
    search_proximity_floor: float = 0.0
    # Stop anchoring the search on the last known pose once the map has stopped
    # believing the object is there. 0.0 keeps the anchor forever (shipped).
    #
    # The two halves of the benchmark want opposite models and no mixture serves
    # both -- swept offline, flat and two-scale alike, every setting lands on one
    # frontier. Inspections a greedy search needs to reach the true destination:
    #
    #                       reaches it   median   within 10
    #     in_anchor   prox    23/57         2        23
    #     in_anchor   flat    15/57        17         5
    #     cross       prox    12/57        22         4
    #     cross       flat    20/57        16         9
    #
    # The agent does not have to guess which half it is in: its own presence
    # belief answers -- but the right event is an ARRIVAL, not a belief level.
    # A first attempt keyed this on `presence.p < min_presence` and fired in 56%
    # of in_anchor episodes against the 30% predicted, because presence also
    # decays from ordinary missed expectations while the agent walks past. It
    # dropped the anchor on the half that needs it and cost three episodes on
    # the first scene. `ObjectTrack.absence_arrivals` counts only "I went to
    # look and it was gone".
    search_drop_proximity_after_absence: bool = False
    # Belief carried by the single most plausible mapped surface. The candidate
    # priors are affinity x proximity normalised so the best of them equals this,
    # which separates the ORDERING (what the proximity model is for) from the
    # SCALE (what `search_frontier_weight` prices unexplored space against).
    # 0.5 because that is where the previously tuned model sat -- median top
    # prior 0.479 over 19 (scene, target) pairs -- so sharpening proximity does
    # not silently re-tune the search-versus-explore trade at the same time.
    #
    # Raised from 0.5 to 1.0 after condition E. Matching the previously tuned
    # model's TOP prior (median 0.479) turned out to under-fund the search,
    # because the old model's tail was held up by its 0.2 floor and the new
    # one's is not: measured over 96 episodes, E ran the search in 38 episodes
    # for 1.24 inspections each against C0's 53 and 3.34. The ranking is the
    # part that improved -- E arrived at the surface the object was moved to
    # six times against C0's zero -- so the search deserves to outbid
    # unexplored space more often, not less.
    search_surface_mass: float = 1.0
    # Scales a frontier's utility against a surface's, i.e. the price of
    # preferring unmapped space over a plausible surface. Must be non-zero or
    # the agent stops exploring once its surfaces are exhausted.
    search_frontier_weight: float = 1.0
    # Ask the text LLM where a class of object gets put down, for targets the
    # static table in graph/priors.py does not cover (every YCB target). Cached
    # to disk, so a run is deterministic after the first and the priors used are
    # inspectable afterwards.
    # How close counts as having inspected a surface, and how long to stay
    # committed to reaching one before giving up on it.
    # A surface in plain view counts as searched without driving to it: the
    # binding budget is inspections (about fifty steps each), not travel.
    search_glance_detect_prob: float = 0.35
    # The floor glancing alone may not push a surface past. 0.0 is the shipped
    # behaviour: unbounded compounding, once per keyframe, for every container
    # in view. `_scan_at_viewpoint` refuses exactly this multiplication and says
    # why -- "twelve looks at the same object from the same pose are not twelve
    # independent observations ... measured: doing it dropped SR from 0.429 to
    # 0.286" -- and the glance path was never given the same treatment.
    # 0.2 is not a taste: it is what a real arrival and inspection is worth
    # (1 - search_detect_prob), so it makes the docstring's claim true, that a
    # passing look is weaker evidence than standing there.
    search_glance_floor: float = 0.0
    search_glance_range_m: float = 4.0
    # Finishing the room you are in beats crossing the house and coming back.
    search_same_room_bonus: float = 4.0
    # ...but that bonus is a PRIOR ("the object is in this room") that never got
    # a likelihood update, and an un-updated prior in a container-dense room is
    # an absorbing state. Scene 00848 is a kitchen holding 35 cabinets, 20
    # shelves and 13 refrigerators; every one keeps its x4 however many come up
    # empty, and since utility divides by path cost the nearest of the 68 always
    # outbids a frontier discounted to 0.3. Measured over the six 00848 pitcher
    # episodes under condition N: 11 surfaces inspected, ZERO frontier
    # selections in 286 steps, the target never once in view, SR 0.15.
    #
    # Each fruitless ARRIVAL in a room multiplies its bonus by (1 - rate) --
    # glances and give-ups on the way do not count, the same distinction
    # InspectionLog already makes between a look and a visit. 0.0 disables it
    # and reproduces every earlier condition exactly.
    search_room_saturation: float = 0.0
    # Arrivals a room gets for free before any decay starts.
    #
    # Without this, condition V decayed the bonus from the FIRST fruitless
    # arrival and cost 6 episodes over 90 matched (in_anchor 0.822 -> 0.733).
    # The reason is in the distribution: over condition N's 96 episodes NO
    # success ever needed more than 7 surface inspections (median 0, p90 5),
    # while the stuck 00848 failures need 11-13. Decaying from the first
    # arrival therefore weakens the room prior precisely inside the window
    # where successes happen, and the agent -- pushed out of a room it had not
    # finished -- enters the next one with a fresh x4 and scatters, doing MORE
    # inspections than before (10 against 2 on the episodes V lost).
    #
    # So the room keeps its full bonus until it has already given more looks
    # than any success has ever needed. Beyond that the room is, empirically,
    # not the room.
    search_room_saturation_free: int = 0
    # How far the bonus may fall. 1.0 means saturation can cancel the bonus but
    # never invert it into a penalty: a room that has disappointed becomes
    # ordinary, not worse than one never visited. Below 1.0 actively pushes the
    # agent out of a room it has been failing in.
    search_room_saturation_floor: float = 1.0
    search_arrival_m: float = 1.2
    # Turns spent looking AT a surface on arrival, before its belief is scored.
    # `_mark_surface_searched` multiplies belief by (1 - search_detect_prob) on
    # a single frame taken at whatever heading the follower stopped on. In
    # condition D the search reached the true surface five times and converted
    # one. A few turns are cheap against the ~50 steps an inspection costs.
    search_face_turns: int = 8
    # Turns spent sweeping AFTER the agent has faced the surface it came to
    # inspect. 0 is the shipped behaviour and reproduces every earlier
    # condition; 12 at turn_deg=30 is one full revolution.
    #
    # `search_face_turns` cannot do this job, which cost a wasted arm to learn:
    # `face_surface` zeroes it the moment the heading error drops below 15
    # degrees, so raising it 8 -> 12 moved the turns actually spent from a mean
    # 4.1 to 4.5 and changed nothing else. Facing is not scanning.
    #
    # The bucket this aims at: on 00848, ten of the fifteen cross_anchor
    # failures never get the target into the frustum at all, and six of those
    # come within 3 m of it. The object was RELOCATED, so it is usually on a
    # surface next to the one the posterior chose, and facing the chosen one
    # looks straight past it.
    #
    # The cost is real and is the thing to watch: every failure in this
    # benchmark ends on the step cap, and this spends up to 12 extra steps at
    # each of a median 8 surfaces.
    search_scan_turns: int = 0
    search_max_steps: int = 60
    # Credit for a surface the agent set off towards but never reached: it has
    # barely been ruled out, and spending full belief on it would retire the
    # very surfaces that were never inspected.
    search_unreached_credit: float = 0.25
    affinity_llm: bool = False
    # Rank affinity over the container categories the MAP actually contains,
    # rather than over all of CONTAINER_CATEGORIES.
    #
    # A prior whose top choices do not exist in this house is not a prior. The
    # LLM ranks "blue plastic pitcher" as counter > table > shelf > ..., and
    # scene 00848 contains ZERO counters and ZERO tables out of 343 tracks --
    # its counter runs are labelled `cabinet`, which is unlisted and so takes
    # UNLISTED_AFFINITY, below shelf. The 35 cabinets that ARE the counters
    # therefore rank beneath everything, and what reaches the search posterior
    # is a near-flat prior over 68 containers.
    #
    # Grounding is surgical here by construction: 00829 and 00880 contain all
    # 14 categories, so their grounded option set IS the full set and their
    # ranking cannot change. Only 00848 is affected.
    affinity_grounded: bool = False
    affinity_cache: str = "outputs/affinity_cache.json"
    # --- room posterior (LLM) -------------------------------------------
    # Ask the model WHICH ROOM the target moved to, once a room has been
    # searched and refused, and use the answer in place of the positional
    # `search_same_room_bonus`.
    #
    # The bonus asserts "the object is in the room I am in". On the in_anchor
    # half (median relocation 0.72 m) that is right; on cross_anchor (6.06 m)
    # it is wrong, and it is wrong at the same time as proximity, which at
    # exp(-6.06/1.0) = 0.0023 already hands the origin room a 209x advantage.
    # Measured over M2..W: cross_anchor 0.44-0.53 for eleven conditions while
    # in_anchor reached 0.822.
    #
    # This is the room level deliberately. `author_semantic_layouts.choose_
    # surface` draws BOTH layout types from the same home categories, so the
    # halves differ only by destination region -- the one level at which an
    # answer can be selective for the half that is failing. The earlier
    # frontier scorer asked per-frontier and got 0.3/0.35/0.4 for it.
    room_posterior_llm: bool = False
    # Fruitless ARRIVALS in a room before the question is asked. Arrivals, not
    # the presence belief: `_last_known_target_xy` records a `presence.p`
    # threshold firing on 56% of in_anchor episodes against 30% predicted,
    # because presence also decays from ordinary missed expectations while the
    # agent walks past. Presence is what the prompt READS; the trigger is an
    # event. 2 keeps a room that merely disappointed once from being abandoned.
    room_posterior_after: int = 2
    # Top-to-bottom dynamic range of the multiplier, geometric and symmetric in
    # log space: the first room gets `spread`, the last `1/spread`. 4.0 makes
    # the top room exactly the x4 the same-room bonus asserts today, so when the
    # model agrees with the agent's position the posterior IS the shipped one.
    room_posterior_spread: float = 4.0
    # Keep the positional bonus alongside the model's answer. False (default)
    # REPLACES it -- two priors over the same variable, one positional and one
    # semantic, must not both multiply in.
    room_posterior_keep_bonus: bool = False
    room_posterior_cache: str = "outputs/room_prior_cache.json"
    # This call gets its own, much longer budget than `llm.timeout_s` (120 s).
    # Measured on nvidia/nemotron-3.5-lightning-30b-a3b, the only text model this
    # NIM account still serves: a room ordering costs a median ~7.8k completion
    # tokens of reasoning and 60-300 s, against ~7 s for the affinity call it
    # shares an endpoint with. It can afford that because it is asynchronous and
    # its answer is long-lived -- room ids are stable for the episode, so a reply
    # landing 150 steps late still re-ranks the rest of the search. Turning the
    # reasoning OFF makes it 100x faster and, measured over five targets, ranks
    # the kitchen below two bedrooms every time. The latency is the answer.
    # 420, not 300: a measured end-to-end answer took 255 s and the endpoint's
    # spread is wide (134-300+ s over six targets). At 300 the tail lands on the
    # timeout, and a timeout is not a soft failure here -- it is three retries
    # and no cache entry, so the NEXT episode pays the same cost again.
    room_posterior_timeout_s: float = 420.0
    # Wait this long for a COLD answer before carrying on. 0.0 is pure
    # asynchrony, which was measured to deliver nothing: 36 episodes on 00829,
    # 10 queries asked, 8 answered, 0 applied, and X identical to Y episode for
    # episode. An answer costs ~180 s against an ~80 s episode and the provider
    # is per-episode, so every reply outlived the run that wanted it -- the same
    # shape as the frontier scorer that made 63 calls and changed no selection.
    # A warm cache resolves synchronously and pays none of this.
    room_posterior_block_s: float = 0.0
    # Clamp on the BOTTOM of the multiplier. 0.0 is the symmetric term (top
    # room x spread, bottom room x 1/spread); 1.0 makes it promote-only.
    #
    # Symmetric was measured to be dangerous exactly where the positional prior
    # it replaces is correct. On 00829, across the nine episodes the posterior
    # reached, the control scored 5/9 and the symmetric posterior 2/9 -- every
    # loss ran to the 500-step cap and one had the target in view 64 times
    # against the control's 26. The model named a room other than the agent's in
    # seven of the nine, so a correct room fell from x4 to x0.25.
    #
    # `search_room_saturation_floor` is the same guard one level down and says
    # why: a room that has disappointed becomes ordinary, not worse than one
    # never visited. A model's opinion about where an object is NOT deserves the
    # same restraint.
    room_posterior_floor: float = 0.0
    # Ask where a PERSON would have set the object down, rather than where the
    # object belongs.
    #
    # The evaluated layouts are the collector's (`outputs/substituted_layouts`,
    # whose `authoring` block reads "imported_by": import_collector_layouts.py),
    # NOT the kitchen-biased ones author_semantic_layouts.py generates. They put
    # relocated objects on beds, nightstands and desks regardless of class:
    # 00848's tomato soup can is on a bed in all three cross_anchor layouts.
    # Measured on 00848's five rooms, "where does it belong" ranks the one room
    # with NO bed first for 6 targets of 6; "where would someone have put it
    # down" ranks that same room last.
    room_posterior_placement: bool = False
    # Information-gain weighting: boost frontiers that expose more unknown area
    # (estimated as the count of UNKNOWN costmap cells within info_gain_radius_m
    # of the frontier), so exploration commits to directions that open large
    # unexplored regions instead of crawling the nearest small frontier. A
    # frontier's score is multiplied by (1 + info_gain_weight * gain/gain_max),
    # normalized against the best candidate each round. 0 weight disables it.
    info_gain_weight: float = 2.0
    info_gain_radius_m: float = 2.5
    # Continuity / momentum bonus: prefer the next frontier to lie AHEAD of the
    # agent's current heading, so exploration sweeps continuously instead of the
    # greedy argmax ping-ponging between far-apart frontiers (~30 steps/trip).
    # 0 = off; higher = stronger preference for staying the course.
    continuity_weight: float = 0.0
    # Line-of-sight visibility down-weighting: multiply the score of frontiers
    # the agent has clear line of sight to (no wall between => same room) by this
    # factor, so exploration prefers occluded, behind-a-doorway frontiers that
    # open new rooms. 1.0 = off; <1.0 penalizes visible/same-room frontiers.
    # Drive to the frontier's free-snapped centroid instead of an UNKNOWN
    # frontier cell. On the navmesh, unknown space is snapped by snap_point to
    # an arbitrary nearby navigable point, so the follower reports
    # arrived-or-unreachable at once: measured 289 stub-blocks vs 24 give-ups
    # over 100 episodes, 53% of selections repeating an earlier one, and 2x the
    # planned distance walked. See exploration/selector.frontier_goal_xy.
    frontier_goal_free_cell: bool = False
    # Measure the RANKING path cost to the free centroid while still driving to
    # the frontier cell. Fixes the planner failures without the coverage loss
    # that moving the drive goal causes -- see select_frontier.
    frontier_cost_free_cell: bool = False
    los_visibility_penalty: float = 1.0


