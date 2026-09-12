"""A staircase read off the height layer, so the target is its foot.

Stage 4's detector asked which cells are steppable by their local height step
and fragmented every flight into riser edges, because a tread's interior is
flat. This asks which cells sit strictly between this floor and the next and
are contiguous, and keeps a run only if its heights span a metre: a table is a
plateau, a flight joins two storeys.

The detector's `stairs` mask, seen from below, put its target 2.7 m beside the
true foot on 00821, and twelve climbs steered at it rose 0.00 m. The lowest cell
of an intermediate-height run IS the foot.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.mapping.costmap import FREE, Costmap2D
from osg.mapping.stairs import find_flights

FLOOR_Y = 0.0


def _costmap(size_m=12.0, res=0.05):
    cm = Costmap2D(resolution=res, size_m=size_m, track_height=True)
    cm.grid[:] = FREE
    return cm


def _paint(cm, x0, x1, z0, z1, y):
    r0, c0 = cm.world_to_grid(np.array([x0, z0]))
    r1, c1 = cm.world_to_grid(np.array([x1, z1]))
    r0, r1 = sorted((int(r0), int(r1))); c0, c1 = sorted((int(c0), int(c1)))
    cm.height[r0:r1 + 1, c0:c1 + 1] = y


def _staircase(cm, x_foot=1.0, z=2.0, treads=12, tread_m=0.28, riser_m=0.17, width_m=1.0):
    """Treads rising along +x from `x_foot`, 1 m wide, contiguous."""
    for i in range(treads):
        x0 = x_foot + i * tread_m
        _paint(cm, x0, x0 + tread_m, z, z + width_m, FLOOR_Y + (i + 1) * riser_m)


def test_a_flight_is_found_and_its_foot_is_the_lowest_tread():
    cm = _costmap()
    _staircase(cm)
    flights = find_flights(cm, FLOOR_Y, new_level_m=2.6, min_span_m=1.0, min_cells=50)
    assert len(flights) == 1
    f = flights[0]
    assert f.kind == "up"
    assert f.span_m >= 1.0
    assert abs(f.foot_xy[0] - 1.0) < 0.35, "foot at the first tread, not mid-flight"
    assert abs(f.foot_xy[1] - 2.5) < 0.6
    assert f.top_xy[0] > f.foot_xy[0] + 2.0


def test_a_table_is_a_plateau_and_not_a_flight():
    cm = _costmap()
    _paint(cm, 1.0, 3.0, 2.0, 4.0, FLOOR_Y + 0.75)  # a 2x2 m table top
    assert find_flights(cm, FLOOR_Y, min_cells=50) == []


def test_a_descent_is_read_from_the_treads_below():
    cm = _costmap()
    for i in range(12):
        x0 = 1.0 + i * 0.28
        _paint(cm, x0, x0 + 0.28, 2.0, 3.0, FLOOR_Y - (i + 1) * 0.17)
    flights = find_flights(cm, FLOOR_Y, new_level_m=2.6, min_span_m=1.0, min_cells=50)
    assert len(flights) == 1 and flights[0].kind == "down"
    # The mouth on THIS floor is the highest tread, nearest the lip.
    assert abs(flights[0].foot_xy[0] - 1.0) < 0.35


def test_a_flight_on_another_storey_is_not_this_floors_flight():
    """Cells a whole storey up are the next floor, not treads."""
    cm = _costmap()
    _paint(cm, 1.0, 4.0, 2.0, 4.0, FLOOR_Y + 2.9)
    assert find_flights(cm, FLOOR_Y, new_level_m=2.6, min_cells=50) == []


def test_a_huge_sloped_region_is_capped():
    cm = _costmap(size_m=12.0)
    # a gentle ramp over the whole map: thousands of cells spanning a metre
    n = cm.grid.shape[0]
    cm.height[:] = np.linspace(0.3, 1.5, n)[:, None]
    assert find_flights(cm, FLOOR_Y, min_cells=50, max_cells=4000) == []


def test_nothing_without_a_height_layer():
    cm = Costmap2D(resolution=0.05, size_m=6.0, track_height=False)
    assert find_flights(cm, FLOOR_Y) == []
