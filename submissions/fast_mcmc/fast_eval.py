from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def _rects_overlap(cx1, cy1, hw1, hh1, cx2, cy2, hw2, hh2, gap) -> bool:
    return (abs(cx1 - cx2) < (hw1 + hw2 + gap)) and (abs(cy1 - cy2) < (hh1 + hh2 + gap))


@njit(cache=True)
def hpwl_for_net(pos_xy: np.ndarray, net_ptr: np.ndarray, net_macros: np.ndarray, net_id: int) -> float:
    a = int(net_ptr[net_id])
    b = int(net_ptr[net_id + 1])
    if b - a <= 1:
        return 0.0
    minx = 1e30
    maxx = -1e30
    miny = 1e30
    maxy = -1e30
    for k in range(a, b):
        m = int(net_macros[k])
        x = pos_xy[m, 0]
        y = pos_xy[m, 1]
        if x < minx:
            minx = x
        if x > maxx:
            maxx = x
        if y < miny:
            miny = y
        if y > maxy:
            maxy = y
    return (maxx - minx) + (maxy - miny)


@njit(cache=True)
def total_hpwl(pos_xy: np.ndarray, net_ptr: np.ndarray, net_macros: np.ndarray) -> float:
    wl = 0.0
    for net_id in range(len(net_ptr) - 1):
        wl += hpwl_for_net(pos_xy, net_ptr, net_macros, net_id)
    return wl


@njit(cache=True)
def init_grid(rows: int, cols: int, max_per_bin: int) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.full((rows, cols, max_per_bin), -1, dtype=np.int32)
    counts = np.zeros((rows, cols), dtype=np.int32)
    return grid, counts


@njit(cache=True)
def _bin_range_for_rect(x0, y0, x1, y1, bin_w, bin_h, rows, cols):
    c0 = int(math.floor(x0 / bin_w))
    c1 = int(math.floor(x1 / bin_w))
    r0 = int(math.floor(y0 / bin_h))
    r1 = int(math.floor(y1 / bin_h))
    if c0 < 0:
        c0 = 0
    if r0 < 0:
        r0 = 0
    if c1 >= cols:
        c1 = cols - 1
    if r1 >= rows:
        r1 = rows - 1
    if c1 < 0:
        c1 = 0
    if r1 < 0:
        r1 = 0
    return r0, r1, c0, c1


@njit(cache=True)
def grid_insert_macro(
    grid: np.ndarray,
    counts: np.ndarray,
    macro_id: int,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    bin_w: float,
    bin_h: float,
    rows: int,
    cols: int,
) -> bool:
    x0 = cx - hw
    y0 = cy - hh
    x1 = cx + hw
    y1 = cy + hh
    r0, r1, c0, c1 = _bin_range_for_rect(x0, y0, x1, y1, bin_w, bin_h, rows, cols)
    ok = True
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            n = int(counts[r, c])
            if n >= grid.shape[2]:
                ok = False
                continue
            grid[r, c, n] = macro_id
            counts[r, c] = n + 1
    return ok


@njit(cache=True)
def grid_remove_macro(
    grid: np.ndarray,
    counts: np.ndarray,
    macro_id: int,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    bin_w: float,
    bin_h: float,
    rows: int,
    cols: int,
):
    x0 = cx - hw
    y0 = cy - hh
    x1 = cx + hw
    y1 = cy + hh
    r0, r1, c0, c1 = _bin_range_for_rect(x0, y0, x1, y1, bin_w, bin_h, rows, cols)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            n = int(counts[r, c])
            if n <= 0:
                continue
            # linear scan: remove by swap-with-last
            for k in range(n):
                if int(grid[r, c, k]) == macro_id:
                    last = int(grid[r, c, n - 1])
                    grid[r, c, k] = last
                    grid[r, c, n - 1] = -1
                    counts[r, c] = n - 1
                    break


@njit(cache=True)
def grid_has_overlap_for_macro(
    pos_xy: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    grid: np.ndarray,
    counts: np.ndarray,
    macro_id: int,
    new_cx: float,
    new_cy: float,
    bin_w: float,
    bin_h: float,
    rows: int,
    cols: int,
    gap: float,
) -> bool:
    hw = half_w[macro_id]
    hh = half_h[macro_id]
    x0 = new_cx - hw
    y0 = new_cy - hh
    x1 = new_cx + hw
    y1 = new_cy + hh
    r0, r1, c0, c1 = _bin_range_for_rect(x0, y0, x1, y1, bin_w, bin_h, rows, cols)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            n = int(counts[r, c])
            for k in range(n):
                j = int(grid[r, c, k])
                if j < 0 or j == macro_id:
                    continue
                if _rects_overlap(
                    new_cx,
                    new_cy,
                    hw,
                    hh,
                    pos_xy[j, 0],
                    pos_xy[j, 1],
                    half_w[j],
                    half_h[j],
                    gap,
                ):
                    return True
    return False


@njit(cache=True)
def delta_hpwl_for_macro_move(
    pos_xy: np.ndarray,
    net_ptr: np.ndarray,
    net_macros: np.ndarray,
    macro_net_ptr: np.ndarray,
    macro_nets: np.ndarray,
    macro_id: int,
    new_cx: float,
    new_cy: float,
) -> float:
    old_x = pos_xy[macro_id, 0]
    old_y = pos_xy[macro_id, 1]

    a = int(macro_net_ptr[macro_id])
    b = int(macro_net_ptr[macro_id + 1])
    if b - a <= 0:
        return 0.0

    before = 0.0
    for k in range(a, b):
        net_id = int(macro_nets[k])
        before += hpwl_for_net(pos_xy, net_ptr, net_macros, net_id)

    pos_xy[macro_id, 0] = new_cx
    pos_xy[macro_id, 1] = new_cy

    after = 0.0
    for k in range(a, b):
        net_id = int(macro_nets[k])
        after += hpwl_for_net(pos_xy, net_ptr, net_macros, net_id)

    pos_xy[macro_id, 0] = old_x
    pos_xy[macro_id, 1] = old_y
    return after - before

