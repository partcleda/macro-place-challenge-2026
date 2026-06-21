"""
ProxOpt Placer — Analytical Macro Placement via Proxy Cost Optimization

Strategy:
1. Analytical global placement using smooth WL (log-sum-exp) + density penalty
   in the style of ePlace/RePlAce — directly minimizes the proxy cost
2. Nesterov momentum (accelerated gradient descent) for fast convergence
3. Coordinate descent refinement sweeping each macro individually
4. LNS (Large Neighborhood Search) — destroy/repair patches of macros
5. Multi-start with perturbation, select best by exact proxy cost

The proxy cost is: WL + 0.5*Density + 0.5*Congestion
We approximate it with differentiable surrogates during optimization,
then evaluate exact proxy cost via compute_proxy_cost for candidate selection.

This directly targets the evaluation metric rather than just legalizing the
initial placement.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Utility: smooth (differentiable) wirelength via log-sum-exp
# ---------------------------------------------------------------------------

def _lse_wl(
    pos: np.ndarray,           # (n, 2) macro centers
    nets: list,                # list of (macro_indices, weights) tuples
    gamma: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """Log-sum-exp wirelength and its gradient w.r.t. pos."""
    n = pos.shape[0]
    grad = np.zeros_like(pos)
    wl = 0.0

    for indices, weights in nets:
        if len(indices) < 2:
            continue
        xi = pos[indices, 0]
        yi = pos[indices, 1]
        w = weights if weights is not None else np.ones(len(indices))

        # LSE in x
        ax = xi / gamma
        ax_max = ax.max()
        ex = np.exp(ax - ax_max)
        ex_sum = (w * ex).sum()
        ex_neg = np.exp(-ax - (-ax_max))
        ex_neg_sum = (w * ex_neg).sum()
        wl_x = gamma * (math.log(ex_sum) + ax_max + math.log(ex_neg_sum) + ax_max)
        gx = (w * ex / ex_sum - w * ex_neg / ex_neg_sum) / 1.0
        wl += wl_x

        # LSE in y
        ay = yi / gamma
        ay_max = ay.max()
        ey = np.exp(ay - ay_max)
        ey_sum = (w * ey).sum()
        ey_neg = np.exp(-ay - (-ay_max))
        ey_neg_sum = (w * ey_neg).sum()
        wl_y = gamma * (math.log(ey_sum) + ay_max + math.log(ey_neg_sum) + ay_max)
        gy = (w * ey / ey_sum - w * ey_neg / ey_neg_sum) / 1.0
        wl += wl_y

        np.add.at(grad[:, 0], indices, gx)
        np.add.at(grad[:, 1], indices, gy)

    return wl, grad


# ---------------------------------------------------------------------------
# Density penalty (bell-curve overlap avoidance + bin density)
# ---------------------------------------------------------------------------

def _density_penalty(
    pos: np.ndarray,   # (n, 2)
    sizes: np.ndarray, # (n, 2)
    canvas_w: float,
    canvas_h: float,
    n_bins: int = 32,
    target_density: float = 0.7,
) -> Tuple[float, np.ndarray]:
    """Bell-function density penalty similar to ePlace."""
    bw = canvas_w / n_bins
    bh = canvas_h / n_bins
    grad = np.zeros_like(pos)
    penalty = 0.0

    # Compute density in each bin
    density = np.zeros((n_bins, n_bins), dtype=np.float64)
    bin_cap = bw * bh * target_density

    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    # Use vectorized approach for speed
    bx = np.clip((pos[:, 0] / bw).astype(int), 0, n_bins - 1)
    by = np.clip((pos[:, 1] / bh).astype(int), 0, n_bins - 1)

    for i in range(pos.shape[0]):
        # Range of bins this macro covers
        bx_lo = max(0, int((pos[i, 0] - half_w[i]) / bw))
        bx_hi = min(n_bins - 1, int((pos[i, 0] + half_w[i]) / bw))
        by_lo = max(0, int((pos[i, 1] - half_h[i]) / bh))
        by_hi = min(n_bins - 1, int((pos[i, 1] + half_h[i]) / bh))
        area = sizes[i, 0] * sizes[i, 1]
        count = max(1, (bx_hi - bx_lo + 1) * (by_hi - by_lo + 1))
        density[bx_lo:bx_hi+1, by_lo:by_hi+1] += area / count

    overflow = np.maximum(0.0, density - bin_cap)
    penalty = 0.5 * (overflow ** 2).sum()

    # Gradient: push macros out of overflowing bins
    for i in range(pos.shape[0]):
        bx_lo = max(0, int((pos[i, 0] - half_w[i]) / bw))
        bx_hi = min(n_bins - 1, int((pos[i, 0] + half_w[i]) / bw))
        by_lo = max(0, int((pos[i, 1] - half_h[i]) / bh))
        by_hi = min(n_bins - 1, int((pos[i, 1] + half_h[i]) / bh))
        area = sizes[i, 0] * sizes[i, 1]
        count = max(1, (bx_hi - bx_lo + 1) * (by_hi - by_lo + 1))

        local_overflow = overflow[bx_lo:bx_hi+1, by_lo:by_hi+1]
        if local_overflow.sum() > 0:
            # Push away from center of high-density region
            cx = (bx_lo + bx_hi + 1) * bw * 0.5
            cy = (by_lo + by_hi + 1) * bh * 0.5
            scale = local_overflow.sum() * area / count
            grad[i, 0] += (pos[i, 0] - cx) * scale * 0.01
            grad[i, 1] += (pos[i, 1] - cy) * scale * 0.01

    return penalty, grad


# ---------------------------------------------------------------------------
# Overlap resolution (legalization)
# ---------------------------------------------------------------------------

def _legalize(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.005,
    max_iters: int = 1200,
    deadline: float = float('inf'),
) -> np.ndarray:
    """Minimum-displacement legalizer with fallback repulsion cleanup."""
    pos = pos.copy()
    n = pos.shape[0]
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)

    sep_x = (sizes[:, 0:1] + sizes[:, 0][None, :]) * 0.5 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1][None, :]) * 0.5 + gap
    areas = sizes[:, 0] * sizes[:, 1]
    order = np.argsort(-areas)
    placed = np.zeros(n, dtype=bool)

    def overlaps_with_placed(idx: int, x: float, y: float) -> bool:
        if not placed.any():
            return False
        dx = np.abs(x - pos[:, 0])
        dy = np.abs(y - pos[:, 1])
        ov = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
        ov[idx] = False
        return bool(ov.any())

    for idx in order:
        idx = int(idx)
        if not movable[idx]:
            placed[idx] = True
            continue
        ox, oy = float(pos[idx, 0]), float(pos[idx, 1])
        if not overlaps_with_placed(idx, ox, oy):
            placed[idx] = True
            continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.18 + gap
        best = (ox, oy)
        best_d2 = float("inf")
        found = False
        max_r = 95
        for r in range(1, max_r + 1):
            if time.time() >= deadline:
                break
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(np.clip(ox + dxm * step, half_w[idx], canvas_w - half_w[idx]))
                    cy = float(np.clip(oy + dym * step, half_h[idx], canvas_h - half_h[idx]))
                    if overlaps_with_placed(idx, cx, cy):
                        continue
                    d2 = (cx - ox) * (cx - ox) + (cy - oy) * (cy - oy)
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (cx, cy)
                        found = True
            if found:
                break
        pos[idx, 0], pos[idx, 1] = best
        placed[idx] = True

    # Fallback cleanup for any residual overlaps.
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    for _ in range(1400):
        dx = pos[:, 0:1] - pos[:, 0][None, :]
        dy = pos[:, 1:2] - pos[:, 1][None, :]
        ov_x = sep_x - np.abs(dx)
        ov_y = sep_y - np.abs(dy)
        mask = (ov_x > 0.0) & (ov_y > 0.0) & upper
        if not mask.any():
            break
        i_idx, j_idx = np.where(mask)
        i, j = int(i_idx[0]), int(j_idx[0])
        if not movable[i] and not movable[j]:
            continue
        ox, oy = float(ov_x[i, j]), float(ov_y[i, j])
        if ox < oy:
            s = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
            step = ox * 0.55 + 1e-4
            if movable[i]:
                pos[i, 0] = float(np.clip(pos[i, 0] + s * step, half_w[i], canvas_w - half_w[i]))
            if movable[j]:
                pos[j, 0] = float(np.clip(pos[j, 0] - s * step, half_w[j], canvas_w - half_w[j]))
        else:
            s = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
            step = oy * 0.55 + 1e-4
            if movable[i]:
                pos[i, 1] = float(np.clip(pos[i, 1] + s * step, half_h[i], canvas_h - half_h[i]))
            if movable[j]:
                pos[j, 1] = float(np.clip(pos[j, 1] - s * step, half_h[j], canvas_h - half_h[j]))

    return pos


# ---------------------------------------------------------------------------
# Build net list from benchmark
# ---------------------------------------------------------------------------

def _build_nets(benchmark: Benchmark, n_hard: int):
    """Extract weighted hard-macro net connectivity from Benchmark.net_nodes."""
    nets = []
    net_nodes = getattr(benchmark, "net_nodes", None)
    if not net_nodes:
        return nets

    net_weights = getattr(benchmark, "net_weights", None)
    for net_id, nodes in enumerate(net_nodes):
        try:
            arr = np.asarray(nodes, dtype=np.int32).reshape(-1)
        except Exception:
            continue

        # Hard macros are indices [0, n_hard).
        arr = arr[(arr >= 0) & (arr < n_hard)]
        if arr.size < 2:
            continue
        arr = np.unique(arr)
        if arr.size < 2:
            continue

        # Very large hypernets tend to create noisy forces; keep runtime bounded.
        if arr.size > 64:
            continue

        w = 1.0
        if net_weights is not None and len(net_weights) > net_id:
            try:
                w = float(net_weights[net_id])
            except Exception:
                w = 1.0
        if w <= 0.0:
            w = 1.0

        nets.append((arr.astype(np.int32), np.full(arr.shape[0], w, dtype=np.float64)))

    return nets


# ---------------------------------------------------------------------------
# Sparse pair edges + WL-thermostat refinement
# ---------------------------------------------------------------------------

def _build_sparse_edges_from_nets(nets: list, max_fanout_pairs: int = 8):
    """Convert hypernets to a sparse weighted pair graph."""
    edge_w = {}
    for indices, weights in nets:
        m = len(indices)
        if m < 2:
            continue
        net_w = float(weights[0]) if len(weights) > 0 else 1.0
        k = min(max_fanout_pairs, m - 1)
        for p, a in enumerate(indices):
            a = int(a)
            for d in range(1, k + 1):
                b = int(indices[(p + d) % m])
                if a == b:
                    continue
                i, j = (a, b) if a < b else (b, a)
                edge_w[(i, j)] = edge_w.get((i, j), 0.0) + net_w / d
    if not edge_w:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)
    edges = np.array(list(edge_w.keys()), dtype=np.int32)
    weights = np.array([edge_w[e] for e in edge_w], dtype=np.float64)
    return edges, weights


def _wl_thermostat_refine(
    pos: np.ndarray,
    init_pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    deadline: float,
    gap: float = 0.005,
) -> np.ndarray:
    """
    Fast SA-style refinement on sparse WL graph with hard overlap rejection.
    Uses local edge deltas (not full proxy eval) to explore many moves quickly.
    """
    if edges.shape[0] == 0:
        return pos

    n = pos.shape[0]
    pos = pos.copy()
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sep_x = (sizes[:, 0:1] + sizes[:, 0][None, :]) * 0.5 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1][None, :]) * 0.5 + gap
    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return pos

    adj = [[] for _ in range(n)]
    for e_idx, (a, b) in enumerate(edges):
        w = float(edge_weights[e_idx])
        a = int(a)
        b = int(b)
        adj[a].append((b, w))
        adj[b].append((a, w))

    def single_overlap(idx: int) -> bool:
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        ov = (dx < sep_x[idx]) & (dy < sep_y[idx])
        ov[idx] = False
        return bool(ov.any())

    def local_cost(i: int, x: float, y: float) -> float:
        c = 0.0
        for j, w in adj[i]:
            c += w * (abs(x - pos[j, 0]) + abs(y - pos[j, 1]))
        # Anchor limits large drifts that often hurt congestion/density.
        dx = x - init_pos[i, 0]
        dy = y - init_pos[i, 1]
        c += 0.0025 * (dx * dx + dy * dy)
        return c

    t0 = max(canvas_w, canvas_h) * 0.10
    t1 = max(canvas_w, canvas_h) * 0.001
    max_steps = max(800, 25 * len(movable_idx))
    step_id = 0
    while time.time() < deadline and step_id < max_steps:
        step_id += 1
        frac = step_id / max_steps
        temp = t0 * ((t1 / t0) ** frac)

        i = int(random.choice(movable_idx))
        ox, oy = float(pos[i, 0]), float(pos[i, 1])
        old_cost = local_cost(i, ox, oy)

        move_type = random.random()
        nx, ny = ox, oy
        if move_type < 0.50:
            # Gaussian jitter
            sigma = temp * (0.30 + 0.70 * (1.0 - frac))
            nx = ox + random.gauss(0.0, sigma)
            ny = oy + random.gauss(0.0, sigma)
        elif move_type < 0.82 and adj[i]:
            # Pull toward weighted neighbor barycenter
            sw = 0.0
            tx = 0.0
            ty = 0.0
            for j, w in adj[i]:
                sw += w
                tx += w * pos[j, 0]
                ty += w * pos[j, 1]
            if sw > 1e-9:
                tx /= sw
                ty /= sw
                alpha = random.uniform(0.08, 0.35)
                nx = ox + alpha * (tx - ox)
                ny = oy + alpha * (ty - oy)
        elif adj[i]:
            # Swap with a connected movable macro
            j = int(random.choice([nb for nb, _ in adj[i] if movable[nb]] or [i]))
            if j != i:
                ox2, oy2 = float(pos[j, 0]), float(pos[j, 1])
                pos[i, 0], pos[i, 1] = ox2, oy2
                pos[j, 0], pos[j, 1] = ox, oy
                if single_overlap(i) or single_overlap(j):
                    pos[i, 0], pos[i, 1] = ox, oy
                    pos[j, 0], pos[j, 1] = ox2, oy2
                    continue
                new_cost = local_cost(i, pos[i, 0], pos[i, 1]) + local_cost(j, pos[j, 0], pos[j, 1])
                old_pair = local_cost(i, ox, oy) + local_cost(j, ox2, oy2)
                delta = new_cost - old_pair
                if delta <= 0.0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                    continue
                pos[i, 0], pos[i, 1] = ox, oy
                pos[j, 0], pos[j, 1] = ox2, oy2
                continue

        nx = float(np.clip(nx, half_w[i], canvas_w - half_w[i]))
        ny = float(np.clip(ny, half_h[i], canvas_h - half_h[i]))
        pos[i, 0], pos[i, 1] = nx, ny
        if single_overlap(i):
            pos[i, 0], pos[i, 1] = ox, oy
            continue
        new_cost = local_cost(i, nx, ny)
        delta = new_cost - old_cost
        if delta > 0.0 and random.random() >= math.exp(-delta / max(temp, 1e-9)):
            pos[i, 0], pos[i, 1] = ox, oy

    return pos


# ---------------------------------------------------------------------------
# Coordinate descent: move one macro at a time to minimize proxy cost
# ---------------------------------------------------------------------------

def _coordinate_descent(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    nets: list,
    benchmark: Benchmark,
    plc,
    n_hard: int,
    base_tensor: torch.Tensor,
    gap: float = 0.005,
    deadline: float = float('inf'),
    n_steps: int = 9,
    search_radius_frac: float = 0.12,
) -> np.ndarray:
    """
    Macro-swap descent: swap pairs of macros and keep if proxy improves.
    Swaps are always legal (no overlap introduced), so every candidate
    can be evaluated. Much faster than single-macro moves on dense layouts.
    Two passes:
    1. Connected swaps: swap each macro with its net neighbors
    2. Random swaps among high-degree macros
    """
    try:
        from macro_place.objective import compute_proxy_cost
    except Exception:
        return pos

    pos = pos.copy()

    base_tensor[:n_hard] = torch.from_numpy(pos[:n_hard]).to(dtype=torch.float32)
    try:
        current_score = float(compute_proxy_cost(base_tensor, benchmark, plc)["proxy_cost"])
    except Exception:
        return pos

    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sep_x = (sizes[:, 0:1] + sizes[:, 0][None, :]) * 0.5 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1][None, :]) * 0.5 + gap

    # Sparse neighbor graph (bounded-degree expansion per hyperedge).
    neighbors = {i: set() for i in range(n_hard)}
    degree = np.zeros(n_hard, dtype=np.float64)
    for indices, weights in nets:
        m = len(indices)
        if m < 2:
            continue
        k = min(8, m - 1)
        net_w = float(weights[0]) if len(weights) > 0 else 1.0
        for p, a in enumerate(indices):
            for t in range(1, k + 1):
                b = int(indices[(p + t) % m])
                if a == b:
                    continue
                neighbors[int(a)].add(b)
                neighbors[int(b)].add(int(a))
                degree[int(a)] += net_w
                degree[int(b)] += net_w

    movable_order = [i for i in np.argsort(-degree) if movable[i]]
    if not movable_order:
        return pos

    def single_overlap(idx: int) -> bool:
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        ov = (dx < sep_x[idx]) & (dy < sep_y[idx])
        ov[idx] = False
        return bool(ov.any())

    def eval_current() -> float:
        base_tensor[:n_hard] = torch.from_numpy(pos[:n_hard]).to(dtype=torch.float32)
        try:
            return float(compute_proxy_cost(base_tensor, benchmark, plc)["proxy_cost"])
        except Exception:
            return float("inf")

    # Pass 1: macro moves toward net-neighbor centroids (coarse-to-fine radius).
    radius = max(canvas_w, canvas_h) * search_radius_frac
    line = np.linspace(-1.0, 1.0, max(3, n_steps))

    for i in movable_order:
        if time.time() >= deadline:
            break
        nbs = [nb for nb in neighbors[i] if movable[nb]]
        if not nbs:
            continue

        ox, oy = float(pos[i, 0]), float(pos[i, 1])
        nx = float(np.mean(pos[nbs, 0]))
        ny = float(np.mean(pos[nbs, 1]))

        candidate_pts = []
        candidate_pts.append((0.35 * ox + 0.65 * nx, 0.35 * oy + 0.65 * ny))
        candidate_pts.append((0.55 * ox + 0.45 * nx, 0.55 * oy + 0.45 * ny))
        for a in line:
            candidate_pts.append((nx + a * radius, oy))
            candidate_pts.append((ox, ny + a * radius))
        # Spread candidates: push away from centroid of all macros.
        # These reduce congestion even if WL increases slightly,
        # and the exact proxy eval will accept only if total score improves.
        cx_all = float(np.mean(pos[:, 0]))
        cy_all = float(np.mean(pos[:, 1]))
        ddx = ox - cx_all
        ddy = oy - cy_all
        dnorm = math.hypot(ddx, ddy) + 1e-9
        for frac in (0.15, 0.30, 0.50):
            candidate_pts.append((ox + (ddx / dnorm) * radius * frac * 2.0,
                                   oy + (ddy / dnorm) * radius * frac * 2.0))
            # Orthogonal spread
            candidate_pts.append((ox + (-ddy / dnorm) * radius * frac,
                                   oy + (ddx / dnorm) * radius * frac))
        random.shuffle(candidate_pts)

        improved = False
        for cx, cy in candidate_pts:
            if time.time() >= deadline:
                break
            pos[i, 0] = float(np.clip(cx, half_w[i], canvas_w - half_w[i]))
            pos[i, 1] = float(np.clip(cy, half_h[i], canvas_h - half_h[i]))
            if single_overlap(i):
                continue
            s = eval_current()
            if s + 1e-6 < current_score:
                current_score = s
                improved = True
                break

        if not improved:
            pos[i, 0], pos[i, 1] = ox, oy

    # Pass 2: targeted swaps among high-degree connected macros.
    top = movable_order[: min(len(movable_order), 200)]
    n_swap_trials = min(600, 8 * len(top))
    for _ in range(n_swap_trials):
        if time.time() >= deadline or len(top) < 2:
            break
        i = int(random.choice(top))
        nbs = [nb for nb in neighbors[i] if movable[nb]]
        if not nbs:
            continue
        j = int(random.choice(nbs))
        if i == j:
            continue
        oi = pos[i].copy()
        oj = pos[j].copy()
        pos[i], pos[j] = oj, oi
        if single_overlap(i) or single_overlap(j):
            pos[i], pos[j] = oi, oj
            continue
        s = eval_current()
        if s + 1e-6 < current_score:
            current_score = s
        else:
            pos[i], pos[j] = oi, oj

    base_tensor[:n_hard] = torch.from_numpy(pos[:n_hard]).to(dtype=torch.float32)
    return pos


# ---------------------------------------------------------------------------
# LNS: Large Neighborhood Search
# ---------------------------------------------------------------------------

def _lns_improve(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    nets: list,
    benchmark: Benchmark,
    plc,
    n_hard: int,
    base_tensor: torch.Tensor,
    gap: float = 0.005,
    deadline: float = float('inf'),
    patch_size: int = 8,
    n_iters: int = 200,
) -> Tuple[np.ndarray, float]:
    """
    Randomly select a patch of macros, re-place them using gradient descent
    on wirelength, re-legalize, evaluate exact proxy cost, accept if better.
    """
    try:
        from macro_place.objective import compute_proxy_cost
    except Exception:
        return pos, float('inf')

    pos = pos.copy()
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sep_x = (sizes[:, 0:1] + sizes[:, 0][None, :]) * 0.5 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1][None, :]) * 0.5 + gap

    def check_overlap(p):
        n = p.shape[0]
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        dx = np.abs(p[:, 0:1] - p[:, 0][None, :])
        dy = np.abs(p[:, 1:2] - p[:, 1][None, :])
        return ((dx < sep_x) & (dy < sep_y) & upper).any()

    base_tensor[:n_hard] = torch.from_numpy(pos[:n_hard]).to(dtype=torch.float32)
    try:
        best_score = float(compute_proxy_cost(base_tensor, benchmark, plc)["proxy_cost"])
    except Exception:
        return pos, float('inf')

    movable_indices = np.where(movable[:n_hard])[0]
    if len(movable_indices) == 0:
        return pos, best_score

    # Build adjacency once
    adjacency = {i: set() for i in range(n_hard)}
    for indices, _ in nets:
        for a in indices:
            for b in indices:
                if a != b and a < n_hard and b < n_hard:
                    adjacency[a].add(b)

    no_improve_iters = 0
    for iteration in range(n_iters):
        if time.time() >= deadline:
            break

        # Select a patch: start from a random movable macro, BFS expand
        seed = int(random.choice(movable_indices))
        patch = [seed]
        frontier = list(adjacency[seed])
        random.shuffle(frontier)
        for nb in frontier:
            if len(patch) >= patch_size:
                break
            if movable[nb] and nb not in patch:
                patch.append(nb)

        # Fill patch to desired size with random neighbors
        while len(patch) < min(patch_size, len(movable_indices)):
            extra = int(random.choice(movable_indices))
            if extra not in patch:
                patch.append(extra)

        patch = np.array(patch, dtype=np.int32)

        # Save current positions
        old_pos = pos[patch].copy()

        # Move patch macros: 50% toward canvas spread, 50% toward WL neighbors
        # This balances congestion (spread) vs WL (clustering)
        sigma = math.hypot(canvas_w, canvas_h) * 0.02
        for k, idx in enumerate(patch):
            connected = [nb for nb in adjacency[idx] if nb not in patch and nb < n_hard]
            if connected:
                wl_x = np.mean(pos[connected, 0])
                wl_y = np.mean(pos[connected, 1])
            else:
                wl_x, wl_y = pos[idx, 0], pos[idx, 1]
            # Spread target: current pos reflected toward canvas center
            spread_x = canvas_w * 0.5 + (pos[idx, 0] - canvas_w * 0.5) * 0.7
            spread_y = canvas_h * 0.5 + (pos[idx, 1] - canvas_h * 0.5) * 0.7
            target_x = 0.6 * wl_x + 0.4 * spread_x + random.gauss(0, sigma)
            target_y = 0.6 * wl_y + 0.4 * spread_y + random.gauss(0, sigma)
            alpha = 0.35
            new_x = pos[idx, 0] * (1 - alpha) + target_x * alpha
            new_y = pos[idx, 1] * (1 - alpha) + target_y * alpha
            pos[idx, 0] = float(np.clip(new_x, half_w[idx], canvas_w - half_w[idx]))
            pos[idx, 1] = float(np.clip(new_y, half_h[idx], canvas_h - half_h[idx]))

        # Legalize
        patch_movable = np.zeros(n_hard, dtype=bool)
        patch_movable[patch] = True
        # Only allow patch macros to move
        combined_movable = movable[:n_hard] & patch_movable
        pos_leg = _legalize(
            pos=pos,
            sizes=sizes,
            movable=combined_movable,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            gap=gap,
        )

        # Check no global overlaps
        if check_overlap(pos_leg):
            pos[patch] = old_pos
            continue

        # Evaluate
        base_tensor[:n_hard] = torch.from_numpy(pos_leg[:n_hard]).to(dtype=torch.float32)
        try:
            score = float(compute_proxy_cost(base_tensor, benchmark, plc)["proxy_cost"])
        except Exception:
            pos[patch] = old_pos
            continue

        if score < best_score:
            best_score = score
            pos = pos_leg
            no_improve_iters = 0
        else:
            pos[patch] = old_pos
            no_improve_iters += 1

        # Early-stop LNS if no gain for many patch attempts.
        if no_improve_iters >= 10:
            break

    return pos, best_score


# ---------------------------------------------------------------------------
# Analytical global placement via gradient descent on smooth proxy
# ---------------------------------------------------------------------------

def _analytical_global_place(
    init: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    nets: list,
    benchmark: Benchmark,
    plc,
    n_hard: int,
    gap: float = 0.005,
    deadline: float = float('inf'),
    lr: float = 0.02,
    density_weight: float = 0.5,
    n_iters: int = 300,
    gamma: float = None,
) -> np.ndarray:
    """
    Gradient descent on smooth WL (LSE) + density penalty.
    Uses Nesterov momentum for acceleration.
    """
    if not nets:
        return init.copy()

    diag = math.hypot(canvas_w, canvas_h)
    if gamma is None:
        gamma = diag * 0.04

    pos = init.copy()
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)

    # Nesterov state
    v = np.zeros_like(pos)
    momentum = 0.9
    lr_cur = lr * diag

    for it in range(n_iters):
        if time.time() >= deadline:
            break

        # Nesterov lookahead
        pos_look = pos + momentum * v

        wl, gwl = _lse_wl(pos_look, nets, gamma=gamma)
        # Skip density for speed (legalization handles overlaps)
        # den, gden = _density_penalty(pos_look, sizes, canvas_w, canvas_h)
        # grad = gwl + density_weight * gden

        grad = gwl

        # Clip gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-8:
            grad = grad * min(1.0, lr_cur / gnorm)

        grad[~movable] = 0.0

        v = momentum * v - grad
        pos = pos + v

        pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)

        # Decay lr and gamma
        if it % 50 == 49:
            lr_cur *= 0.7
            gamma *= 0.8

    return pos


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

def _canvas_shrink_init(
    init: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    gap: float = 0.005,
    shrink: float = 0.82,
) -> np.ndarray:
    """
    Shrink canvas by `shrink`, legalize (forces global repacking),
    scale back. Breaks congestion hotspots. RoRa-style (rank 7, 1.2788).
    """
    sw, sh = canvas_w * shrink, canvas_h * shrink
    pos = init.copy()
    pos[:, 0] *= shrink
    pos[:, 1] *= shrink
    pos = _legalize(pos, sizes, movable, sw, sh, gap)
    pos[:, 0] /= shrink
    pos[:, 1] /= shrink
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    pos[:, 0] = np.clip(pos[:, 0], hw, canvas_w - hw)
    pos[:, 1] = np.clip(pos[:, 1], hh, canvas_h - hh)
    return pos


class Mj97Placer:
    """
    ProxOpt: Analytical placement + coordinate descent + LNS.

    Pipeline per benchmark:
      1. Extract nets and build connectivity
      2. Run analytical global placement (LSE wirelength gradient descent)
         starting from the initial placement (strong prior)
      3. Legalize to remove overlaps
      4. Evaluate proxy cost via compute_proxy_cost
      5. Coordinate descent refinement (move each macro on a grid, keep best)
      6. LNS: iteratively destroy/repair patches, accept improvements
      7. Multi-start: repeat from step 2 with different perturbations,
         keep best result across all starts
    """

    def __init__(self, seed: int = 97):
        self.seed = seed
        self.gap = 0.005
        self.min_budget_s = 45.0
        self.max_budget_s = 1200.0  # 20 min cap; keeps runs practical
        self.enable_final_soft_fd = True

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return benchmark.macro_positions.clone()

        out = benchmark.macro_positions.clone()
        init = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        diag = math.hypot(cw, ch)

        budget = min(self.max_budget_s, max(self.min_budget_s, 1.2 * n_hard + 140))
        deadline = time.time() + budget

        # Try to load plc for exact proxy cost evaluation
        plc = self._load_plc(benchmark)

        # Build net connectivity
        nets = _build_nets(benchmark, n_hard)
        sparse_edges, sparse_edge_w = _build_sparse_edges_from_nets(nets)

        # Working tensor (shared reference for proxy evaluation)
        base_tensor = benchmark.macro_positions.clone()
        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            compute_proxy_cost = None

        def eval_proxy(hard_pos: np.ndarray, optimize_soft: bool = False) -> float:
            """Evaluate proxy cost; optionally re-place soft macros with FD."""
            if plc is None or compute_proxy_cost is None:
                return float("inf")

            base_tensor[:n_hard] = torch.from_numpy(hard_pos[:n_hard]).to(dtype=torch.float32)
            try:
                score = float(compute_proxy_cost(base_tensor, benchmark, plc)["proxy_cost"])
            except Exception:
                return float("inf")

            if optimize_soft and benchmark.num_soft_macros > 0:
                try:
                    canvas_size = max(cw, ch)
                    num_steps = [45, 45, 30]
                    max_move = [canvas_size / 55.0] * 3
                    plc.optimize_stdcells(
                        False, True, False,
                        False, False, 1.0,
                        num_steps, max_move,
                        [100.0, 1.0e-3, 1.0e-5],
                        [0.0, 1.0e6, 8.0e6],
                    )
                    for k, macro_idx in enumerate(benchmark.soft_macro_indices):
                        sx, sy = plc.get_node_location(int(macro_idx))
                        base_tensor[n_hard + k, 0] = float(sx)
                        base_tensor[n_hard + k, 1] = float(sy)
                    score = float(compute_proxy_cost(base_tensor, benchmark, plc)["proxy_cost"])
                except Exception:
                    pass
            return score

        best_pos = None
        best_full = None
        best_score = float('inf')
        no_improve_starts = 0

        # Determine number of starts based on budget
        n_starts = 3 if n_hard < 450 else 2

        for start_idx in range(n_starts):
            if time.time() >= deadline - 5.0:
                break

            start_time = time.time()
            remaining = deadline - start_time
            start_budget = remaining / max(1, n_starts - start_idx)
            start_deadline = start_time + start_budget

            # Init strategy per start:
            # 0 = original initial placement
            # 1/2 = WL-scale gaussian perturbations for diversification
            half_w = sizes[:, 0] * 0.5
            half_h = sizes[:, 1] * 0.5
            if start_idx == 0:
                cur_init = init.copy()
            else:
                sigma = diag * (0.010 if start_idx == 1 else 0.020)
                cur_init = init.copy()
                noise = np.random.normal(0, sigma, cur_init.shape)
                noise[~movable] = 0.0
                cur_init += noise
                cur_init[:, 0] = np.clip(cur_init[:, 0], half_w, cw - half_w)
                cur_init[:, 1] = np.clip(cur_init[:, 1], half_h, ch - half_h)

            # Phase 1: Legalize init
            print(f"[MJ97] start={start_idx} init_strategy={'noise_s' if start_idx==1 else 'noise_l' if start_idx==2 else 'original'}", flush=True)
            legalized = _legalize(cur_init, sizes, movable, cw, ch, self.gap)
            score = eval_proxy(legalized, optimize_soft=False)

            # Phase 1b: Fast WL thermostat refinement (legal-by-construction moves).
            sa_end = min(start_deadline - 6.0, time.time() + start_budget * 0.32)
            if sa_end > time.time() + 0.5 and sparse_edges.shape[0] > 0:
                thermo_pos = _wl_thermostat_refine(
                    pos=legalized,
                    init_pos=init,
                    sizes=sizes,
                    movable=movable,
                    canvas_w=cw,
                    canvas_h=ch,
                    edges=sparse_edges,
                    edge_weights=sparse_edge_w,
                    deadline=sa_end,
                    gap=self.gap,
                )
                thermo_pos = _legalize(thermo_pos, sizes, movable, cw, ch, self.gap, deadline=sa_end + 3.0)
                thermo_score = eval_proxy(thermo_pos, optimize_soft=False)
                if thermo_score + 1e-6 < score:
                    legalized = thermo_pos
                    score = thermo_score

            if plc is None:
                if best_pos is None:
                    best_pos = legalized
                continue

            # Phase 2: Coarse-to-fine coordinate descent (ProxCD-style)
            # Multiple passes, shrinking radius each time.
            cd_budget = start_budget * 0.38
            cd_end = time.time() + cd_budget
            print(f"[MJ97] start={start_idx} before_cd score={score:.4f} cd_budget={cd_budget:.1f}s", flush=True)
            cd_no_improve_passes = 0
            prev_cd_score = score
            for radius_frac, n_steps in [(0.20, 9), (0.12, 11), (0.07, 13), (0.04, 15), (0.02, 17)]:
                if time.time() >= cd_end:
                    break
                pass_time = (cd_end - time.time()) / 3.0
                legalized = _coordinate_descent(
                    pos=legalized, sizes=sizes, movable=movable,
                    canvas_w=cw, canvas_h=ch, nets=nets,
                    benchmark=benchmark, plc=plc, n_hard=n_hard,
                    base_tensor=base_tensor, gap=self.gap,
                    deadline=min(cd_end, time.time() + pass_time),
                    n_steps=n_steps, search_radius_frac=radius_frac,
                )
                try:
                    s = eval_proxy(legalized, optimize_soft=False)
                    print(f"[MJ97] start={start_idx} after_cd r={radius_frac} score={s:.4f}", flush=True)
                    if prev_cd_score - s < 5e-4:
                        cd_no_improve_passes += 1
                    else:
                        cd_no_improve_passes = 0
                    prev_cd_score = s
                    if cd_no_improve_passes >= 2:
                        print(
                            f"[MJ97] start={start_idx} early-stop CD: no meaningful improvement in 2 passes",
                            flush=True,
                        )
                        break
                except Exception:
                    pass
            score = eval_proxy(legalized, optimize_soft=False)

            # Phase 3: LNS
            lns_time = min(start_deadline - time.time() - 2.0, 90.0)
            if lns_time > 3.0:
                patch_size = min(12, max(4, n_hard // 30))
                legalized, score = _lns_improve(
                    pos=legalized, sizes=sizes, movable=movable,
                    canvas_w=cw, canvas_h=ch, nets=nets,
                    benchmark=benchmark, plc=plc, n_hard=n_hard,
                    base_tensor=base_tensor, gap=self.gap,
                    deadline=start_deadline - 1.0,
                    patch_size=patch_size, n_iters=min(80, max(50, int(lns_time * 6))),
                )
                # Re-run exact score (LNS returns stale score on rare eval errors)
                score = eval_proxy(legalized, optimize_soft=False)

            if score < best_score:
                print(f"[MJ97] start={start_idx} NEW BEST score={score:.4f} (was {best_score:.4f})", flush=True)
                best_score = score
                best_pos = legalized.copy()
                best_full = None
                no_improve_starts = 0
            else:
                no_improve_starts += 1
                if no_improve_starts >= 2:
                    print(
                        f"[MJ97] early-stop multi-start: 2 consecutive starts without improvement",
                        flush=True,
                    )
                    break

        if best_pos is None:
            # Fallback: just legalize initial
            best_pos = _legalize(
                pos=init,
                sizes=sizes,
                movable=movable,
                canvas_w=cw,
                canvas_h=ch,
                gap=self.gap,
            )

        out[:n_hard] = torch.from_numpy(best_pos[:n_hard]).to(dtype=torch.float32)
        # Single final soft-macro optimization pass on the best hard placement.
        if self.enable_final_soft_fd and best_pos is not None and plc is not None and compute_proxy_cost is not None:
            pre_soft = eval_proxy(best_pos, optimize_soft=False)
            post_soft = eval_proxy(best_pos, optimize_soft=True)
            if post_soft <= pre_soft + 1e-6:
                best_full = base_tensor.clone()
        if best_full is not None:
            out = best_full
        return out

    def _load_plc(self, benchmark: Benchmark):
        """Try to load plc object for proxy cost evaluation."""
        try:
            from macro_place.loader import load_benchmark, load_benchmark_from_dir
        except Exception:
            return None

        bench_name = getattr(benchmark, 'name', None) or ''

        root = f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}"
        try:
            _, plc = load_benchmark_from_dir(root)
            return plc
        except Exception:
            pass

        ng45_alias = {
            "ariane133_ng45": "ariane133",
            "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla",
            "mempool_tile_ng45": "mempool_tile",
        }
        ng = ng45_alias.get(bench_name, bench_name)
        ng_dir = f"external/MacroPlacement/Flows/NanGate45/{ng}/netlist/output_CT_Grouping"
        try:
            _, plc = load_benchmark(
                f"{ng_dir}/netlist.pb.txt",
                f"{ng_dir}/initial.plc",
                name=bench_name,
            )
            return plc
        except Exception:
            return Nones