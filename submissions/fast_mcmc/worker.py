"""
fast_mcmc.worker
================

Core Metropolis-Hastings simulated-annealing optimiser for the MCMC
macro placer.

This module is the entire "computational engine" called out in
``cursor.md`` §D.  A worker process owns one cloned :class:`fast_mcmc.
state.PlacementState`, a private ``density_grid`` float matrix, a few
fixed-size scratch buffers, and a seeded :class:`numpy.random.Generator`.
It then executes a single, long-running optimisation loop:

1. **GRASP construction** (delegated to :func:`fast_mcmc.initialization.
   grasp_initialize`) yields a structurally diverse starting layout and
   stamps every hard macro onto ``spatial_grid``.

2. **Temperature warmup** runs a few hundred trial mutations to estimate
   :math:`T_0 = -\\overline{|\\Delta C|} / \\ln(p_\\text{accept})`, so the
   geometric cooling schedule starts at a physically meaningful scale.

3. **Three-phase MCMC loop** drives the Boltzmann acceptance test
   :math:`P = e^{-\\Delta C / T}` over geometrically cooled temperatures
   :math:`T_{k+1} = \\alpha T_k`.  Each iteration:

   * picks a **mutation type** – *shift*, *swap*, *reshape* – with a
     temperature-dependent mix (more reshapes when hot, almost only
     shifts when cold);
   * picks the **target macro(s)** through a *dynamic selection-weight
     array* that biases towards "worst offenders" (large net length,
     active grid collisions);
   * applies the **multi-tier overlap filter** (cursor.md §F):

     - Phase-2 **hard canvas filter** – any proposal that pushes the
       macro outside the canvas perimeter is rejected before any cost
       maths is run;
     - Phase-1 **soft overlap penalty** – grid collisions are *allowed*
       but cost a penalty term ``Δ_overlap_cells · k(T)`` whose
       coefficient grows exponentially as ``T`` drops, so the engine
       tolerates overlaps while exploring and pushes them out as the
       temperature falls;
     - Phase-3 **strict legalisation** – once the temperature falls
       below ``T_legalization`` the kernel additionally rejects any
       move that would *increase* the foreign-cell count in the moving
       macro's bbox, driving the layout to zero overlap.

   * calls the Numba delta kernels in :mod:`fast_mcmc.fast_eval` to get
     exact ``(Δ_wl, Δ_density, Δ_congestion)`` and commits them
     atomically on acceptance.

4. **Periodic refreshes** rebuild ``spatial_grid`` from scratch every
   ``grid_refresh_iters`` steps (so the last-writer-wins integer matrix
   converges back to the ground truth) and recompute selection weights
   from current collision / wirelength statistics.

5. **Final validity sweep** (cursor.md §F.4) runs three independent
   checks – vectorised canvas containment, ``count_grid_collisions_njit``
   over the rebuilt spatial grid, and an exact pairwise hard-macro bbox
   intersection – and the worker marks itself ``valid=False`` if *any*
   of them flags a violation, forcing :mod:`fast_mcmc.main` to discard
   the run.

All randomness flows through ``numpy.random.default_rng(seed)`` so two
workers with the same seed produce identical layouts; different seeds
yield diverse exploration trajectories (this is the parallel pool's
basic premise).

The module is import-safe with or without Numba: the kernels in
:mod:`fast_mcmc.fast_eval` already provide a graceful fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import fast_eval as fe
from initialization import GraspReport, grasp_initialize
from state import (
    EMPTY_CELL,
    EPS,
    PlacementState,
    clone_state,
    stamp_all_hard_macros,
)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 0. Hyperparameter container                                            ║
# ╚════════════════════════════════════════════════════════════════════════╝

@dataclass
class WorkerConfig:
    """Single source of truth for every tunable knob of the worker loop.

    Defaults are calibrated for ibm-scale benchmarks (≈ 250 – 1500 hard
    macros, 600 – 12 000 nets) on a single CPU core; ``main.py`` is free
    to override any field per benchmark or per worker.
    """

    # ── Top-level control ───────────────────────────────────────────────
    seed: int = 0
    time_budget_seconds: float = 60.0
    max_iterations: int = 200_000_000  # hard safety cap

    # ── Cost weights (cursor.md §1 objective function) ──────────────────
    weight_wirelength: float = 1.0
    weight_density:    float = 0.5
    weight_congestion: float = 0.5

    # ── Annealing schedule ──────────────────────────────────────────────
    warmup_iters:             int   = 500
    warmup_target_accept:     float = 0.85   # ⇒ T0 = -|ΔC|̄ / ln(p)
    final_temp_factor:        float = 1e-4   # T_end = T0 * this
    cooling_recalibrate_iters: int  = 10_000 # recompute α from live iter/s

    # ── Multi-tier overlap thresholds ───────────────────────────────────
    legalization_temp_factor: float = 5e-3   # T < T0*this ⇒ Phase 3
    # The soft per-cell overlap penalty must stay small compared to the
    # density / congestion cost (which is the *exact* overlap signal)
    # while the temperature is hot, otherwise the MCMC chases the lossy
    # spatial-grid counter and macros collapse on top of each other.
    # By T → 0 the coefficient has grown ~e⁸ ≈ 3000× so a one-cell
    # overlap reduction reliably overrides a moderate density / HPWL
    # increase – this is what carries the final legalisation pass.
    overlap_penalty_base:     float = 0.05
    overlap_penalty_grow_k:   float = 8.0    # penalty(T) = base·exp(k·(1−T/T₀))

    # ── Mutation mix (probabilities for hot vs cold phases) ─────────────
    p_shift_hot:    float = 0.65
    p_swap_hot:     float = 0.25
    p_reshape_hot:  float = 0.10
    p_shift_cold:   float = 0.92
    p_swap_cold:    float = 0.08
    p_reshape_cold: float = 0.00

    # ── Shift mutation ──────────────────────────────────────────────────
    shift_sigma_init_frac:  float = 0.20  # σ_init = canvas_size × this
    shift_sigma_final_frac: float = 0.005 # σ_end  = canvas_size × this

    # ── Swap mutation ───────────────────────────────────────────────────
    swap_area_tolerance: float = 0.40   # |area(a)−area(b)| / max(area) ≤ this
    swap_max_partner_tries: int = 16

    # ── Reshape mutation (soft macros only) ─────────────────────────────
    reshape_log_aspect_sigma: float = 0.6  # log-normal σ for aspect-ratio
    reshape_max_aspect: float = 4.0        # clamp aspect to [1/max, max]

    # ── Biased selector ─────────────────────────────────────────────────
    selector_refresh_iters:    int   = 2_000
    selector_base_weight:      float = 1.0
    selector_overlap_weight:   float = 4.0
    selector_netlen_weight:    float = 1.0e-4  # canvas-units of HPWL

    # ── Spatial-grid maintenance ────────────────────────────────────────
    # The integer spatial grid is *last-writer-wins*, so it drifts during
    # the run as macros overstamp each other.  A frequent re-stamp keeps
    # the overlap-penalty signal and Phase-3 legalisation honest.
    grid_refresh_iters: int = 2_000

    # ── Plumbing ────────────────────────────────────────────────────────
    enable_grasp: bool = True
    grasp_num_candidates: int = 16
    grasp_top_k: int = 4
    enable_final_validity_check: bool = True
    log_progress_every_iters: int = 0     # 0 ⇒ silent

    # ── Validity check tuning ───────────────────────────────────────────
    pairwise_check_cap: int = 8_000  # skip O(N²) exact check above this

    # ── Post-MCMC greedy legalisation sweep ─────────────────────────────
    # Deterministic safety-net run *after* the Metropolis loop ends and
    # *before* the final validity check.  The MCMC's bin-based overlap
    # signal is last-writer-wins and can miss the last few residual
    # geometric overlaps; the sweep uses exact bbox arithmetic to nudge
    # each colliding macro to the nearest collision-free position via an
    # 8-direction spiral search.  This is what *guarantees* all-zero
    # hard-macro overlap at termination on tough benchmarks (ibm02 etc.).
    enable_legalization_sweep: bool = True
    legalization_sweep_max_passes:  int = 50
    legalization_sweep_max_radius:  int = 256   # spiral steps in each direction
    legalization_sweep_step_scale: float = 0.5  # step = scale × min(hard dim)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 1. Result container                                                    ║
# ╚════════════════════════════════════════════════════════════════════════╝

@dataclass
class WorkerResult:
    """Pickleable bundle returned to ``main.py`` by every worker.

    The final layout is carried by ``state.macro_coords`` /
    ``state.macro_dims`` (bottom-left convention; convert via
    :func:`fast_mcmc.state.bottom_left_to_centers` before handing to the
    evaluator).  ``valid`` is ``True`` iff the final-validity sweep
    found zero canvas violations *and* zero hard-macro overlaps.
    """

    seed: int
    state: PlacementState
    grasp_report: GraspReport

    valid: bool
    violations: List[str] = field(default_factory=list)

    cost_wirelength:      float = 0.0
    cost_density:         float = 0.0
    cost_congestion:      float = 0.0
    cost_proxy:           float = 0.0  # weighted sum (matches cursor.md §1)
    cost_overlap_penalty: float = 0.0  # final penalty contribution
    initial_cost_proxy:   float = 0.0

    initial_temperature: float = 0.0
    final_temperature:   float = 0.0
    cooling_alpha:       float = 0.0

    iterations: int = 0
    elapsed_seconds: float = 0.0
    accepts_total: int = 0
    rejects_total: int = 0

    accepts_shift:   int = 0; rejects_shift:   int = 0
    accepts_swap:    int = 0; rejects_swap:    int = 0
    accepts_reshape: int = 0; rejects_reshape: int = 0
    rejects_canvas:    int = 0   # Phase-2 hard rejections
    rejects_legalization: int = 0  # Phase-3 strict rejections

    # ── Post-MCMC greedy legalisation sweep diagnostics ─────────────────
    legalization_sweep_initial_overlaps: int = 0
    legalization_sweep_final_overlaps:   int = 0
    legalization_sweep_moves:            int = 0
    legalization_sweep_passes:           int = 0


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 2. Numba helper kernels (worker-local)                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝
#
# These kernels live here rather than in ``fast_eval.py`` because they
# are higher-level helpers wired specifically to the worker's bookkeeping
# (selector weights, overlap penalty, validity sweep).  They are still
# pure functions over flat numpy arrays so Numba compiles them just fine.

njit = fe.njit  # reuse the decorator shim


@njit(cache=True)
def count_foreign_cells_in_bbox_njit(
    spatial_grid: np.ndarray,
    macro_idx: int,
    x_ll: float, y_ll: float, width: float, height: float,
    bin_width: float, bin_height: float,
) -> int:
    """Number of cells in macro ``macro_idx``'s candidate bbox that
    currently hold a *different*, non-empty macro id.

    This is the per-move "collision count" used both by the Phase-1
    overlap-penalty math and the Phase-3 legalisation rejection.

    The reading is approximate – the integer ``spatial_grid`` is
    last-writer-wins so it cannot represent more than one occupant per
    cell – but it is *consistent*: if the same query is run before and
    after a hypothetical move on the same grid, the delta correctly
    reflects how many cells the move would newly land on top of.
    """
    grid_num_rows = spatial_grid.shape[0]
    grid_num_cols = spatial_grid.shape[1]
    c_lo, c_hi, r_lo, r_hi = fe.compute_macro_bin_range_njit(
        x_ll, y_ll, width, height,
        bin_width, bin_height, grid_num_cols, grid_num_rows,
    )
    n = 0
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            occ = spatial_grid[r, c]
            if occ != EMPTY_CELL and occ != macro_idx:
                n += 1
    return n


@njit(cache=True)
def per_macro_overlap_counts_njit(
    spatial_grid: np.ndarray,
    macro_coords: np.ndarray,
    macro_dims: np.ndarray,
    macro_is_hard: np.ndarray,
    num_macros: int,
    bin_width: float, bin_height: float,
    out_counts: np.ndarray,
) -> int:
    """Per-macro foreign-cell counts; total is returned for convenience.

    ``out_counts[m]`` is the same metric as
    :func:`count_foreign_cells_in_bbox_njit` evaluated at the macro's
    current position.  Used by :func:`refresh_selector_weights_njit` to
    bias the perturbation selector towards "worst offenders".
    """
    total = 0
    for m in range(num_macros):
        if not macro_is_hard[m]:
            out_counts[m] = 0
            continue
        c = count_foreign_cells_in_bbox_njit(
            spatial_grid, m,
            macro_coords[m, 0], macro_coords[m, 1],
            macro_dims[m, 0],   macro_dims[m, 1],
            bin_width, bin_height,
        )
        out_counts[m] = c
        total += c
    return total


@njit(cache=True)
def refresh_selector_weights_njit(
    macro_fixed: np.ndarray,
    macro_is_hard: np.ndarray,
    macro_net_ids: np.ndarray,
    macro_net_offsets: np.ndarray,
    net_bbox: np.ndarray,
    net_weights: np.ndarray,
    overlap_counts: np.ndarray,
    base_weight: float,
    overlap_weight: float,
    netlen_weight: float,
    out_weights: np.ndarray,
) -> float:
    """Rebuild ``out_weights[m]`` from current overlap / netlen state.

    Formula::

        w_m = 0                                                     if fixed
            = base + overlap_w · ovl(m) + netlen_w · Σ_n w_n·hpwl(n) otherwise

    The cumulative sum used for fast ``np.searchsorted`` sampling is
    *not* built here – it lives in pure-Python `_sample_biased` below.

    Returns the new total weight (caller may early-exit if zero).
    """
    num_macros = macro_fixed.shape[0]
    total = 0.0
    for m in range(num_macros):
        if macro_fixed[m]:
            out_weights[m] = 0.0
            continue
        # Net-length contribution: weighted sum of HPWL over connected nets.
        s = macro_net_offsets[m]
        e = macro_net_offsets[m + 1]
        netlen = 0.0
        for i in range(s, e):
            nid = macro_net_ids[i]
            hpwl_n = (net_bbox[nid, 1] - net_bbox[nid, 0]) \
                   + (net_bbox[nid, 3] - net_bbox[nid, 2])
            netlen += net_weights[nid] * hpwl_n
        w = base_weight + netlen_weight * netlen
        if macro_is_hard[m]:
            w += overlap_weight * float(overlap_counts[m])
        if w < 1e-12:
            w = 1e-12  # guard against zero in degenerate runs
        out_weights[m] = w
        total += w
    return total


@njit(cache=True)
def exact_hard_macro_overlap_pairs_njit(
    macro_coords: np.ndarray,
    macro_dims: np.ndarray,
    num_hard_macros: int,
    tolerance: float,
) -> int:
    """Count unordered pairs of hard macros whose bboxes properly intersect.

    Uses an :math:`O(N^2)` sweep because it runs *exactly once* per
    worker (final validity check).  ``tolerance`` lets a tiny numerical
    touch (≤ ``tolerance`` along either axis) pass without flagging –
    matching the half-open bin convention used by ``compute_macro_bin_range``.
    """
    n = 0
    for i in range(num_hard_macros):
        xi = macro_coords[i, 0]; yi = macro_coords[i, 1]
        wi = macro_dims[i, 0];   hi = macro_dims[i, 1]
        for j in range(i + 1, num_hard_macros):
            xj = macro_coords[j, 0]; yj = macro_coords[j, 1]
            wj = macro_dims[j, 0];   hj = macro_dims[j, 1]
            overlap_x = (xi + wi) - xj
            if (xj + wj) - xi < overlap_x:
                overlap_x = (xj + wj) - xi
            overlap_y = (yi + hi) - yj
            if (yj + hj) - yi < overlap_y:
                overlap_y = (yj + hj) - yi
            if overlap_x > tolerance and overlap_y > tolerance:
                n += 1
    return n


@njit(cache=True)
def _macro_has_overlap_with_others_njit(
    m: int,
    x_ll: float, y_ll: float, w: float, h: float,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    num_hard_macros: int, tolerance: float,
) -> bool:
    """Exact O(N) test: would placing macro ``m`` at ``(x_ll, y_ll, w, h)``
    properly intersect *any* other hard macro at its current placement?

    Used inside the deterministic post-MCMC legalisation sweep where
    correctness trumps throughput: O(N) per query is acceptable because
    the sweep runs only a few thousand queries total, not millions.
    """
    for j in range(num_hard_macros):
        if j == m:
            continue
        xj = macro_coords[j, 0]; yj = macro_coords[j, 1]
        wj = macro_dims[j, 0];   hj = macro_dims[j, 1]
        ox = (x_ll + w) - xj
        if (xj + wj) - x_ll < ox:
            ox = (xj + wj) - x_ll
        oy = (y_ll + h) - yj
        if (yj + hj) - y_ll < oy:
            oy = (yj + hj) - y_ll
        if ox > tolerance and oy > tolerance:
            return True
    return False


@njit(cache=True)
def _per_hard_macro_overlap_neighbors_njit(
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    num_hard_macros: int, tolerance: float,
    out_counts: np.ndarray,
) -> int:
    """For each hard macro ``i``, count how many *other* hard macros'
    bboxes it properly intersects.  Stores per-macro counts in
    ``out_counts`` and returns the number of macros with ``count > 0``.

    Quadratic O(N²), but the sweep calls this at most a few dozen times
    so the absolute cost stays bounded.
    """
    n_involved = 0
    for i in range(num_hard_macros):
        cnt = 0
        xi = macro_coords[i, 0]; yi = macro_coords[i, 1]
        wi = macro_dims[i, 0];   hi = macro_dims[i, 1]
        for j in range(num_hard_macros):
            if j == i:
                continue
            xj = macro_coords[j, 0]; yj = macro_coords[j, 1]
            wj = macro_dims[j, 0];   hj = macro_dims[j, 1]
            ox = (xi + wi) - xj
            if (xj + wj) - xi < ox:
                ox = (xj + wj) - xi
            oy = (yi + hi) - yj
            if (yj + hj) - yi < oy:
                oy = (yj + hj) - yi
            if ox > tolerance and oy > tolerance:
                cnt += 1
        out_counts[i] = cnt
        if cnt > 0:
            n_involved += 1
    return n_involved


@njit(cache=True)
def _find_collision_free_spiral_njit(
    m: int,
    macro_coords: np.ndarray, macro_dims: np.ndarray,
    num_hard_macros: int,
    canvas_width: float, canvas_height: float,
    step_x: float, step_y: float, max_radius: int,
    tolerance: float,
) -> Tuple[float, float, int]:
    """8-direction spiral search for the nearest collision-free position
    for macro ``m``.

    Walks outward at integer multiples of ``(step_x, step_y)`` along the
    eight compass directions (axis-aligned first, then diagonals, so the
    closest legal slot is found first when ties occur).  Each candidate
    is canvas-clamped and passed through the exact O(N) bbox check.

    Returns ``(best_x_ll, best_y_ll, radius_found)``.  ``radius_found ==
    -1`` indicates no collision-free position was located within
    ``max_radius`` steps in any direction (caller should leave the macro
    where it was and try again in a later pass after its neighbours have
    moved out of the way).  ``radius_found == 0`` means the current
    position is already clean (sub-tolerance phantom flagged by the
    coarser involvement counter).
    """
    w = macro_dims[m, 0]; h = macro_dims[m, 1]
    x0 = macro_coords[m, 0]; y0 = macro_coords[m, 1]

    # Macros physically larger than the canvas cannot ever be legalised;
    # bail out so the caller can flag this as a structural infeasibility.
    if w > canvas_width + tolerance or h > canvas_height + tolerance:
        return x0, y0, -1

    if not _macro_has_overlap_with_others_njit(
        m, x0, y0, w, h, macro_coords, macro_dims, num_hard_macros, tolerance,
    ):
        return x0, y0, 0

    for r in range(1, max_radius + 1):
        for d in range(8):
            # Axis-aligned directions first so the nearest equidistant
            # axis-aligned slot is preferred over a diagonal one.
            if d == 0:   dx, dy = +1.0,  0.0   # E
            elif d == 1: dx, dy =  0.0, +1.0   # N
            elif d == 2: dx, dy = -1.0,  0.0   # W
            elif d == 3: dx, dy =  0.0, -1.0   # S
            elif d == 4: dx, dy = +1.0, +1.0   # NE
            elif d == 5: dx, dy = -1.0, +1.0   # NW
            elif d == 6: dx, dy = -1.0, -1.0   # SW
            else:        dx, dy = +1.0, -1.0   # SE
            nx = x0 + dx * r * step_x
            ny = y0 + dy * r * step_y
            if nx < 0.0: nx = 0.0
            if ny < 0.0: ny = 0.0
            if nx + w > canvas_width:  nx = canvas_width  - w
            if ny + h > canvas_height: ny = canvas_height - h
            if nx < -tolerance or ny < -tolerance:
                continue  # macro doesn't fit; should not happen here
            if not _macro_has_overlap_with_others_njit(
                m, nx, ny, w, h,
                macro_coords, macro_dims, num_hard_macros, tolerance,
            ):
                return nx, ny, r
    return x0, y0, -1


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 3. Scratch-buffer pack                                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝

@dataclass
class _Scratch:
    """Owned, per-worker scratch arrays.

    Sized once from the state and reused across the entire MCMC loop so
    the inner iteration allocates no Python objects.
    """
    affected_nets: np.ndarray       # int32, capacity = 2 * max_nets_per_macro
    new_bboxes:    np.ndarray       # float64, (capacity, 4)
    cell_rs:       np.ndarray       # int32, capacity = max_cells_per_swap
    cell_cs:       np.ndarray       # int32, capacity = max_cells_per_swap
    new_densities: np.ndarray       # float64, capacity = max_cells_per_swap

    overlap_counts: np.ndarray      # int32, (num_macros,)
    selector_weights: np.ndarray    # float64, (num_macros,)
    selector_cumsum:  np.ndarray    # float64, (num_macros,)


def _allocate_scratch(state: PlacementState) -> _Scratch:
    max_deg = max(1, fe.max_nets_per_macro(state.macro_net_offsets))
    max_cells_swap = max(
        1,
        fe.max_cells_per_swap(state.macro_dims, state.grid_bin_width, state.grid_bin_height),
    )
    cap_nets = 2 * max_deg
    return _Scratch(
        affected_nets=np.zeros(cap_nets, dtype=np.int32),
        new_bboxes   =np.zeros((cap_nets, 4), dtype=np.float64),
        cell_rs      =np.zeros(max_cells_swap, dtype=np.int32),
        cell_cs      =np.zeros(max_cells_swap, dtype=np.int32),
        new_densities=np.zeros(max_cells_swap, dtype=np.float64),
        overlap_counts  =np.zeros(state.num_macros, dtype=np.int32),
        selector_weights=np.zeros(state.num_macros, dtype=np.float64),
        selector_cumsum =np.zeros(state.num_macros, dtype=np.float64),
    )


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 4. Selector sampling helpers                                           ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _rebuild_cumsum(weights: np.ndarray, cumsum: np.ndarray) -> float:
    """Cumulative-sum rebuild for ``np.searchsorted`` sampling.

    Stored in-place in ``cumsum`` (same shape as ``weights``).  Returns
    the total weight so callers can detect "all-zero" pathologies.
    """
    np.cumsum(weights, out=cumsum)
    return float(cumsum[-1]) if cumsum.size > 0 else 0.0


def _sample_biased(
    cumsum: np.ndarray, total: float, rng: np.random.Generator,
    eligible_mask: Optional[np.ndarray] = None,
) -> int:
    """Draw one index according to the cumulative-weight distribution.

    ``cumsum`` must be a non-decreasing array; ``total = cumsum[-1]``.
    If ``eligible_mask`` is supplied, we redraw up to a small number of
    times if the sampled index is masked out – this is cheaper than
    rebuilding a masked cumsum every call and keeps the sampler ``O(log
    N)``.
    """
    if total <= 0.0:
        # Fallback: pure uniform over eligible mask (or full range).
        if eligible_mask is None or not eligible_mask.any():
            return int(rng.integers(0, cumsum.size))
        idx = np.flatnonzero(eligible_mask)
        return int(idx[rng.integers(0, idx.size)])

    for _ in range(8):
        u = rng.random() * total
        pos = int(np.searchsorted(cumsum, u, side="right"))
        if pos >= cumsum.size:
            pos = cumsum.size - 1
        if eligible_mask is None or eligible_mask[pos]:
            return pos
    # Rare fallback: pick uniformly from the eligible set.
    if eligible_mask is None:
        return int(rng.integers(0, cumsum.size))
    idx = np.flatnonzero(eligible_mask)
    if idx.size == 0:
        return int(rng.integers(0, cumsum.size))
    return int(idx[rng.integers(0, idx.size)])


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 5. Mutation proposals                                                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _propose_shift(
    state: PlacementState, m: int,
    sigma_x: float, sigma_y: float,
    rng: np.random.Generator,
) -> Tuple[float, float, bool]:
    """Sample a Gaussian shift around the current bottom-left of ``m``.

    Returns ``(new_x_ll, new_y_ll, in_canvas)``.  ``in_canvas`` is
    ``False`` iff the proposed bbox would protrude past the canvas (a
    Phase-2 hard rejection).
    """
    w = state.macro_dims[m, 0]; h = state.macro_dims[m, 1]
    dx = float(rng.normal(0.0, sigma_x))
    dy = float(rng.normal(0.0, sigma_y))
    new_x = float(state.macro_coords[m, 0]) + dx
    new_y = float(state.macro_coords[m, 1]) + dy

    cw = state.canvas_width; ch = state.canvas_height
    if w > cw + EPS or h > ch + EPS:
        return new_x, new_y, False
    # Clamp to canvas perimeter so a hot temperature does not waste 80 % of
    # proposals on out-of-bounds samples.  We still flag overshoots that
    # cannot be clamped (macros bigger than canvas, already filtered above).
    if new_x < 0.0:           new_x = 0.0
    if new_y < 0.0:           new_y = 0.0
    if new_x + w > cw - EPS:  new_x = max(0.0, cw - w)
    if new_y + h > ch - EPS:  new_y = max(0.0, ch - h)
    # After clamping, the proposal is by construction inside the canvas.
    return new_x, new_y, True


def _pick_swap_partner(
    state: PlacementState, a: int,
    movable_hard_idx: np.ndarray,
    area_tolerance: float, max_tries: int,
    rng: np.random.Generator,
) -> int:
    """Find a movable-hard macro ``b ≠ a`` of comparable area.

    Returns ``-1`` if no acceptable partner is found within
    ``max_tries`` draws.  The tolerance is applied symmetrically:
    ``|area(a) - area(b)| / max(area(a), area(b)) ≤ tolerance``.
    """
    if movable_hard_idx.size < 2:
        return -1
    area_a = float(state.macro_dims[a, 0] * state.macro_dims[a, 1])
    for _ in range(max_tries):
        b = int(movable_hard_idx[rng.integers(0, movable_hard_idx.size)])
        if b == a:
            continue
        area_b = float(state.macro_dims[b, 0] * state.macro_dims[b, 1])
        denom = area_a if area_a > area_b else area_b
        if denom <= 0.0:
            continue
        if abs(area_a - area_b) / denom <= area_tolerance:
            return b
    return -1


def _propose_swap(
    state: PlacementState, a: int, b: int,
) -> Tuple[float, float, float, float, bool]:
    """Compute new bottom-lefts for swap and verify canvas containment.

    Exchanges the *centres* of ``a`` and ``b``: macro ``a`` lands at
    ``b``'s old centre (translated to its own bottom-left) and vice
    versa.  Returns ``(xa_new, ya_new, xb_new, yb_new, in_canvas)``.
    """
    wa = float(state.macro_dims[a, 0]); ha = float(state.macro_dims[a, 1])
    wb = float(state.macro_dims[b, 0]); hb = float(state.macro_dims[b, 1])

    cx_a = float(state.macro_coords[a, 0]) + 0.5 * wa
    cy_a = float(state.macro_coords[a, 1]) + 0.5 * ha
    cx_b = float(state.macro_coords[b, 0]) + 0.5 * wb
    cy_b = float(state.macro_coords[b, 1]) + 0.5 * hb

    xa_new = cx_b - 0.5 * wa; ya_new = cy_b - 0.5 * ha
    xb_new = cx_a - 0.5 * wb; yb_new = cy_a - 0.5 * hb

    cw = state.canvas_width; ch = state.canvas_height
    in_canvas = (
        xa_new >= -EPS and xa_new + wa <= cw + EPS and
        ya_new >= -EPS and ya_new + ha <= ch + EPS and
        xb_new >= -EPS and xb_new + wb <= cw + EPS and
        yb_new >= -EPS and yb_new + hb <= ch + EPS
    )
    # Tighten to canvas if a sub-EPS overshoot snuck through.
    if in_canvas:
        if xa_new < 0.0: xa_new = 0.0
        if ya_new < 0.0: ya_new = 0.0
        if xb_new < 0.0: xb_new = 0.0
        if yb_new < 0.0: yb_new = 0.0
        if xa_new + wa > cw: xa_new = max(0.0, cw - wa)
        if ya_new + ha > ch: ya_new = max(0.0, ch - ha)
        if xb_new + wb > cw: xb_new = max(0.0, cw - wb)
        if yb_new + hb > ch: yb_new = max(0.0, ch - hb)
    return xa_new, ya_new, xb_new, yb_new, in_canvas


def _propose_reshape(
    state: PlacementState, m: int,
    log_aspect_sigma: float, max_aspect: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, bool]:
    """Soft-macro reshape: vary aspect ratio while preserving area and centre.

    Returns ``(new_x_ll, new_y_ll, new_w, new_h, in_canvas)``.
    """
    w_old = float(state.macro_dims[m, 0]); h_old = float(state.macro_dims[m, 1])
    area  = w_old * h_old
    if area <= 0.0:
        return 0.0, 0.0, w_old, h_old, False
    cx = float(state.macro_coords[m, 0]) + 0.5 * w_old
    cy = float(state.macro_coords[m, 1]) + 0.5 * h_old

    cur_aspect = w_old / h_old if h_old > 0.0 else 1.0
    log_a = math.log(cur_aspect) + float(rng.normal(0.0, log_aspect_sigma))
    # Clamp to a sensible aspect-ratio range so degenerate slivers never appear.
    log_max = math.log(max_aspect)
    if log_a >  log_max: log_a =  log_max
    if log_a < -log_max: log_a = -log_max
    aspect = math.exp(log_a)
    new_h = math.sqrt(area / aspect)
    new_w = aspect * new_h

    new_x = cx - 0.5 * new_w
    new_y = cy - 0.5 * new_h

    cw = state.canvas_width; ch = state.canvas_height
    if new_w > cw or new_h > ch:
        return new_x, new_y, new_w, new_h, False
    if new_x < 0.0: new_x = 0.0
    if new_y < 0.0: new_y = 0.0
    if new_x + new_w > cw: new_x = max(0.0, cw - new_w)
    if new_y + new_h > ch: new_y = max(0.0, ch - new_h)
    return new_x, new_y, new_w, new_h, True


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 6. Cost helpers                                                        ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _compute_overlap_penalty_coef(
    T: float, T0: float, base: float, grow_k: float,
) -> float:
    """Per-foreign-cell penalty coefficient as a function of temperature.

    ``cursor.md`` §F.1: "overlaps are permitted... but are assigned an
    explicit penalty factor that scales up *exponentially* as the
    temperature parameter T drops."

    .. math::
        k(T) = k_0 \\cdot \\exp\\bigl( g \\cdot (1 - T / T_0) \\bigr)

    so at ``T = T0`` the cost is just ``k_0`` and at ``T → 0`` the cost
    grows like ``k_0 · e^g``.
    """
    if T0 <= 0.0:
        return base
    ratio = T / T0
    if ratio > 1.0:
        ratio = 1.0
    if ratio < 0.0:
        ratio = 0.0
    return base * math.exp(grow_k * (1.0 - ratio))


def _proxy_cost(
    cost_wl: float, cost_d: float, cost_c: float, cfg: WorkerConfig,
) -> float:
    return (
        cfg.weight_wirelength * cost_wl
        + cfg.weight_density  * cost_d
        + cfg.weight_congestion * cost_c
    )


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 7. Validity sweep                                                      ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _final_validity_check(
    state: PlacementState,
    *,
    tolerance: float = 1e-6,
    pairwise_check_cap: int = 8_000,
) -> Tuple[bool, List[str]]:
    """Comprehensive end-of-run sweep mandated by ``cursor.md`` §F.4.

    Three checks are performed but only two contribute to the final
    ``valid`` decision:

    1. **Canvas containment** for every macro (vectorised, authoritative).
    2. **Bin-grid collision count** after a full re-stamp – *diagnostic
       only*.  The integer ``spatial_grid`` is approximate: two macros
       can share a bin without their bounding boxes intersecting (e.g.
       ``A ∈ [0.0, 0.4]`` and ``B ∈ [0.5, 0.9]`` with ``bin_w = 1.0``
       both fall into bin 0).  The challenge evaluator's
       ``validate_placement`` uses exact geometric arithmetic and would
       *not* flag this as an overlap, so neither do we.  The bin count
       is kept as a non-fatal diagnostic message.
    3. **Exact pairwise hard-macro bbox intersection** – authoritative
       when ``num_hard ≤ pairwise_check_cap``.  When this O(N²) check
       is available, it is the sole signal that decides hard-macro
       legality.  When it is skipped (very large benchmarks), the bin
       count falls back to authoritative.
    """
    violations: List[str] = []

    # 1. Canvas containment (authoritative).
    if state.num_macros > 0:
        x_lo = state.macro_coords[:, 0]
        y_lo = state.macro_coords[:, 1]
        x_hi = x_lo + state.macro_dims[:, 0]
        y_hi = y_lo + state.macro_dims[:, 1]
        bad_x = (x_lo < -tolerance) | (x_hi > state.canvas_width  + tolerance)
        bad_y = (y_lo < -tolerance) | (y_hi > state.canvas_height + tolerance)
        n_bad = int(np.count_nonzero(bad_x | bad_y))
        if n_bad > 0:
            violations.append(
                f"{n_bad} macros exceed canvas perimeter "
                f"({state.canvas_width:.2f}×{state.canvas_height:.2f})"
            )

    # 2. Re-stamp the bin grid and read its (approximate) collision
    #    count.  This is kept as a diagnostic so any spatial-grid
    #    drift during the MCMC inner loop is visible to the user.
    stamp_all_hard_macros(state)
    n_grid_coll = int(fe.count_grid_collisions_njit(
        state.spatial_grid,
        state.macro_coords, state.macro_dims,
        state.num_hard_macros,
        state.grid_bin_width, state.grid_bin_height,
    ))

    # 3. Exact pairwise hard-macro bbox intersection (authoritative
    #    when below cap).  Runs O(N²) but only once at end-of-run.
    exact_pairs_known = False
    n_pairs = 0
    if state.num_hard_macros <= pairwise_check_cap:
        n_pairs = int(exact_hard_macro_overlap_pairs_njit(
            state.macro_coords, state.macro_dims,
            state.num_hard_macros, tolerance,
        ))
        exact_pairs_known = True
        if n_pairs > 0:
            violations.append(
                f"{n_pairs} exact hard-macro pair overlaps (bbox intersect)"
            )

    # Bin-grid signal: fatal only when the exact check is unavailable
    # (too many macros) or when both agree the layout has overlaps.
    if n_grid_coll > 0 and not exact_pairs_known:
        violations.append(
            f"{n_grid_coll} foreign-cell hits in spatial grid "
            f"(hard-macro overlap; bin signal authoritative – exact "
            f"check skipped, ``num_hard={state.num_hard_macros}`` "
            f"exceeds pairwise_check_cap={pairwise_check_cap})"
        )

    return (len(violations) == 0), violations


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 7b. Post-MCMC greedy legalisation sweep                                ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _greedy_legalize_sweep(
    state: PlacementState,
    *,
    max_passes: int = 50,
    max_radius_steps: int = 256,
    step_scale: float = 0.5,
    tolerance: float = 1e-6,
) -> Tuple[int, int, int, int]:
    """Deterministic legalisation pass that runs *after* the Metropolis
    loop and *before* the final validity check.

    The MCMC's per-cell foreign-count signal is approximate – the
    integer spatial grid stores only one occupant per cell, so the last
    few residual geometric overlaps (typical on dense benchmarks such as
    ibm02) can survive even when the bin signal claims a clean layout.
    This sweep uses *exact* bbox arithmetic to detect every collision
    pair and an 8-direction spiral search to evict each colliding macro
    to the nearest collision-free position, respecting canvas bounds.

    The function mutates ``state.macro_coords`` in place.  The spatial
    grid is intentionally *not* re-stamped here; the immediately
    following :func:`_final_validity_check` call already does a full
    re-stamp before counting residual collisions, so re-stamping inside
    the sweep would be wasted work.

    Args:
        state: ``PlacementState`` with the post-MCMC layout.
        max_passes: hard cap on outer iterations.  Most benchmarks
            converge within ≤ 3 passes; the cap is the failure budget.
        max_radius_steps: max distance (in step units) the spiral search
            walks before giving up on a macro for this pass.
        step_scale: scale factor applied to the smallest hard-macro
            dimension when picking the per-step size.  ``0.5`` means
            each spiral hop is half a tiny-macro wide – fine enough to
            slip into narrow gaps without being so small that the search
            stalls.
        tolerance: same tolerance convention as
            :func:`exact_hard_macro_overlap_pairs_njit`.

    Returns:
        ``(initial_pair_count, final_pair_count, macros_moved,
        passes_executed)``.  When
        ``initial_pair_count == final_pair_count > 0`` the sweep made no
        progress – the layout is structurally infeasible at the
        current canvas size.
    """
    coords = state.macro_coords
    dims   = state.macro_dims
    fixed  = state.macro_fixed
    n_h    = int(state.num_hard_macros)
    cw     = float(state.canvas_width)
    ch     = float(state.canvas_height)

    if n_h <= 0:
        return (0, 0, 0, 0)

    initial_pairs = int(exact_hard_macro_overlap_pairs_njit(
        coords, dims, n_h, tolerance,
    ))
    if initial_pairs == 0:
        return (0, 0, 0, 0)

    # Step size: half the smallest hard-macro side, floored so we never
    # crawl by zero on a degenerate input.  This is intentionally
    # *smaller* than the MCMC's σ so the sweep can slot macros into
    # narrow gaps the MCMC couldn't resolve on its own.
    min_w = float(dims[:n_h, 0].min())
    min_h = float(dims[:n_h, 1].min())
    step_x = max(min_w * step_scale, cw / 1024.0)
    step_y = max(min_h * step_scale, ch / 1024.0)

    involvement = np.zeros(n_h, dtype=np.int64)
    moved_total = 0
    passes_executed = 0

    for pass_id in range(max_passes):
        passes_executed = pass_id + 1
        # Recompute per-macro overlap involvement so we always attack the
        # worst offender first.  O(N²) per pass but only a handful of
        # passes are ever needed in practice.
        _per_hard_macro_overlap_neighbors_njit(
            coords, dims, n_h, tolerance, involvement,
        )
        # ``np.argsort(-involvement)`` would put fixed-and-clean macros
        # at the tail; we still scan top-down so the inner break works.
        order = np.argsort(-involvement, kind="stable")

        moved_this_pass = 0
        for idx in order:
            m = int(idx)
            if involvement[m] == 0:
                break  # remaining macros have no overlap; we're done this pass
            if fixed[m]:
                continue
            nx, ny, r_used = _find_collision_free_spiral_njit(
                m, coords, dims, n_h, cw, ch,
                step_x, step_y, max_radius_steps, tolerance,
            )
            if r_used > 0:
                coords[m, 0] = nx
                coords[m, 1] = ny
                moved_this_pass += 1

        moved_total += moved_this_pass
        if moved_this_pass == 0:
            # No further progress possible in this configuration – break
            # rather than waste the rest of the budget on no-ops.
            break

        # Early termination if we've already cleared all overlaps; the
        # next involvement scan would confirm zero anyway.
        if int(exact_hard_macro_overlap_pairs_njit(
            coords, dims, n_h, tolerance,
        )) == 0:
            break

    final_pairs = int(exact_hard_macro_overlap_pairs_njit(
        coords, dims, n_h, tolerance,
    ))
    return (initial_pairs, final_pairs, moved_total, passes_executed)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 8. Temperature warmup                                                  ║
# ╚════════════════════════════════════════════════════════════════════════╝

def _warmup_initial_temperature(
    state: PlacementState,
    density_grid: np.ndarray,
    scratch: _Scratch,
    movable_hard_idx: np.ndarray,
    cfg: WorkerConfig,
    rng: np.random.Generator,
) -> float:
    """Estimate ``T₀`` so that warm acceptance ≈ ``warmup_target_accept``.

    Generates ``warmup_iters`` random shift proposals (without
    committing any), measures the mean *positive* delta-cost, and
    inverts the Boltzmann criterion::

        p_target = exp(-|ΔC|̄ / T₀)  ⇒  T₀ = -|ΔC|̄ / ln(p_target)

    This is the standard adaptive simulated-annealing warmup.  We use
    only shift proposals because they are the cheapest and the most
    representative of the overall cost landscape.
    """
    if movable_hard_idx.size == 0 or cfg.warmup_iters <= 0:
        return 1.0

    sigma_x = max(EPS, state.canvas_width  * cfg.shift_sigma_init_frac)
    sigma_y = max(EPS, state.canvas_height * cfg.shift_sigma_init_frac)

    abs_deltas: List[float] = []
    n = cfg.warmup_iters
    for _ in range(n):
        m = int(movable_hard_idx[rng.integers(0, movable_hard_idx.size)])
        new_x, new_y, ok = _propose_shift(state, m, sigma_x, sigma_y, rng)
        if not ok:
            continue
        # HPWL delta (no commit).
        d_wl, _ = fe.hpwl_delta_for_shift_njit(
            m, new_x, new_y,
            state.macro_coords, state.macro_dims,
            state.port_coords, state.num_macros,
            state.macro_net_ids, state.macro_net_offsets,
            state.net_pin_owners, state.net_pin_offsets,
            state.net_weights, state.net_bbox,
            scratch.affected_nets, scratch.new_bboxes,
        )
        # Density / congestion delta (no commit).
        d_d, d_c, _ = fe.density_grid_shift_delta_njit(
            m, new_x, new_y,
            state.macro_coords, state.macro_dims,
            state.grid_bin_width, state.grid_bin_height,
            state.grid_num_rows, state.grid_num_cols,
            density_grid,
            scratch.cell_rs, scratch.cell_cs, scratch.new_densities,
        )
        d_proxy = _proxy_cost(d_wl, d_d, d_c, cfg)
        if d_proxy > 0.0:
            abs_deltas.append(d_proxy)

    if not abs_deltas:
        return 1.0
    mean_dc = float(np.mean(abs_deltas))
    p = cfg.warmup_target_accept
    if not (0.0 < p < 1.0):
        p = 0.85
    T0 = -mean_dc / math.log(p)
    if not math.isfinite(T0) or T0 <= 0.0:
        T0 = mean_dc if mean_dc > 0.0 else 1.0
    return T0


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 9. Main worker entry point                                             ║
# ╚════════════════════════════════════════════════════════════════════════╝

def run_worker(
    state: PlacementState,
    config: Optional[WorkerConfig] = None,
    *,
    copy_state: bool = True,
) -> WorkerResult:
    """Run the complete MCMC optimisation loop on ``state``.

    Parameters
    ----------
    state
        Input :class:`PlacementState`.  Cloned by default so the caller's
        copy is untouched.
    config
        Optional :class:`WorkerConfig`.  When omitted a default-tuned
        config is used.
    copy_state
        Set ``False`` to mutate ``state`` in place (e.g. when the worker
        already owns a per-process clone from ``multiprocessing.fork``).
    """
    cfg = config or WorkerConfig()
    if copy_state:
        state = clone_state(state)
    rng = np.random.default_rng(int(cfg.seed))

    # ── Stage 0: GRASP construction ─────────────────────────────────────
    if cfg.enable_grasp:
        state, grasp_report = grasp_initialize(
            state, seed=int(cfg.seed),
            num_candidates=cfg.grasp_num_candidates,
            top_k=cfg.grasp_top_k,
            copy_state=False,
        )
    else:
        # Just rebuild the cached net_bbox in case the caller mutated coords.
        fe.populate_net_bbox_njit(
            state.net_pin_owners, state.net_pin_offsets,
            state.macro_coords, state.macro_dims,
            state.port_coords, state.num_macros,
            state.num_nets, state.net_bbox,
        )
        stamp_all_hard_macros(state)
        grasp_report = GraspReport(
            seed=int(cfg.seed), num_movable_hard=0, num_placed_clean=0,
            num_placed_via_blf=0, num_placed_with_overlap=0,
            sum_partial_hpwl=0.0, movable_order=np.zeros(0, dtype=np.int64),
        )

    # ── Stage 1: allocate scratch / auxiliary arrays ────────────────────
    scratch = _allocate_scratch(state)
    density_grid = np.zeros(
        (state.grid_num_rows, state.grid_num_cols), dtype=np.float64,
    )
    fe.compute_density_grid_njit(
        state.macro_coords, state.macro_dims,
        state.num_macros,
        state.grid_bin_width, state.grid_bin_height,
        state.grid_num_rows, state.grid_num_cols,
        density_grid,
    )

    movable_mask = ~state.macro_fixed
    movable_hard_mask = movable_mask & state.macro_is_hard
    movable_soft_mask = movable_mask & (~state.macro_is_hard)
    movable_hard_idx = np.flatnonzero(movable_hard_mask).astype(np.int64, copy=False)
    movable_soft_idx = np.flatnonzero(movable_soft_mask).astype(np.int64, copy=False)
    has_movable_hard = movable_hard_idx.size > 0
    has_movable_soft = movable_soft_idx.size > 0
    has_anything = has_movable_hard or has_movable_soft

    # ── Stage 2: initial cost ───────────────────────────────────────────
    cost_wl = float(fe.compute_total_hpwl_njit(
        state.net_pin_owners, state.net_pin_offsets,
        state.net_weights,
        state.macro_coords, state.macro_dims,
        state.port_coords, state.num_macros,
        state.num_nets,
    ))
    cost_d  = float(fe.density_cost_from_grid_njit(density_grid))
    cost_c  = float(fe.congestion_cost_from_grid_njit(density_grid))
    cost_proxy = _proxy_cost(cost_wl, cost_d, cost_c, cfg)
    initial_cost_proxy = cost_proxy

    # ── Stage 3: warm-up temperature ────────────────────────────────────
    if has_movable_hard:
        T0 = _warmup_initial_temperature(
            state, density_grid, scratch, movable_hard_idx, cfg, rng,
        )
    else:
        T0 = 1.0
    T_end = max(T0 * cfg.final_temp_factor, 1e-12)
    T_legalization = T0 * cfg.legalization_temp_factor
    T = T0

    # ── Stage 4: cooling schedule (time-budget driven) ──────────────────
    # We hold ``T = T0`` for the first ``cooling_recalibrate_iters`` steps
    # so the *first* recalibration tick has a measured iter/sec to base
    # its α on.  After that, every recalibration period sets α so that
    # ``T`` reaches ``T_end`` precisely at the remaining wall-clock
    # deadline – the schedule is therefore self-tuning regardless of
    # benchmark size or CPU speed.
    deadline = time.perf_counter() + max(0.0, cfg.time_budget_seconds)
    alpha = 1.0
    last_recal_iter = 0
    last_recal_time = time.perf_counter()

    # ── Stage 5: selector weights (initial build) ───────────────────────
    total_overlap = int(per_macro_overlap_counts_njit(
        state.spatial_grid,
        state.macro_coords, state.macro_dims,
        state.macro_is_hard,
        state.num_macros,
        state.grid_bin_width, state.grid_bin_height,
        scratch.overlap_counts,
    ))
    refresh_selector_weights_njit(
        state.macro_fixed, state.macro_is_hard,
        state.macro_net_ids, state.macro_net_offsets,
        state.net_bbox, state.net_weights,
        scratch.overlap_counts,
        cfg.selector_base_weight,
        cfg.selector_overlap_weight,
        cfg.selector_netlen_weight,
        scratch.selector_weights,
    )
    _rebuild_cumsum(scratch.selector_weights, scratch.selector_cumsum)

    # ── Stage 6: counters ───────────────────────────────────────────────
    accepts_shift = rejects_shift = 0
    accepts_swap  = rejects_swap  = 0
    accepts_reshape = rejects_reshape = 0
    rejects_canvas = rejects_legalization = 0
    iterations = 0
    overlap_penalty_total = 0.0  # cumulative penalty contribution

    bw = state.grid_bin_width
    bh = state.grid_bin_height
    canvas_w = state.canvas_width
    canvas_h = state.canvas_height

    # Mutation probability bands (recomputed every iter from temperature).
    def _mut_probs(T_now: float) -> Tuple[float, float, float]:
        t_ratio = 0.0 if T0 <= 0.0 else T_now / T0
        if t_ratio > 1.0: t_ratio = 1.0
        if t_ratio < 0.0: t_ratio = 0.0
        ps = cfg.p_shift_cold   + (cfg.p_shift_hot   - cfg.p_shift_cold)   * t_ratio
        pw = cfg.p_swap_cold    + (cfg.p_swap_hot    - cfg.p_swap_cold)    * t_ratio
        pr = cfg.p_reshape_cold + (cfg.p_reshape_hot - cfg.p_reshape_cold) * t_ratio
        # Zero out impossible mutations and renormalise.
        if not has_movable_hard:
            ps = 0.0; pw = 0.0
        if movable_hard_idx.size < 2:
            pw = 0.0
        if not has_movable_soft or cfg.p_reshape_hot <= 0.0:
            pr = 0.0
        s = ps + pw + pr
        if s <= 0.0:
            if has_movable_hard:  return 1.0, 0.0, 0.0
            if has_movable_soft:  return 0.0, 0.0, 1.0
            return 0.0, 0.0, 0.0
        return ps / s, pw / s, pr / s

    start_time = time.perf_counter()

    # Pre-bind locals for the hot loop.
    _hpwl_shift = fe.hpwl_delta_for_shift_njit
    _hpwl_swap  = fe.hpwl_delta_for_swap_njit
    _commit_hpwl = fe.commit_hpwl_delta_njit
    _d_shift = fe.density_grid_shift_delta_njit
    _d_swap  = fe.density_grid_swap_delta_njit
    _d_reshape = fe.density_grid_reshape_delta_njit
    _commit_d = fe.commit_density_grid_delta_njit
    _paint = fe.paint_macro_njit
    _clear = fe.clear_macro_njit
    _count_foreign = count_foreign_cells_in_bbox_njit

    macro_coords = state.macro_coords
    macro_dims   = state.macro_dims
    spatial_grid = state.spatial_grid
    net_bbox     = state.net_bbox
    macro_is_hard = state.macro_is_hard
    grid_rows = state.grid_num_rows
    grid_cols = state.grid_num_cols
    aff = scratch.affected_nets
    nbb = scratch.new_bboxes
    crs = scratch.cell_rs
    ccs = scratch.cell_cs
    nds = scratch.new_densities

    # ── Stage 7: main MCMC loop ─────────────────────────────────────────
    if not has_anything:
        # No movable macros at all — degenerate but valid.  Fall through
        # to the validity check directly.
        pass

    while has_anything and iterations < cfg.max_iterations:
        now = time.perf_counter()
        if now >= deadline:
            break

        # ── 7a. Cooling-schedule recalibration ──────────────────────────
        if iterations - last_recal_iter >= cfg.cooling_recalibrate_iters:
            dt = now - last_recal_time
            dn = iterations - last_recal_iter
            iter_per_sec = (dn / dt) if dt > 0.0 else 0.0
            time_left = deadline - now
            est_remaining = max(1, int(iter_per_sec * time_left))
            if T > T_end:
                alpha = (T_end / T) ** (1.0 / est_remaining)
            last_recal_iter = iterations
            last_recal_time = now

        # ── 7b. Mutation type ───────────────────────────────────────────
        p_shift, p_swap, p_reshape = _mut_probs(T)
        r_mut = rng.random()
        if r_mut < p_shift:
            mut = 0  # shift
        elif r_mut < p_shift + p_swap:
            mut = 1  # swap
        else:
            mut = 2  # reshape

        # Temperature-scaled shift sigma (decays linearly with T).
        t_ratio = 0.0 if T0 <= 0.0 else T / T0
        if t_ratio > 1.0: t_ratio = 1.0
        if t_ratio < 0.0: t_ratio = 0.0
        sigma_frac = (
            cfg.shift_sigma_final_frac
            + (cfg.shift_sigma_init_frac - cfg.shift_sigma_final_frac) * t_ratio
        )
        sigma_x = max(bw, canvas_w * sigma_frac)
        sigma_y = max(bh, canvas_h * sigma_frac)

        in_legalization = T < T_legalization
        penalty_coef = _compute_overlap_penalty_coef(
            T, T0, cfg.overlap_penalty_base, cfg.overlap_penalty_grow_k,
        )

        if mut == 0:
            # ── SHIFT ────────────────────────────────────────────────────
            m = _sample_biased(
                scratch.selector_cumsum,
                float(scratch.selector_cumsum[-1]),
                rng,
                eligible_mask=movable_mask,
            )
            old_x = float(macro_coords[m, 0]); old_y = float(macro_coords[m, 1])
            w = float(macro_dims[m, 0]); h = float(macro_dims[m, 1])

            new_x, new_y, in_canvas = _propose_shift(state, m, sigma_x, sigma_y, rng)
            if not in_canvas:
                rejects_canvas += 1
                rejects_shift  += 1
                iterations += 1
                T *= alpha
                continue

            is_hard_m = bool(macro_is_hard[m])

            # Phase-3 strict legalisation rejection (hard macros only).
            old_coll = 0
            new_coll = 0
            if is_hard_m and in_legalization:
                old_coll = _count_foreign(spatial_grid, m, old_x, old_y, w, h, bw, bh)
                new_coll = _count_foreign(spatial_grid, m, new_x, new_y, w, h, bw, bh)
                if new_coll > old_coll:
                    rejects_legalization += 1
                    rejects_shift += 1
                    iterations += 1
                    T *= alpha
                    continue

            # Cost deltas.
            d_wl, n_aff = _hpwl_shift(
                m, new_x, new_y,
                macro_coords, macro_dims,
                state.port_coords, state.num_macros,
                state.macro_net_ids, state.macro_net_offsets,
                state.net_pin_owners, state.net_pin_offsets,
                state.net_weights, net_bbox,
                aff, nbb,
            )
            d_d, d_c, n_cells = _d_shift(
                m, new_x, new_y,
                macro_coords, macro_dims,
                bw, bh, grid_rows, grid_cols,
                density_grid, crs, ccs, nds,
            )
            d_proxy = _proxy_cost(d_wl, d_d, d_c, cfg)

            # Phase-1 overlap penalty (counts foreign cells the macro
            # would land on – ignores cells it would vacate; if we didn't
            # already compute old/new collision counts above, do so now).
            d_penalty = 0.0
            if is_hard_m and penalty_coef > 0.0:
                if not in_legalization:
                    old_coll = _count_foreign(spatial_grid, m, old_x, old_y, w, h, bw, bh)
                    new_coll = _count_foreign(spatial_grid, m, new_x, new_y, w, h, bw, bh)
                d_overlap_cells = new_coll - old_coll
                d_penalty = penalty_coef * float(d_overlap_cells)

            d_total = d_proxy + d_penalty

            # Metropolis criterion.
            accept = False
            if d_total <= 0.0:
                accept = True
            else:
                p = math.exp(-d_total / T) if T > 0.0 else 0.0
                if rng.random() < p:
                    accept = True

            if accept:
                _commit_hpwl(aff, nbb, n_aff, net_bbox)
                _commit_d(crs, ccs, nds, n_cells, density_grid)
                if is_hard_m:
                    _clear(spatial_grid, m, old_x, old_y, w, h, bw, bh)
                macro_coords[m, 0] = new_x
                macro_coords[m, 1] = new_y
                if is_hard_m:
                    _paint(spatial_grid, m, new_x, new_y, w, h, bw, bh)
                cost_wl += d_wl
                cost_d  += d_d
                cost_c  += d_c
                cost_proxy += d_proxy
                overlap_penalty_total += d_penalty
                accepts_shift += 1
            else:
                rejects_shift += 1

        elif mut == 1:
            # ── SWAP ────────────────────────────────────────────────────
            a = _sample_biased(
                scratch.selector_cumsum,
                float(scratch.selector_cumsum[-1]),
                rng,
                eligible_mask=movable_hard_mask,
            )
            if not movable_hard_mask[a]:
                # Sampler may have picked a non-hard macro under degenerate
                # weight conditions; rejoin the loop with a uniform pick.
                a = int(movable_hard_idx[rng.integers(0, movable_hard_idx.size)])

            b = _pick_swap_partner(
                state, a, movable_hard_idx,
                cfg.swap_area_tolerance, cfg.swap_max_partner_tries,
                rng,
            )
            if b < 0:
                rejects_swap += 1
                iterations += 1
                T *= alpha
                continue

            xa_old = float(macro_coords[a, 0]); ya_old = float(macro_coords[a, 1])
            wa = float(macro_dims[a, 0]);       ha = float(macro_dims[a, 1])
            xb_old = float(macro_coords[b, 0]); yb_old = float(macro_coords[b, 1])
            wb = float(macro_dims[b, 0]);       hb = float(macro_dims[b, 1])

            xa_new, ya_new, xb_new, yb_new, in_canvas = _propose_swap(state, a, b)
            if not in_canvas:
                rejects_canvas += 1
                rejects_swap   += 1
                iterations += 1
                T *= alpha
                continue

            # Phase-3 strict legalisation rejection.
            if in_legalization:
                # Temporarily clear both macros from the grid so the
                # collision count reflects the post-swap configuration
                # without double-counting each other.
                _clear(spatial_grid, a, xa_old, ya_old, wa, ha, bw, bh)
                _clear(spatial_grid, b, xb_old, yb_old, wb, hb, bw, bh)
                new_a = _count_foreign(spatial_grid, a, xa_new, ya_new, wa, ha, bw, bh)
                new_b = _count_foreign(spatial_grid, b, xb_new, yb_new, wb, hb, bw, bh)
                # Restore the integer grid before deciding accept/reject.
                _paint(spatial_grid, a, xa_old, ya_old, wa, ha, bw, bh)
                _paint(spatial_grid, b, xb_old, yb_old, wb, hb, bw, bh)
                # "Before swap" foreign-cell counts at current positions:
                old_a = _count_foreign(spatial_grid, a, xa_old, ya_old, wa, ha, bw, bh)
                old_b = _count_foreign(spatial_grid, b, xb_old, yb_old, wb, hb, bw, bh)
                if new_a + new_b > old_a + old_b:
                    rejects_legalization += 1
                    rejects_swap += 1
                    iterations += 1
                    T *= alpha
                    continue

            # Cost deltas.
            d_wl, n_aff = _hpwl_swap(
                a, b, xa_new, ya_new, xb_new, yb_new,
                macro_coords, macro_dims,
                state.port_coords, state.num_macros,
                state.macro_net_ids, state.macro_net_offsets,
                state.net_pin_owners, state.net_pin_offsets,
                state.net_weights, net_bbox,
                aff, nbb,
            )
            d_d, d_c, n_cells = _d_swap(
                a, b, xa_new, ya_new, xb_new, yb_new,
                macro_coords, macro_dims,
                bw, bh, grid_rows, grid_cols,
                density_grid, crs, ccs, nds,
            )
            d_proxy = _proxy_cost(d_wl, d_d, d_c, cfg)

            # Phase-1 soft overlap penalty (computed with both macros
            # temporarily cleared from the grid, so foreign-cell counts
            # exclude the swap partners themselves).
            d_penalty = 0.0
            if penalty_coef > 0.0:
                _clear(spatial_grid, a, xa_old, ya_old, wa, ha, bw, bh)
                _clear(spatial_grid, b, xb_old, yb_old, wb, hb, bw, bh)
                new_a = _count_foreign(spatial_grid, a, xa_new, ya_new, wa, ha, bw, bh)
                new_b = _count_foreign(spatial_grid, b, xb_new, yb_new, wb, hb, bw, bh)
                old_a = _count_foreign(spatial_grid, a, xa_old, ya_old, wa, ha, bw, bh)
                old_b = _count_foreign(spatial_grid, b, xb_old, yb_old, wb, hb, bw, bh)
                _paint(spatial_grid, a, xa_old, ya_old, wa, ha, bw, bh)
                _paint(spatial_grid, b, xb_old, yb_old, wb, hb, bw, bh)
                d_overlap_cells = (new_a + new_b) - (old_a + old_b)
                d_penalty = penalty_coef * float(d_overlap_cells)

            d_total = d_proxy + d_penalty

            accept = False
            if d_total <= 0.0:
                accept = True
            else:
                p = math.exp(-d_total / T) if T > 0.0 else 0.0
                if rng.random() < p:
                    accept = True

            if accept:
                _commit_hpwl(aff, nbb, n_aff, net_bbox)
                _commit_d(crs, ccs, nds, n_cells, density_grid)
                _clear(spatial_grid, a, xa_old, ya_old, wa, ha, bw, bh)
                _clear(spatial_grid, b, xb_old, yb_old, wb, hb, bw, bh)
                macro_coords[a, 0] = xa_new; macro_coords[a, 1] = ya_new
                macro_coords[b, 0] = xb_new; macro_coords[b, 1] = yb_new
                _paint(spatial_grid, a, xa_new, ya_new, wa, ha, bw, bh)
                _paint(spatial_grid, b, xb_new, yb_new, wb, hb, bw, bh)
                cost_wl += d_wl
                cost_d  += d_d
                cost_c  += d_c
                cost_proxy += d_proxy
                overlap_penalty_total += d_penalty
                accepts_swap += 1
            else:
                rejects_swap += 1

        else:
            # ── RESHAPE (soft macros only) ──────────────────────────────
            m = int(movable_soft_idx[rng.integers(0, movable_soft_idx.size)])
            w_old = float(macro_dims[m, 0]); h_old = float(macro_dims[m, 1])
            x_old = float(macro_coords[m, 0]); y_old = float(macro_coords[m, 1])

            new_x, new_y, new_w, new_h, in_canvas = _propose_reshape(
                state, m,
                cfg.reshape_log_aspect_sigma, cfg.reshape_max_aspect, rng,
            )
            if not in_canvas:
                rejects_canvas  += 1
                rejects_reshape += 1
                iterations += 1
                T *= alpha
                continue

            # Reshape moves both the bottom-left (because we keep the
            # *centre* fixed) and the dims.  HPWL needs the shift delta
            # using the new bottom-left; density needs the reshape delta
            # (dims-aware).
            d_wl, n_aff = _hpwl_shift(
                m, new_x, new_y,
                macro_coords, macro_dims,
                state.port_coords, state.num_macros,
                state.macro_net_ids, state.macro_net_offsets,
                state.net_pin_owners, state.net_pin_offsets,
                state.net_weights, net_bbox,
                aff, nbb,
            )
            d_d, d_c, n_cells = _d_reshape(
                m, new_x, new_y, new_w, new_h,
                macro_coords, macro_dims,
                bw, bh, grid_rows, grid_cols,
                density_grid, crs, ccs, nds,
            )
            d_proxy = _proxy_cost(d_wl, d_d, d_c, cfg)

            # Soft macros do not occupy the integer spatial grid, so no
            # overlap-penalty term and no Phase-3 rejection.
            d_total = d_proxy

            accept = False
            if d_total <= 0.0:
                accept = True
            else:
                p = math.exp(-d_total / T) if T > 0.0 else 0.0
                if rng.random() < p:
                    accept = True

            if accept:
                _commit_hpwl(aff, nbb, n_aff, net_bbox)
                _commit_d(crs, ccs, nds, n_cells, density_grid)
                macro_coords[m, 0] = new_x
                macro_coords[m, 1] = new_y
                macro_dims[m, 0]   = new_w
                macro_dims[m, 1]   = new_h
                cost_wl += d_wl
                cost_d  += d_d
                cost_c  += d_c
                cost_proxy += d_proxy
                accepts_reshape += 1
            else:
                rejects_reshape += 1

        # ── Cooling step (geometric) ────────────────────────────────────
        if T > T_end:
            T *= alpha
            if T < T_end:
                T = T_end

        iterations += 1

        # ── Periodic spatial-grid refresh + selector rebuild ────────────
        if (iterations % cfg.grid_refresh_iters) == 0:
            stamp_all_hard_macros(state)
        if (iterations % cfg.selector_refresh_iters) == 0:
            per_macro_overlap_counts_njit(
                state.spatial_grid,
                state.macro_coords, state.macro_dims,
                state.macro_is_hard,
                state.num_macros,
                state.grid_bin_width, state.grid_bin_height,
                scratch.overlap_counts,
            )
            refresh_selector_weights_njit(
                state.macro_fixed, state.macro_is_hard,
                state.macro_net_ids, state.macro_net_offsets,
                state.net_bbox, state.net_weights,
                scratch.overlap_counts,
                cfg.selector_base_weight,
                cfg.selector_overlap_weight,
                cfg.selector_netlen_weight,
                scratch.selector_weights,
            )
            _rebuild_cumsum(scratch.selector_weights, scratch.selector_cumsum)

        if cfg.log_progress_every_iters > 0 and (iterations % cfg.log_progress_every_iters) == 0:
            print(
                f"[worker seed={cfg.seed}] iter={iterations} "
                f"T={T:.4g} cost={cost_proxy:.3f} "
                f"acc={accepts_shift + accepts_swap + accepts_reshape} "
                f"rej={rejects_shift + rejects_swap + rejects_reshape}"
            )

    elapsed = time.perf_counter() - start_time

    # ── Stage 8a: deterministic post-MCMC legalisation sweep ────────────
    # The MCMC's bin-based foreign-cell signal is approximate (last-
    # writer-wins integer grid); on dense benchmarks the loop can end
    # with a handful of residual geometric overlaps that the signal
    # never saw.  Run the exact-arithmetic sweep here to convert
    # "MCMC near-legal" into "exact legal" before reporting.
    sweep_initial = sweep_final = sweep_moves = sweep_passes = 0
    if cfg.enable_legalization_sweep and state.num_hard_macros > 0:
        sweep_initial, sweep_final, sweep_moves, sweep_passes = (
            _greedy_legalize_sweep(
                state,
                max_passes=cfg.legalization_sweep_max_passes,
                max_radius_steps=cfg.legalization_sweep_max_radius,
                step_scale=cfg.legalization_sweep_step_scale,
                tolerance=1e-6,
            )
        )
        if cfg.log_progress_every_iters > 0 and sweep_initial > 0:
            print(
                f"[worker seed={cfg.seed}] greedy legalisation sweep: "
                f"{sweep_initial} → {sweep_final} overlap pairs in "
                f"{sweep_passes} passes ({sweep_moves} moves)"
            )

    # ── Stage 8b: final validity sweep ──────────────────────────────────
    if cfg.enable_final_validity_check:
        valid, violations = _final_validity_check(
            state,
            tolerance=1e-6,
            pairwise_check_cap=cfg.pairwise_check_cap,
        )
    else:
        valid, violations = True, []

    # Re-derive global costs from scratch to defeat any incremental
    # drift before reporting; this is cheap (O(N + cells)).
    fe.populate_net_bbox_njit(
        state.net_pin_owners, state.net_pin_offsets,
        state.macro_coords, state.macro_dims,
        state.port_coords, state.num_macros,
        state.num_nets, state.net_bbox,
    )
    cost_wl_final = float(fe.compute_total_hpwl_njit(
        state.net_pin_owners, state.net_pin_offsets,
        state.net_weights,
        state.macro_coords, state.macro_dims,
        state.port_coords, state.num_macros,
        state.num_nets,
    ))
    # Rebuild density grid from scratch as well.
    density_grid.fill(0.0)
    fe.compute_density_grid_njit(
        state.macro_coords, state.macro_dims,
        state.num_macros,
        state.grid_bin_width, state.grid_bin_height,
        state.grid_num_rows, state.grid_num_cols,
        density_grid,
    )
    cost_d_final = float(fe.density_cost_from_grid_njit(density_grid))
    cost_c_final = float(fe.congestion_cost_from_grid_njit(density_grid))
    cost_proxy_final = _proxy_cost(cost_wl_final, cost_d_final, cost_c_final, cfg)

    accepts_total = accepts_shift + accepts_swap + accepts_reshape
    rejects_total = rejects_shift + rejects_swap + rejects_reshape

    return WorkerResult(
        seed=int(cfg.seed),
        state=state,
        grasp_report=grasp_report,
        valid=valid,
        violations=violations,
        cost_wirelength=cost_wl_final,
        cost_density=cost_d_final,
        cost_congestion=cost_c_final,
        cost_proxy=cost_proxy_final,
        cost_overlap_penalty=overlap_penalty_total,
        initial_cost_proxy=initial_cost_proxy,
        initial_temperature=T0,
        final_temperature=T,
        cooling_alpha=alpha,
        iterations=iterations,
        elapsed_seconds=elapsed,
        accepts_total=accepts_total,
        rejects_total=rejects_total,
        accepts_shift=accepts_shift, rejects_shift=rejects_shift,
        accepts_swap =accepts_swap,  rejects_swap =rejects_swap,
        accepts_reshape=accepts_reshape, rejects_reshape=rejects_reshape,
        rejects_canvas=rejects_canvas,
        rejects_legalization=rejects_legalization,
        legalization_sweep_initial_overlaps=sweep_initial,
        legalization_sweep_final_overlaps=sweep_final,
        legalization_sweep_moves=sweep_moves,
        legalization_sweep_passes=sweep_passes,
    )


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ 10. Self-test (synthetic + ibm01)                                      ║
# ╚════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":  # pragma: no cover
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the MCMC worker on a benchmark layout.",
    )
    parser.add_argument("--benchmark", "-b", default="ibm01")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--timeout", "-t", type=float, default=10.0)
    parser.add_argument("--no-grasp", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Skip the benchmark load and run a tiny synthetic test.")
    args = parser.parse_args()

    if args.quick:
        # Synthetic smoke test: a 4-macro layout on a 100×80 canvas.
        # Construct the PlacementState directly so we don't depend on torch.
        from state import (
            PlacementState as _PS,
            build_spatial_grid,
            build_csr_netlist,
        )
        canvas_w, canvas_h = 100.0, 80.0
        coords_ll = np.array([
            [5.0, 5.0], [40.0, 5.0], [5.0, 40.0], [40.0, 40.0],
        ], dtype=np.float64)
        dims = np.array([
            [20.0, 15.0], [20.0, 15.0], [20.0, 15.0], [20.0, 15.0],
        ], dtype=np.float64)
        fixed   = np.array([False, False, False, False])
        is_hard = np.array([True, True, True, True])
        ports = np.zeros((0, 2), dtype=np.float64)
        # Two simple nets connecting macros (0-1) and (2-3).
        net_pin_owners, net_pin_offsets, mn_ids, mn_off = build_csr_netlist(
            [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)],
            num_macros=4, num_total_nodes=4,
        )
        bin_w = bin_h = 2.0
        grid, gr, gc = build_spatial_grid(canvas_w, canvas_h, bin_w, bin_h)
        st = _PS(
            name="synthetic-4macro",
            canvas_width=canvas_w, canvas_height=canvas_h,
            num_macros=4, num_hard_macros=4, num_soft_macros=0,
            macro_coords=coords_ll.copy(),
            macro_dims=dims.copy(),
            macro_fixed=fixed,
            macro_is_hard=is_hard,
            macro_names=[f"m{i}" for i in range(4)],
            num_ports=0,
            port_coords=ports,
            num_nets=2,
            net_pin_owners=net_pin_owners,
            net_pin_offsets=net_pin_offsets,
            net_weights=np.ones(2, dtype=np.float64),
            macro_net_ids=mn_ids,
            macro_net_offsets=mn_off,
            net_bbox=np.zeros((2, 4), dtype=np.float64),
            grid_bin_width=bin_w, grid_bin_height=bin_h,
            grid_num_cols=gc, grid_num_rows=gr,
            spatial_grid=grid,
            bench_grid_rows=gr, bench_grid_cols=gc,
            inv_bin_width=1.0 / bin_w, inv_bin_height=1.0 / bin_h,
        )
        cfg = WorkerConfig(
            seed=args.seed,
            time_budget_seconds=2.0,
            enable_grasp=False,
            warmup_iters=100,
            cooling_recalibrate_iters=1_000,
            grid_refresh_iters=2_000,
            selector_refresh_iters=500,
        )
        res = run_worker(st, cfg, copy_state=True)
        print("=== Synthetic smoke test ===")
        print(f"valid={res.valid}  iterations={res.iterations}  "
              f"cost_proxy {res.initial_cost_proxy:.3f} → {res.cost_proxy:.3f}")
        print(f"accepts shift/swap/reshape = "
              f"{res.accepts_shift}/{res.accepts_swap}/{res.accepts_reshape}")
        print(f"rejects shift/swap/reshape = "
              f"{res.rejects_shift}/{res.rejects_swap}/{res.rejects_reshape}")
        print(f"rejects canvas/legalization = {res.rejects_canvas}/{res.rejects_legalization}")
        print(f"violations: {res.violations}")
        sys.exit(0 if res.valid else 1)

    # Full benchmark path.  Prefer the cached ``.pt`` so we avoid the
    # ``absl`` dependency of the live ``macro_place.loader``.  Load the
    # ``Benchmark`` dataclass via importlib so ``macro_place/__init__.py``
    # is not eagerly imported (it would also pull ``absl``).
    try:
        import importlib.util as _ilu
        import os
        bm_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "macro_place", "benchmark.py",
        ))
        _spec = _ilu.spec_from_file_location("_mpb", bm_path)
        _bm_mod = _ilu.module_from_spec(_spec)
        assert _spec is not None and _spec.loader is not None
        _spec.loader.exec_module(_bm_mod)
        Benchmark = _bm_mod.Benchmark
        from state import build_state
    except Exception as exc:  # pragma: no cover
        print(f"[worker] cannot load harness: {exc}")
        sys.exit(2)

    cached_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "benchmarks", "processed", "public",
        f"{args.benchmark}.pt",
    )
    cached_path = os.path.abspath(cached_path)
    if not os.path.exists(cached_path):
        print(f"[worker] cached benchmark not found at {cached_path}")
        sys.exit(2)
    print(f"[worker] loading cached benchmark from {cached_path}")
    bench = Benchmark.load(cached_path)
    base_state = build_state(bench)
    cfg = WorkerConfig(
        seed=args.seed,
        time_budget_seconds=args.timeout,
        enable_grasp=not args.no_grasp,
        log_progress_every_iters=50_000,
    )
    print(f"[worker] benchmark={args.benchmark} seed={args.seed} "
          f"timeout={args.timeout}s")
    res = run_worker(base_state, cfg, copy_state=True)
    print("=== Worker result ===")
    print(f"  iterations       = {res.iterations:,}")
    print(f"  elapsed_seconds  = {res.elapsed_seconds:.2f}")
    print(f"  iters/sec        = {res.iterations / max(res.elapsed_seconds, 1e-9):,.0f}")
    print(f"  T0 → T_end       = {res.initial_temperature:.4g} → "
          f"{res.final_temperature:.4g} (alpha={res.cooling_alpha:.6f})")
    print(f"  cost_proxy       = {res.initial_cost_proxy:.3f} → {res.cost_proxy:.3f} "
          f"(Δ={res.cost_proxy - res.initial_cost_proxy:+.3f})")
    print(f"  WL/D/C           = {res.cost_wirelength:.3f} / "
          f"{res.cost_density:.6f} / {res.cost_congestion:.6f}")
    print(f"  accepts (s/w/r)  = {res.accepts_shift}/{res.accepts_swap}/{res.accepts_reshape}")
    print(f"  rejects (s/w/r)  = {res.rejects_shift}/{res.rejects_swap}/{res.rejects_reshape}")
    print(f"  rejects cv/lg    = {res.rejects_canvas}/{res.rejects_legalization}")
    print(f"  overlap_penalty  = {res.cost_overlap_penalty:+.4f}")
    print(f"  valid            = {res.valid}")
    if res.violations:
        print("  violations:")
        for v in res.violations:
            print(f"    - {v}")
    sys.exit(0 if res.valid else 1)
