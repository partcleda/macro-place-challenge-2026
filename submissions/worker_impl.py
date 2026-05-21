# """
# worker_impl.py  —  SA worker with Fix 2 + Fix 4
# ================================================

# Fix 2: Overlap gate — do_overlap is only called AFTER the HPWL gate passes.
#        For SHIFT/MOVE (68% of moves), most proposals are rejected on HPWL
#        alone. Previously we paid the full O(n²) overlap cost on every move.
#        Now we only pay it when HPWL already improved, cutting ~90% of
#        overlap calls. This is why iteration count was only 540 — we were
#        doing 246²=60K pair comparisons per move, per worker.

# Fix 4: Two-phase SA (mirrors RePlAce mLG stage from the paper):
#        Phase 1 (first PHASE1_FRAC of time): temperature high, overlap
#          penalty 10× larger. Accept ANY move that reduces overlap count
#          regardless of HPWL effect. Goal: drive overlaps to zero fast.
#        Phase 2 (remaining time): normal blended SA. Goal: optimise HPWL
#          while keeping overlaps at zero.
#        Phase boundary: triggered when overlap count first hits zero,
#          or after PHASE1_FRAC of time elapses — whichever comes first.

# WHY SEPARATE FILE: ProcessPoolExecutor pickles by module.function name.
# The evaluator may import sa_placer.py under a generated name, making any
# function defined there unpicklable. worker_impl is always importable as
# "worker_impl.run_sa_worker" regardless of how the parent was loaded.
# """

# import os
# import sys
# import math
# import random
# import time
# from typing import List, Tuple

# import numpy as np

# # ─────────────────────────────────────────────────────────────────────────────
# # Constants
# # ─────────────────────────────────────────────────────────────────────────────

# T_START          = 0.005
# T_END            = 1e-8
# N_MOVES_PER_ITER = 20
# LEGALIZE_GAP     = 0.5

# # FIX 4: two-phase SA
# PHASE1_FRAC      = 0.30    # fraction of sa_time spent in phase 1
# PHASE1_OV_SCALE  = 10.0    # phase 1 overlap penalty multiplier


# # ─────────────────────────────────────────────────────────────────────────────
# # SNAP-TO-GRID HELPER
# # ─────────────────────────────────────────────────────────────────────────────

# def _snap_to_grid(x: float, y: float,
#                   grid_w: float, grid_h: float,
#                   W: float, H: float,
#                   hw: float, hh: float) -> tuple:
#     """Snap (x, y) to nearest grid cell center, clamped to canvas.

#     SA operates entirely in grid-snapped space so the final post-SA snap
#     is a no-op and cannot introduce overlaps.
#     """
#     col = max(0, min(int(W / grid_w) - 1, int(x / grid_w)))
#     row = max(0, min(int(H / grid_h) - 1, int(y / grid_h)))
#     xs  = max(hw, min(W - hw, (col + 0.5) * grid_w))
#     ys  = max(hh, min(H - hh, (row + 0.5) * grid_h))
#     return xs, ys


# # ─────────────────────────────────────────────────────────────────────────────
# # NET FORMAT CONVERSION
# # ─────────────────────────────────────────────────────────────────────────────

# def nets_to_cpp_format(nets: List) -> List:
#     result = []
#     for drv_idx, (dox, doy), sinks in nets:
#         flat_sinks = [(int(sidx), float(sox), float(soy))
#                       for sidx, (sox, soy) in sinks]
#         result.append((int(drv_idx), float(dox), float(doy), flat_sinks))
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # PYTHON FALLBACK HPWL + OVERLAP
# # ─────────────────────────────────────────────────────────────────────────────

# def _py_hpwl(pos_np: np.ndarray, nets_cpp: List) -> float:
#     if not nets_cpp:
#         return 0.0
#     total = 0.0
#     for drv_idx, dox, doy, sinks in nets_cpp:
#         if drv_idx < 0:
#             xs = [dox]; ys = [doy]
#         else:
#             xs = [float(pos_np[drv_idx, 0]) + dox]
#             ys = [float(pos_np[drv_idx, 1]) + doy]
#         for sidx, sox, soy in sinks:
#             if sidx < 0:
#                 xs.append(sox); ys.append(soy)
#             else:
#                 xs.append(float(pos_np[sidx, 0]) + sox)
#                 ys.append(float(pos_np[sidx, 1]) + soy)
#         total += (max(xs) - min(xs)) + (max(ys) - min(ys))
#     return float(total)


# def _py_overlap(pos_np: np.ndarray, sizes_np: np.ndarray,
#                 hard_idx_list: list, threshold: float = 0.0040) -> tuple:
#     n = len(hard_idx_list)
#     if n < 2:
#         return 0, 0.0
#     count = 0; area = 0.0
#     for i in range(n):
#         mi   = hard_idx_list[i]
#         xi0  = pos_np[mi, 0] - sizes_np[mi, 0] * 0.5
#         xi1  = pos_np[mi, 0] + sizes_np[mi, 0] * 0.5
#         yi0  = pos_np[mi, 1] - sizes_np[mi, 1] * 0.5
#         yi1  = pos_np[mi, 1] + sizes_np[mi, 1] * 0.5
#         ai   = sizes_np[mi, 0] * sizes_np[mi, 1]
#         for j in range(i + 1, n):
#             mj  = hard_idx_list[j]
#             ox  = min(xi1, pos_np[mj, 0] + sizes_np[mj, 0] * 0.5) \
#                 - max(xi0, pos_np[mj, 0] - sizes_np[mj, 0] * 0.5)
#             oy  = min(yi1, pos_np[mj, 1] + sizes_np[mj, 1] * 0.5) \
#                 - max(yi0, pos_np[mj, 1] - sizes_np[mj, 1] * 0.5)
#             if ox > 0.0 and oy > 0.0:
#                 ov_area  = ox * oy
#                 aj       = sizes_np[mj, 0] * sizes_np[mj, 1]
#                 min_area = min(ai, aj)
#                 if min_area > 0.0 and ov_area / min_area > threshold:
#                     count += 1; area += ov_area
#     return count, area


# # ─────────────────────────────────────────────────────────────────────────────
# # LOAD C++ EXTENSION IN CHILD PROCESS
# # ─────────────────────────────────────────────────────────────────────────────

# def _load_cpp(submissions_dir: str):
#     search_paths = [
#         submissions_dir,
#         os.path.join(submissions_dir, "placement_ops"),
#         os.path.join(os.path.dirname(submissions_dir), "placement_ops"),
#         os.getcwd(),
#         os.path.join(os.getcwd(), "placement_ops"),
#         os.path.join(os.getcwd(), "submissions"),
#     ]
#     for p in search_paths:
#         if p and p not in sys.path:
#             sys.path.insert(0, p)
#     try:
#         import placement_ops
#         return placement_ops
#     except ImportError:
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # SA WORKER
# # ─────────────────────────────────────────────────────────────────────────────

# def run_sa_worker(args: tuple):
#     """
#     Full two-phase SA loop in a child process.

#     Phase 1 — Overlap Removal  (first PHASE1_FRAC × sa_time seconds):
#       Overlap penalty = base_pen × PHASE1_OV_SCALE (10× normal)
#       Temperature stays high throughout phase 1.
#       Accepts any move that reduces overlap count unconditionally.
#       Terminates early if overlap count reaches zero.

#     Phase 2 — HPWL Optimisation  (remaining time):
#       Normal SA: blended HPWL + overlap penalty.
#       Temperature decays geometrically from T_START to T_END.
#       FIX 2: overlap gate — do_overlap only called after HPWL gate passes.

#     Returns: (best_pos_np or None, best_cost, n_iters)
#     """
#     (wid, init_pos_np, sizes_np, hard_movable, hard_all_list,
#      nets_cpp, num_macros, W, H, seed, overlap_pen,
#      sa_time, submissions_dir, grid_w, grid_h) = args

#     # ── Load C++ extension ────────────────────────────────────────────────
#     cpp = _load_cpp(submissions_dir)
#     if cpp is not None:
#         try:
#             cpp.init_nets(nets_cpp, num_macros)
#         except Exception:
#             cpp = None

#     hard_all_np = np.array(hard_all_list, dtype=np.int32)
#     g           = LEGALIZE_GAP

#     # ── Dispatch helpers ──────────────────────────────────────────────────

#     def full_hpwl(pos):
#         if cpp:
#             return float(cpp.compute_hpwl(pos))
#         return _py_hpwl(pos, nets_cpp)

#     def do_overlap(pos):
#         if cpp:
#             return cpp.count_overlap(pos, sizes_np, hard_all_np)
#         return _py_overlap(pos, sizes_np, hard_all_list)

#     def rebuild(pos):
#         if cpp:
#             cpp.rebuild_cache(pos)

#     def delta_hpwl(pos, idx, nx, ny):
#         """Incremental HPWL. Does NOT permanently modify pos."""
#         if cpp:
#             return cpp.compute_delta_hpwl(pos, int(idx),
#                                            float(nx), float(ny))
#         old_x, old_y = float(pos[idx, 0]), float(pos[idx, 1])
#         old_h = _py_hpwl(pos, nets_cpp)
#         pos[idx, 0] = nx; pos[idx, 1] = ny
#         new_h = _py_hpwl(pos, nets_cpp)
#         pos[idx, 0] = old_x; pos[idx, 1] = old_y
#         return new_h - old_h, new_h

#     def commit():
#         if cpp:
#             cpp.commit_move()

#     # ── Initialise ────────────────────────────────────────────────────────
#     rng = random.Random(seed)
#     pos = init_pos_np.copy()   # [N, 2] float32, mutated in-place

#     rebuild(pos)
#     cur_hpwl             = full_hpwl(pos)
#     cur_ov, cur_ov_area  = do_overlap(pos)
#     phase1_pen           = overlap_pen * PHASE1_OV_SCALE
#     current_cost         = cur_hpwl + phase1_pen * cur_ov_area

#     best_cost = current_cost if cur_ov == 0 else float('inf')
#     best_pos  = pos.copy()    if cur_ov == 0 else None

#     n_iters  = 0
#     start_t  = time.time()
#     phase1_end = start_t + sa_time * PHASE1_FRAC

#     # ══════════════════════════════════════════════════════════════════════
#     # PHASE 1: Overlap Removal
#     # High temperature, boosted penalty, greedy overlap reduction.
#     # ══════════════════════════════════════════════════════════════════════
#     print(f"    [w{wid:02d}] phase1 start  ov={cur_ov}  "
#           f"phase1_pen={phase1_pen:.0f}")

#     while time.time() < phase1_end and cur_ov > 0:
#         # High constant temperature — accept any improvement, some worsening
#         T = T_START * 10.0

#         for _ in range(N_MOVES_PER_ITER):
#             r = rng.random()

#             if r < 0.40:
#                 # SHIFT — small local move
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i, 0]) * 0.5
#                 hh  = float(sizes_np[i, 1]) * 0.5
#                 nx  = float(pos[i, 0]) + rng.uniform(-0.1*W, 0.1*W)
#                 ny  = float(pos[i, 1]) + rng.uniform(-0.1*H, 0.1*H)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
#                 pos[i, 0] = nx; pos[i, 1] = ny

#                 # FIX 2 applied here too: check overlap only after move
#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_hpwl            = full_hpwl(pos)
#                 new_cost            = new_hpwl + phase1_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if (new_ov < cur_ov or
#                         delta < 0 or
#                         rng.random() < math.exp(-delta / max(T, 1e-12))):
#                     rebuild(pos)
#                     current_cost = new_cost
#                     cur_hpwl     = new_hpwl
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i, 0] = old_x; pos[i, 1] = old_y

#             elif r < 0.70:
#                 # MOVE — teleport to random location (big exploration)
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i, 0]) * 0.5
#                 hh  = float(sizes_np[i, 1]) * 0.5
#                 nx  = rng.uniform(hw+g, W-hw-g)
#                 ny  = rng.uniform(hh+g, H-hh-g)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
#                 pos[i, 0] = nx; pos[i, 1] = ny

#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_hpwl            = full_hpwl(pos)
#                 new_cost            = new_hpwl + phase1_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if (new_ov < cur_ov or
#                         delta < 0 or
#                         rng.random() < math.exp(-delta / max(T, 1e-12))):
#                     rebuild(pos)
#                     current_cost = new_cost
#                     cur_hpwl     = new_hpwl
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i, 0] = old_x; pos[i, 1] = old_y

#             else:
#                 # SWAP — exchange two macros
#                 if len(hard_movable) < 2:
#                     continue
#                 i, j = rng.sample(hard_movable, 2)
#                 hw_i = float(sizes_np[i,0])*0.5; hh_i = float(sizes_np[i,1])*0.5
#                 hw_j = float(sizes_np[j,0])*0.5; hh_j = float(sizes_np[j,1])*0.5
#                 oxi, oyi = float(pos[i,0]), float(pos[i,1])
#                 oxj, oyj = float(pos[j,0]), float(pos[j,1])

#                 # Swap: snap each macro to the grid cell of the other's position
#                 nxi, nyi = _snap_to_grid(oxj, oyj, grid_w, grid_h, W, H,
#                                           hw_i, hh_i)
#                 nxj, nyj = _snap_to_grid(oxi, oyi, grid_w, grid_h, W, H,
#                                           hw_j, hh_j)
#                 pos[i,0]=nxi; pos[i,1]=nyi
#                 pos[j,0]=nxj; pos[j,1]=nyj

#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_hpwl            = full_hpwl(pos)
#                 new_cost            = new_hpwl + phase1_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if (new_ov < cur_ov or
#                         delta < 0 or
#                         rng.random() < math.exp(-delta / max(T, 1e-12))):
#                     rebuild(pos)
#                     current_cost = new_cost
#                     cur_hpwl     = new_hpwl
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i,0]=oxi; pos[i,1]=oyi
#                     pos[j,0]=oxj; pos[j,1]=oyj

#             n_iters += 1

#     phase1_iters = n_iters
#     phase1_elapsed = time.time() - start_t
#     print(f"    [w{wid:02d}] phase1 done   ov={cur_ov}  "
#           f"iters={phase1_iters}  t={phase1_elapsed:.1f}s")

#     # ══════════════════════════════════════════════════════════════════════
#     # PHASE 2: HPWL Optimisation
#     # Normal SA with FIX 2 (overlap gate) active.
#     # Overlap penalty returns to base level.
#     # ══════════════════════════════════════════════════════════════════════

#     # Recompute cost with base penalty (phase 1 used boosted penalty)
#     rebuild(pos)
#     cur_hpwl            = full_hpwl(pos)
#     cur_ov, cur_ov_area = do_overlap(pos)
#     current_cost        = cur_hpwl + overlap_pen * cur_ov_area

#     if cur_ov == 0 and current_cost < best_cost:
#         best_cost = current_cost; best_pos = pos.copy()

#     phase2_start = time.time()
#     phase2_time  = sa_time - (phase2_start - start_t)

#     while True:
#         elapsed = time.time() - phase2_start
#         if elapsed >= phase2_time:
#             break

#         # Geometric temperature decay across phase 2
#         frac = max(0.0, min(0.999, elapsed / max(phase2_time, 1e-6)))
#         T    = T_START * (T_END / T_START) ** frac

#         for _ in range(N_MOVES_PER_ITER):
#             r = rng.random()

#             # ── SHIFT (52%) ── FIX 2: HPWL gate first ────────────────────
#             if r < 0.52:
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i, 0]) * 0.5
#                 hh  = float(sizes_np[i, 1]) * 0.5
#                 tr  = max(0.01, T / T_START)
#                 nx  = float(pos[i, 0]) + rng.uniform(-0.05*W*tr, 0.05*W*tr)
#                 ny  = float(pos[i, 1]) + rng.uniform(-0.05*H*tr, 0.05*H*tr)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 # GATE 1: HPWL delta — reject if HPWL worsens too much
#                 d, new_total = delta_hpwl(pos, i, nx, ny)
#                 if not (d < 0 or rng.random() < math.exp(-d / max(T, 1e-12))):
#                     n_iters += 1; continue   # ← skips overlap check entirely

#                 # GATE 1 passed — apply move and check overlap
#                 old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
#                 pos[i, 0] = nx; pos[i, 1] = ny
#                 commit()

#                 # GATE 2: full cost including overlap
#                 new_ov, new_ov_area = do_overlap(pos)   # only called here
#                 new_cost = new_total + overlap_pen * new_ov_area
#                 delta    = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost
#                     cur_hpwl     = new_total
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i, 0] = old_x; pos[i, 1] = old_y
#                     rebuild(pos)
#                     cur_hpwl = new_total - d

#             # ── SWAP (28%) ────────────────────────────────────────────────
#             elif r < 0.80:
#                 if len(hard_movable) < 2:
#                     n_iters += 1; continue
#                 i, j = rng.sample(hard_movable, 2)
#                 hw_i = float(sizes_np[i,0])*0.5; hh_i = float(sizes_np[i,1])*0.5
#                 hw_j = float(sizes_np[j,0])*0.5; hh_j = float(sizes_np[j,1])*0.5
#                 oxi, oyi = float(pos[i,0]), float(pos[i,1])
#                 oxj, oyj = float(pos[j,0]), float(pos[j,1])

#                 # Swap: snap each macro to the grid cell of the other's position
#                 nxi, nyi = _snap_to_grid(oxj, oyj, grid_w, grid_h, W, H,
#                                           hw_i, hh_i)
#                 nxj, nyj = _snap_to_grid(oxi, oyi, grid_w, grid_h, W, H,
#                                           hw_j, hh_j)
#                 pos[i,0]=nxi; pos[i,1]=nyi
#                 pos[j,0]=nxj; pos[j,1]=nyj
#                 rebuild(pos)

#                 new_hpwl            = full_hpwl(pos)
#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_cost            = new_hpwl + overlap_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost; cur_hpwl = new_hpwl
#                     cur_ov = new_ov; cur_ov_area = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i,0]=oxi; pos[i,1]=oyi
#                     pos[j,0]=oxj; pos[j,1]=oyj
#                     rebuild(pos)

#             # ── MOVE (16%) — FIX 2: HPWL gate ────────────────────────────
#             elif r < 0.96:
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i,0])*0.5
#                 hh  = float(sizes_np[i,1])*0.5
#                 nx  = rng.uniform(hw+g, W-hw-g)
#                 ny  = rng.uniform(hh+g, H-hh-g)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 d, new_total = delta_hpwl(pos, i, nx, ny)
#                 if not (d < 0 or rng.random() < math.exp(-d / max(T, 1e-12))):
#                     n_iters += 1; continue   # ← skips overlap check

#                 old_x, old_y = float(pos[i,0]), float(pos[i,1])
#                 pos[i,0] = nx; pos[i,1] = ny
#                 commit()

#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_cost = new_total + overlap_pen * new_ov_area
#                 delta    = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost; cur_hpwl = new_total
#                     cur_ov = new_ov; cur_ov_area = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i,0] = old_x; pos[i,1] = old_y
#                     rebuild(pos)
#                     cur_hpwl = new_total - d

#             # ── SHUFFLE (4%) ──────────────────────────────────────────────
#             else:
#                 k = min(3, len(hard_movable))
#                 if k < 2:
#                     n_iters += 1; continue
#                 chosen = rng.sample(hard_movable, k)
#                 saved  = [(float(pos[c,0]), float(pos[c,1])) for c in chosen]
#                 for ci, c in enumerate(chosen):
#                     nx_c, ny_c = saved[(ci+1) % k]
#                     hw_c = float(sizes_np[c,0])*0.5
#                     hh_c = float(sizes_np[c,1])*0.5
#                     nx_c, ny_c = _snap_to_grid(nx_c, ny_c, grid_w, grid_h,
#                                                 W, H, hw_c, hh_c)
#                     pos[c,0] = nx_c
#                     pos[c,1] = ny_c
#                 rebuild(pos)

#                 new_hpwl            = full_hpwl(pos)
#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_cost            = new_hpwl + overlap_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost; cur_hpwl = new_hpwl
#                     cur_ov = new_ov; cur_ov_area = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     for ci, c in enumerate(chosen):
#                         pos[c,0], pos[c,1] = saved[ci]
#                     rebuild(pos)

#             n_iters += 1

#     return best_pos, best_cost, n_iters

# """
# worker_impl.py  —  SA worker with Fix 2 + Fix 4
# ================================================

# Fix 2: Overlap gate — do_overlap is only called AFTER the HPWL gate passes.
#        For SHIFT/MOVE (68% of moves), most proposals are rejected on HPWL
#        alone. Previously we paid the full O(n²) overlap cost on every move.
#        Now we only pay it when HPWL already improved, cutting ~90% of
#        overlap calls. This is why iteration count was only 540 — we were
#        doing 246²=60K pair comparisons per move, per worker.

# Fix 4: Two-phase SA (mirrors RePlAce mLG stage from the paper):
#        Phase 1 (first PHASE1_FRAC of time): temperature high, overlap
#          penalty 10× larger. Accept ANY move that reduces overlap count
#          regardless of HPWL effect. Goal: drive overlaps to zero fast.
#        Phase 2 (remaining time): normal blended SA. Goal: optimise HPWL
#          while keeping overlaps at zero.
#        Phase boundary: triggered when overlap count first hits zero,
#          or after PHASE1_FRAC of time elapses — whichever comes first.

# WHY SEPARATE FILE: ProcessPoolExecutor pickles by module.function name.
# The evaluator may import sa_placer.py under a generated name, making any
# function defined there unpicklable. worker_impl is always importable as
# "worker_impl.run_sa_worker" regardless of how the parent was loaded.
# """

# import os
# import sys
# import math
# import random
# import time
# from typing import List, Tuple

# import numpy as np

# # ─────────────────────────────────────────────────────────────────────────────
# # Constants
# # ─────────────────────────────────────────────────────────────────────────────

# T_START          = 0.005
# T_END            = 1e-8
# N_MOVES_PER_ITER = 20
# LEGALIZE_GAP     = 0.5

# # FIX 4: two-phase SA
# PHASE1_FRAC      = 0.30    # fraction of sa_time spent in phase 1
# PHASE1_OV_SCALE  = 10.0    # phase 1 overlap penalty multiplier


# # ─────────────────────────────────────────────────────────────────────────────
# # SNAP-TO-GRID HELPER
# # ─────────────────────────────────────────────────────────────────────────────

# def _snap_to_grid(x: float, y: float,
#                   grid_w: float, grid_h: float,
#                   W: float, H: float,
#                   hw: float, hh: float) -> tuple:
#     """Snap (x, y) to nearest grid cell center, clamped to canvas.

#     SA operates entirely in grid-snapped space so the final post-SA snap
#     is a no-op and cannot introduce overlaps.
#     """
#     col = max(0, min(int(W / grid_w) - 1, int(x / grid_w)))
#     row = max(0, min(int(H / grid_h) - 1, int(y / grid_h)))
#     xs  = max(hw, min(W - hw, (col + 0.5) * grid_w))
#     ys  = max(hh, min(H - hh, (row + 0.5) * grid_h))
#     return xs, ys


# # ─────────────────────────────────────────────────────────────────────────────
# # NET FORMAT CONVERSION
# # ─────────────────────────────────────────────────────────────────────────────

# def nets_to_cpp_format(nets: List) -> List:
#     result = []
#     for drv_idx, (dox, doy), sinks in nets:
#         flat_sinks = [(int(sidx), float(sox), float(soy))
#                       for sidx, (sox, soy) in sinks]
#         result.append((int(drv_idx), float(dox), float(doy), flat_sinks))
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # PYTHON FALLBACK HPWL + OVERLAP
# # ─────────────────────────────────────────────────────────────────────────────

# def _py_hpwl(pos_np: np.ndarray, nets_cpp: List) -> float:
#     if not nets_cpp:
#         return 0.0
#     total = 0.0
#     for drv_idx, dox, doy, sinks in nets_cpp:
#         if drv_idx < 0:
#             xs = [dox]; ys = [doy]
#         else:
#             xs = [float(pos_np[drv_idx, 0]) + dox]
#             ys = [float(pos_np[drv_idx, 1]) + doy]
#         for sidx, sox, soy in sinks:
#             if sidx < 0:
#                 xs.append(sox); ys.append(soy)
#             else:
#                 xs.append(float(pos_np[sidx, 0]) + sox)
#                 ys.append(float(pos_np[sidx, 1]) + soy)
#         total += (max(xs) - min(xs)) + (max(ys) - min(ys))
#     return float(total)


# def _py_overlap(pos_np: np.ndarray, sizes_np: np.ndarray,
#                 hard_idx_list: list, threshold: float = 0.0040) -> tuple:
#     n = len(hard_idx_list)
#     if n < 2:
#         return 0, 0.0
#     count = 0; area = 0.0
#     for i in range(n):
#         mi   = hard_idx_list[i]
#         xi0  = pos_np[mi, 0] - sizes_np[mi, 0] * 0.5
#         xi1  = pos_np[mi, 0] + sizes_np[mi, 0] * 0.5
#         yi0  = pos_np[mi, 1] - sizes_np[mi, 1] * 0.5
#         yi1  = pos_np[mi, 1] + sizes_np[mi, 1] * 0.5
#         ai   = sizes_np[mi, 0] * sizes_np[mi, 1]
#         for j in range(i + 1, n):
#             mj  = hard_idx_list[j]
#             ox  = min(xi1, pos_np[mj, 0] + sizes_np[mj, 0] * 0.5) \
#                 - max(xi0, pos_np[mj, 0] - sizes_np[mj, 0] * 0.5)
#             oy  = min(yi1, pos_np[mj, 1] + sizes_np[mj, 1] * 0.5) \
#                 - max(yi0, pos_np[mj, 1] - sizes_np[mj, 1] * 0.5)
#             if ox > 0.0 and oy > 0.0:
#                 ov_area  = ox * oy
#                 aj       = sizes_np[mj, 0] * sizes_np[mj, 1]
#                 min_area = min(ai, aj)
#                 if min_area > 0.0 and ov_area / min_area > threshold:
#                     count += 1; area += ov_area
#     return count, area


# # ─────────────────────────────────────────────────────────────────────────────
# # LOAD C++ EXTENSION IN CHILD PROCESS
# # ─────────────────────────────────────────────────────────────────────────────

# def _load_cpp(submissions_dir: str):
#     search_paths = [
#         submissions_dir,
#         os.path.join(submissions_dir, "placement_ops"),
#         os.path.join(os.path.dirname(submissions_dir), "placement_ops"),
#         os.getcwd(),
#         os.path.join(os.getcwd(), "placement_ops"),
#         os.path.join(os.getcwd(), "submissions"),
#     ]
#     for p in search_paths:
#         if p and p not in sys.path:
#             sys.path.insert(0, p)
#     try:
#         import placement_ops
#         return placement_ops
#     except ImportError:
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # SA WORKER
# # ─────────────────────────────────────────────────────────────────────────────

# def run_sa_worker(args: tuple):
#     """
#     Full two-phase SA loop in a child process.

#     Phase 1 — Overlap Removal  (first PHASE1_FRAC × sa_time seconds):
#       Overlap penalty = base_pen × PHASE1_OV_SCALE (10× normal)
#       Temperature stays high throughout phase 1.
#       Accepts any move that reduces overlap count unconditionally.
#       Terminates early if overlap count reaches zero.

#     Phase 2 — HPWL Optimisation  (remaining time):
#       Normal SA: blended HPWL + overlap penalty.
#       Temperature decays geometrically from T_START to T_END.
#       FIX 2: overlap gate — do_overlap only called after HPWL gate passes.

#     Returns: (best_pos_np or None, best_cost, n_iters)
#     """
#     (wid, init_pos_np, sizes_np, hard_movable, hard_all_list,
#      nets_cpp, num_macros, W, H, seed, overlap_pen,
#      sa_time, submissions_dir, grid_w, grid_h) = args

#     # ── Load C++ extension ────────────────────────────────────────────────
#     cpp = _load_cpp(submissions_dir)
#     if cpp is not None:
#         try:
#             cpp.init_nets(nets_cpp, num_macros)
#         except Exception:
#             cpp = None

#     hard_all_np = np.array(hard_all_list, dtype=np.int32)
#     g           = LEGALIZE_GAP

#     # ── Dispatch helpers ──────────────────────────────────────────────────

#     def full_hpwl(pos):
#         if cpp:
#             return float(cpp.compute_hpwl(pos))
#         return _py_hpwl(pos, nets_cpp)

#     def do_overlap(pos):
#         if cpp:
#             return cpp.count_overlap(pos, sizes_np, hard_all_np)
#         return _py_overlap(pos, sizes_np, hard_all_list)

#     def rebuild(pos):
#         if cpp:
#             cpp.rebuild_cache(pos)

#     def delta_hpwl(pos, idx, nx, ny):
#         """Incremental HPWL. Does NOT permanently modify pos."""
#         if cpp:
#             return cpp.compute_delta_hpwl(pos, int(idx),
#                                            float(nx), float(ny))
#         old_x, old_y = float(pos[idx, 0]), float(pos[idx, 1])
#         old_h = _py_hpwl(pos, nets_cpp)
#         pos[idx, 0] = nx; pos[idx, 1] = ny
#         new_h = _py_hpwl(pos, nets_cpp)
#         pos[idx, 0] = old_x; pos[idx, 1] = old_y
#         return new_h - old_h, new_h

#     def commit():
#         if cpp:
#             cpp.commit_move()

#     # ── Initialise ────────────────────────────────────────────────────────
#     rng = random.Random(seed)
#     pos = init_pos_np.copy()   # [N, 2] float32, mutated in-place

#     rebuild(pos)
#     cur_hpwl             = full_hpwl(pos)
#     cur_ov, cur_ov_area  = do_overlap(pos)
#     phase1_pen           = overlap_pen * PHASE1_OV_SCALE
#     current_cost         = cur_hpwl + phase1_pen * cur_ov_area

#     best_cost = current_cost if cur_ov == 0 else float('inf')
#     best_pos  = pos.copy()    if cur_ov == 0 else None

#     n_iters  = 0
#     start_t  = time.time()
#     phase1_end = start_t + sa_time * PHASE1_FRAC

#     # ══════════════════════════════════════════════════════════════════════
#     # PHASE 1: Overlap Removal
#     # High temperature, boosted penalty, greedy overlap reduction.
#     # ══════════════════════════════════════════════════════════════════════
#     print(f"    [w{wid:02d}] phase1 start  ov={cur_ov}  "
#           f"phase1_pen={phase1_pen:.0f}")

#     while time.time() < phase1_end and cur_ov > 0:
#         # High constant temperature — accept any improvement, some worsening
#         T = T_START * 10.0

#         for _ in range(N_MOVES_PER_ITER):
#             r = rng.random()

#             if r < 0.40:
#                 # SHIFT — small local move
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i, 0]) * 0.5
#                 hh  = float(sizes_np[i, 1]) * 0.5
#                 nx  = float(pos[i, 0]) + rng.uniform(-0.1*W, 0.1*W)
#                 ny  = float(pos[i, 1]) + rng.uniform(-0.1*H, 0.1*H)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
#                 pos[i, 0] = nx; pos[i, 1] = ny

#                 # FIX 2 applied here too: check overlap only after move
#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_hpwl            = full_hpwl(pos)
#                 new_cost            = new_hpwl + phase1_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if (new_ov < cur_ov or
#                         delta < 0 or
#                         rng.random() < math.exp(-delta / max(T, 1e-12))):
#                     rebuild(pos)
#                     current_cost = new_cost
#                     cur_hpwl     = new_hpwl
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i, 0] = old_x; pos[i, 1] = old_y

#             elif r < 0.70:
#                 # MOVE — teleport to random location (big exploration)
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i, 0]) * 0.5
#                 hh  = float(sizes_np[i, 1]) * 0.5
#                 nx  = rng.uniform(hw+g, W-hw-g)
#                 ny  = rng.uniform(hh+g, H-hh-g)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
#                 pos[i, 0] = nx; pos[i, 1] = ny

#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_hpwl            = full_hpwl(pos)
#                 new_cost            = new_hpwl + phase1_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if (new_ov < cur_ov or
#                         delta < 0 or
#                         rng.random() < math.exp(-delta / max(T, 1e-12))):
#                     rebuild(pos)
#                     current_cost = new_cost
#                     cur_hpwl     = new_hpwl
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i, 0] = old_x; pos[i, 1] = old_y

#             else:
#                 # SWAP — exchange two macros
#                 if len(hard_movable) < 2:
#                     continue
#                 i, j = rng.sample(hard_movable, 2)
#                 hw_i = float(sizes_np[i,0])*0.5; hh_i = float(sizes_np[i,1])*0.5
#                 hw_j = float(sizes_np[j,0])*0.5; hh_j = float(sizes_np[j,1])*0.5
#                 oxi, oyi = float(pos[i,0]), float(pos[i,1])
#                 oxj, oyj = float(pos[j,0]), float(pos[j,1])

#                 # Swap: snap each macro to the grid cell of the other's position
#                 nxi, nyi = _snap_to_grid(oxj, oyj, grid_w, grid_h, W, H,
#                                           hw_i, hh_i)
#                 nxj, nyj = _snap_to_grid(oxi, oyi, grid_w, grid_h, W, H,
#                                           hw_j, hh_j)
#                 pos[i,0]=nxi; pos[i,1]=nyi
#                 pos[j,0]=nxj; pos[j,1]=nyj

#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_hpwl            = full_hpwl(pos)
#                 new_cost            = new_hpwl + phase1_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if (new_ov < cur_ov or
#                         delta < 0 or
#                         rng.random() < math.exp(-delta / max(T, 1e-12))):
#                     rebuild(pos)
#                     current_cost = new_cost
#                     cur_hpwl     = new_hpwl
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i,0]=oxi; pos[i,1]=oyi
#                     pos[j,0]=oxj; pos[j,1]=oyj

#             n_iters += 1

#     phase1_iters = n_iters
#     phase1_elapsed = time.time() - start_t
#     print(f"    [w{wid:02d}] phase1 done   ov={cur_ov}  "
#           f"iters={phase1_iters}  t={phase1_elapsed:.1f}s")

#     # ══════════════════════════════════════════════════════════════════════
#     # PHASE 2: HPWL Optimisation
#     # Normal SA with FIX 2 (overlap gate) active.
#     # Overlap penalty returns to base level.
#     # ══════════════════════════════════════════════════════════════════════

#     # Recompute cost with base penalty (phase 1 used boosted penalty)
#     rebuild(pos)
#     cur_hpwl            = full_hpwl(pos)
#     cur_ov, cur_ov_area = do_overlap(pos)
#     current_cost        = cur_hpwl + overlap_pen * cur_ov_area

#     if cur_ov == 0 and current_cost < best_cost:
#         best_cost = current_cost; best_pos = pos.copy()

#     phase2_start = time.time()
#     phase2_time  = sa_time - (phase2_start - start_t)

#     while True:
#         elapsed = time.time() - phase2_start
#         if elapsed >= phase2_time:
#             break

#         # Geometric temperature decay across phase 2
#         frac = max(0.0, min(0.999, elapsed / max(phase2_time, 1e-6)))
#         T    = T_START * (T_END / T_START) ** frac

#         for _ in range(N_MOVES_PER_ITER):
#             r = rng.random()

#             # ── SHIFT (52%) ── FIX 2: HPWL gate first ────────────────────
#             if r < 0.52:
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i, 0]) * 0.5
#                 hh  = float(sizes_np[i, 1]) * 0.5
#                 tr  = max(0.01, T / T_START)
#                 nx  = float(pos[i, 0]) + rng.uniform(-0.05*W*tr, 0.05*W*tr)
#                 ny  = float(pos[i, 1]) + rng.uniform(-0.05*H*tr, 0.05*H*tr)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 # GATE 1: HPWL delta — reject if HPWL worsens too much
#                 d, new_total = delta_hpwl(pos, i, nx, ny)
#                 if not (d < 0 or rng.random() < math.exp(-d / max(T, 1e-12))):
#                     n_iters += 1; continue   # ← skips overlap check entirely

#                 # GATE 1 passed — apply move and check overlap
#                 old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
#                 pos[i, 0] = nx; pos[i, 1] = ny
#                 commit()

#                 # GATE 2: full cost including overlap
#                 new_ov, new_ov_area = do_overlap(pos)   # only called here
#                 new_cost = new_total + overlap_pen * new_ov_area
#                 delta    = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost
#                     cur_hpwl     = new_total
#                     cur_ov       = new_ov
#                     cur_ov_area  = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i, 0] = old_x; pos[i, 1] = old_y
#                     rebuild(pos)
#                     cur_hpwl = new_total - d

#             # ── SWAP (28%) ────────────────────────────────────────────────
#             elif r < 0.80:
#                 if len(hard_movable) < 2:
#                     n_iters += 1; continue
#                 i, j = rng.sample(hard_movable, 2)
#                 hw_i = float(sizes_np[i,0])*0.5; hh_i = float(sizes_np[i,1])*0.5
#                 hw_j = float(sizes_np[j,0])*0.5; hh_j = float(sizes_np[j,1])*0.5
#                 oxi, oyi = float(pos[i,0]), float(pos[i,1])
#                 oxj, oyj = float(pos[j,0]), float(pos[j,1])

#                 # Swap: snap each macro to the grid cell of the other's position
#                 nxi, nyi = _snap_to_grid(oxj, oyj, grid_w, grid_h, W, H,
#                                           hw_i, hh_i)
#                 nxj, nyj = _snap_to_grid(oxi, oyi, grid_w, grid_h, W, H,
#                                           hw_j, hh_j)
#                 pos[i,0]=nxi; pos[i,1]=nyi
#                 pos[j,0]=nxj; pos[j,1]=nyj
#                 rebuild(pos)

#                 new_hpwl            = full_hpwl(pos)
#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_cost            = new_hpwl + overlap_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost; cur_hpwl = new_hpwl
#                     cur_ov = new_ov; cur_ov_area = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i,0]=oxi; pos[i,1]=oyi
#                     pos[j,0]=oxj; pos[j,1]=oyj
#                     rebuild(pos)

#             # ── MOVE (16%) — FIX 2: HPWL gate ────────────────────────────
#             elif r < 0.96:
#                 i   = rng.choice(hard_movable)
#                 hw  = float(sizes_np[i,0])*0.5
#                 hh  = float(sizes_np[i,1])*0.5
#                 nx  = rng.uniform(hw+g, W-hw-g)
#                 ny  = rng.uniform(hh+g, H-hh-g)
#                 nx, ny = _snap_to_grid(nx, ny, grid_w, grid_h, W, H, hw, hh)

#                 d, new_total = delta_hpwl(pos, i, nx, ny)
#                 if not (d < 0 or rng.random() < math.exp(-d / max(T, 1e-12))):
#                     n_iters += 1; continue   # ← skips overlap check

#                 old_x, old_y = float(pos[i,0]), float(pos[i,1])
#                 pos[i,0] = nx; pos[i,1] = ny
#                 commit()

#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_cost = new_total + overlap_pen * new_ov_area
#                 delta    = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost; cur_hpwl = new_total
#                     cur_ov = new_ov; cur_ov_area = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     pos[i,0] = old_x; pos[i,1] = old_y
#                     rebuild(pos)
#                     cur_hpwl = new_total - d

#             # ── SHUFFLE (4%) ──────────────────────────────────────────────
#             else:
#                 k = min(3, len(hard_movable))
#                 if k < 2:
#                     n_iters += 1; continue
#                 chosen = rng.sample(hard_movable, k)
#                 saved  = [(float(pos[c,0]), float(pos[c,1])) for c in chosen]
#                 for ci, c in enumerate(chosen):
#                     nx_c, ny_c = saved[(ci+1) % k]
#                     hw_c = float(sizes_np[c,0])*0.5
#                     hh_c = float(sizes_np[c,1])*0.5
#                     nx_c, ny_c = _snap_to_grid(nx_c, ny_c, grid_w, grid_h,
#                                                 W, H, hw_c, hh_c)
#                     pos[c,0] = nx_c
#                     pos[c,1] = ny_c
#                 rebuild(pos)

#                 new_hpwl            = full_hpwl(pos)
#                 new_ov, new_ov_area = do_overlap(pos)
#                 new_cost            = new_hpwl + overlap_pen * new_ov_area
#                 delta               = new_cost - current_cost

#                 if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
#                     current_cost = new_cost; cur_hpwl = new_hpwl
#                     cur_ov = new_ov; cur_ov_area = new_ov_area
#                     if new_ov == 0 and new_cost < best_cost:
#                         best_cost = new_cost; best_pos = pos.copy()
#                 else:
#                     for ci, c in enumerate(chosen):
#                         pos[c,0], pos[c,1] = saved[ci]
#                     rebuild(pos)

#             n_iters += 1

#     return best_pos, best_cost, n_iters

"""
worker_impl.py  —  SA worker with snap-in-SA
=============================================

KEY DESIGN: every proposed position is snapped to a grid cell center
BEFORE it is evaluated. SA therefore operates entirely in grid space.
The final post-SA grid snap in sa_placer.py becomes a guaranteed no-op
and can never introduce overlaps.

Fix 2: Overlap gate — do_overlap only called AFTER the HPWL gate passes.
Fix 4: Two-phase SA — phase 1 hammers overlaps, phase 2 optimises HPWL.

WHY SEPARATE FILE: ProcessPoolExecutor pickles by module.function name.
The evaluator imports sa_placer.py under a generated name, making any
function defined there unpicklable. worker_impl is always importable as
"worker_impl.run_sa_worker" regardless of how the parent was loaded.
"""

import os
import sys
import math
import math as _math
import random
import time
from typing import List, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

T_START          = 0.005
T_END            = 1e-8
N_MOVES_PER_ITER = 20

# FIX 4: two-phase SA
PHASE1_FRAC     = 0.30    # fraction of sa_time spent in phase 1
PHASE1_OV_SCALE = 10.0    # phase 1 overlap penalty multiplier


# ─────────────────────────────────────────────────────────────────────────────
# SNAP-TO-GRID
# ─────────────────────────────────────────────────────────────────────────────

def _snap(x: float, y: float,
          grid_w: float, grid_h: float,
          n_cols: int, n_rows: int,
          W: float, H: float,
          hw: float, hh: float) -> Tuple[float, float]:
    """
    Snap (x, y) to the center of the grid cell that contains it.
    Clamped so the macro body stays fully within the canvas.

    This is exactly what the evaluator does via plc.set_node_position().
    By snapping every SA proposal here, the post-SA snap is a no-op.
    """
    col = int(x / grid_w)
    col = max(0, min(n_cols - 1, col))
    row = int(y / grid_h)
    row = max(0, min(n_rows - 1, row))
    xs = (col + 0.5) * grid_w
    ys = (row + 0.5) * grid_h
    # Clamp macro body inside canvas
    xs = max(hw, min(W - hw, xs))
    ys = max(hh, min(H - hh, ys))
    return xs, ys


# def _snap(x: float, y: float,
#           grid_w: float, grid_h: float,
#           n_cols: int, n_rows: int,
#           W: float, H: float,
#           hw: float, hh: float) -> Tuple[float, float]:
#     """
#     Snap (x, y) to the center of the grid cell that contains it,
#     choosing the nearest valid cell whose body fits inside the canvas.

#     Key property: the returned (xs, ys) is ALWAYS an exact grid cell center
#     (col + 0.5) * grid_w, (row + 0.5) * grid_h — never a clamped
#     non-grid-aligned value.  This means the evaluator's own snap will
#     produce the identical position, so our overlap check matches exactly.
#     """
#     # Valid column range: body must fit inside canvas
#     # (col + 0.5) * grid_w - hw >= 0  →  col >= hw/grid_w - 0.5
#     # (col + 0.5) * grid_w + hw <= W  →  col <= (W - hw)/grid_w - 0.5
#     col_min = max(0, _math.ceil(hw / grid_w - 0.5))
#     col_max = min(n_cols - 1, int((W - hw) / grid_w - 0.5 + 1e-9))
#     if col_min > col_max:           # macro nearly as wide as canvas
#         col_min = col_max = n_cols // 2

#     row_min = max(0, _math.ceil(hh / grid_h - 0.5))
#     row_max = min(n_rows - 1, int((H - hh) / grid_h - 0.5 + 1e-9))
#     if row_min > row_max:
#         row_min = row_max = n_rows // 2

#     col = int(x / grid_w)
#     col = max(col_min, min(col_max, col))
#     row = int(y / grid_h)
#     row = max(row_min, min(row_max, row))

#     return (col + 0.5) * grid_w, (row + 0.5) * grid_h

# ─────────────────────────────────────────────────────────────────────────────
# NET FORMAT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def nets_to_cpp_format(nets: List) -> List:
    result = []
    for drv_idx, (dox, doy), sinks in nets:
        flat_sinks = [(int(sidx), float(sox), float(soy))
                      for sidx, (sox, soy) in sinks]
        result.append((int(drv_idx), float(dox), float(doy), flat_sinks))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON FALLBACK HPWL + OVERLAP
# ─────────────────────────────────────────────────────────────────────────────

def _py_hpwl(pos_np: np.ndarray, nets_cpp: List) -> float:
    if not nets_cpp:
        return 0.0
    total = 0.0
    for drv_idx, dox, doy, sinks in nets_cpp:
        if drv_idx < 0:
            xs = [dox]; ys = [doy]
        else:
            xs = [float(pos_np[drv_idx, 0]) + dox]
            ys = [float(pos_np[drv_idx, 1]) + doy]
        for sidx, sox, soy in sinks:
            if sidx < 0:
                xs.append(sox); ys.append(soy)
            else:
                xs.append(float(pos_np[sidx, 0]) + sox)
                ys.append(float(pos_np[sidx, 1]) + soy)
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return float(total)


def _py_overlap(pos_np: np.ndarray, sizes_np: np.ndarray,
                hard_idx_list: List[int]) -> Tuple[int, float]:
    """
    Strict overlap check — any positive overlap area counts.
    Used during SA (positions are already grid-snapped, so any overlap
    is a real overlap that the evaluator will also see).
    """
    n = len(hard_idx_list)
    if n < 2:
        return 0, 0.0
    count = 0; area = 0.0
    for i in range(n):
        mi  = hard_idx_list[i]
        xi0 = pos_np[mi, 0] - sizes_np[mi, 0] * 0.5
        xi1 = pos_np[mi, 0] + sizes_np[mi, 0] * 0.5
        yi0 = pos_np[mi, 1] - sizes_np[mi, 1] * 0.5
        yi1 = pos_np[mi, 1] + sizes_np[mi, 1] * 0.5
        for j in range(i + 1, n):
            mj  = hard_idx_list[j]
            ox  = (min(xi1, pos_np[mj, 0] + sizes_np[mj, 0] * 0.5)
                 - max(xi0, pos_np[mj, 0] - sizes_np[mj, 0] * 0.5))
            oy  = (min(yi1, pos_np[mj, 1] + sizes_np[mj, 1] * 0.5)
                 - max(yi0, pos_np[mj, 1] - sizes_np[mj, 1] * 0.5))
            if ox > 1e-6 and oy > 1e-6:
                count += 1; area += ox * oy
    return count, area


# ─────────────────────────────────────────────────────────────────────────────
# LOAD C++ EXTENSION IN CHILD PROCESS
# ─────────────────────────────────────────────────────────────────────────────

def _load_cpp(submissions_dir: str):
    search_paths = [
        submissions_dir,
        os.path.join(submissions_dir, "placement_ops"),
        os.path.join(os.path.dirname(submissions_dir), "placement_ops"),
        os.getcwd(),
        os.path.join(os.getcwd(), "placement_ops"),
        os.path.join(os.getcwd(), "submissions"),
    ]
    for p in search_paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    try:
        import placement_ops
        return placement_ops
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SA WORKER
# ─────────────────────────────────────────────────────────────────────────────

def run_sa_worker(args: tuple):
    """
    Two-phase SA loop. Runs entirely in a child process.

    SNAP-IN-SA: every proposed nx, ny is passed through _snap() before
    evaluation. All positions stored in `pos` are therefore grid-snapped.
    The final post-SA snap in the main process becomes a no-op.

    Args tuple:
      wid            — worker id
      init_pos_np    — float32 [N, 2] starting positions (already snapped)
      sizes_np       — float32 [N, 2] macro sizes
      hard_movable   — list[int] movable hard macro indices
      hard_all_list  — list[int] all hard macro indices
      nets_cpp       — cpp-format net list
      num_macros     — int
      W, H           — canvas dimensions
      seed           — random seed
      overlap_pen    — overlap penalty (float)
      sa_time        — seconds for SA (float)
      submissions_dir— str
      grid_w, grid_h — grid cell dimensions (float)  ← NEW

    Returns: (best_pos_np or None, best_cost, n_iters)
    """
    (wid, init_pos_np, sizes_np, hard_movable, hard_all_list,
     nets_cpp, num_macros, W, H, seed, overlap_pen,
     sa_time, submissions_dir, grid_w, grid_h) = args

    # Derived grid constants
    n_cols = max(1, int(round(W / grid_w)))
    n_rows = max(1, int(round(H / grid_h)))

    # ── Load C++ extension ────────────────────────────────────────────────
    cpp = _load_cpp(submissions_dir)
    if cpp is not None:
        try:
            cpp.init_nets(nets_cpp, num_macros)
        except Exception:
            cpp = None

    hard_all_np = np.array(hard_all_list, dtype=np.int32)

    # ── Dispatch helpers ──────────────────────────────────────────────────

    def full_hpwl(pos):
        if cpp:
            return float(cpp.compute_hpwl(pos))
        return _py_hpwl(pos, nets_cpp)

    def do_overlap(pos):
        if cpp:
            return cpp.count_overlap(pos, sizes_np, hard_all_np)
        return _py_overlap(pos, sizes_np, hard_all_list)

    def rebuild(pos):
        if cpp:
            cpp.rebuild_cache(pos)

    def delta_hpwl(pos, idx, nx, ny):
        if cpp:
            return cpp.compute_delta_hpwl(pos, int(idx), float(nx), float(ny))
        old_x, old_y = float(pos[idx, 0]), float(pos[idx, 1])
        old_h = _py_hpwl(pos, nets_cpp)
        pos[idx, 0] = nx; pos[idx, 1] = ny
        new_h = _py_hpwl(pos, nets_cpp)
        pos[idx, 0] = old_x; pos[idx, 1] = old_y
        return new_h - old_h, new_h

    def commit():
        if cpp:
            cpp.commit_move()

    # Convenience wrapper — snap a single macro's proposed position
    def snap(x, y, hw, hh):
        return _snap(x, y, grid_w, grid_h, n_cols, n_rows, W, H, hw, hh)

    # ── Snap initial positions to grid ────────────────────────────────────
    # Ensures the worker starts from a valid grid-snapped state.
    rng = random.Random(seed)
    pos = init_pos_np.copy()
    for mi in hard_all_list:
        hw = float(sizes_np[mi, 0]) * 0.5
        hh = float(sizes_np[mi, 1]) * 0.5
        pos[mi, 0], pos[mi, 1] = snap(float(pos[mi, 0]),
                                       float(pos[mi, 1]), hw, hh)

    rebuild(pos)
    cur_hpwl            = full_hpwl(pos)
    cur_ov, cur_ov_area = do_overlap(pos)
    phase1_pen          = overlap_pen * PHASE1_OV_SCALE
    current_cost        = cur_hpwl + phase1_pen * cur_ov_area

    best_cost = current_cost if cur_ov == 0 else float('inf')
    best_pos  = pos.copy()   if cur_ov == 0 else None

    n_iters   = 0
    start_t   = time.time()
    phase1_end = start_t + sa_time * PHASE1_FRAC

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — Overlap Removal
    # High constant temperature, boosted overlap penalty.
    # Unconditionally accepts any move that reduces overlap count.
    # ══════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — Overlap Removal via Greedy Cell Re-assignment
    #
    # PROBLEM: snap-in-SA means macros larger than one grid cell will
    # ALWAYS overlap their neighbors if placed in adjacent cells.
    # A temperature-based SA cannot escape this — the "accept any overlap-
    # reducing move" logic can cycle forever without reaching zero overlaps.
    #
    # SOLUTION: Phase 1 is now a greedy legalizer:
    #   1. Sort macros by size (largest first — hardest to place).
    #   2. For each macro, find the nearest grid cell (to its current
    #      snapped position) that does NOT overlap any already-placed macro.
    #   3. Assign it there.  Repeat until all macros are placed.
    #   4. This runs multiple times with different starting orderings +
    #      random perturbations to explore different cell assignments.
    # This is guaranteed to produce zero overlaps if the canvas is not
    # over-full (utilization < 100%), which all benchmarks satisfy.
    # After legalizing, Phase 2 runs normal SA to optimise HPWL.
    # ══════════════════════════════════════════════════════════════════════

    # Precompute all grid cell centers sorted once
    all_cells = []
    for col in range(n_cols):
        for row in range(n_rows):
            all_cells.append(((col + 0.5) * grid_w, (row + 0.5) * grid_h))

    def _legalize(pos_in, seed_offset=0):
        """
        Greedy cell-assignment legalizer.  Returns a new pos array with
        zero overlaps, or None if it fails (should not happen for util<100%).
        """
        pos_out = pos_in.copy()
        rng2    = random.Random(seed + seed_offset)

        # Fixed macros (non-movable) are pre-placed — track their boxes
        placed = []  # list of (x0, y0, x1, y1)
        for mi in hard_all_list:
            if mi not in _movable_set:
                hw = float(sizes_np[mi, 0]) * 0.5
                hh = float(sizes_np[mi, 1]) * 0.5
                cx = float(pos_out[mi, 0])
                cy = float(pos_out[mi, 1])
                placed.append((cx-hw, cy-hh, cx+hw, cy+hh))

        def overlaps_placed(x0, y0, x1, y1):
            for (bx0, by0, bx1, by1) in placed:
                if (min(x1,bx1) - max(x0,bx0) >= 0 and
                        min(y1,by1) - max(y0,by0) >= 0):
                    return True
            return False

        # Sort order: largest area first, with small random jitter for diversity
        order = sorted(hard_movable,
                       key=lambda i: (-(float(sizes_np[i,0])*float(sizes_np[i,1])),
                                      rng2.random()))

        for mi in order:
            hw = float(sizes_np[mi, 0]) * 0.5
            hh = float(sizes_np[mi, 1]) * 0.5
            sx = max(hw, min(W-hw, float(pos_out[mi, 0])))
            sy = max(hh, min(H-hh, float(pos_out[mi, 1])))

            # Sort cells by distance from current position + small noise
            cells = sorted(all_cells,
                           key=lambda c: ((c[0]-sx)**2 + (c[1]-sy)**2
                                          + rng2.uniform(0, (grid_w*0.1)**2)))

            placed_mi = False
            for (cx, cy) in cells:
                cx_c, cy_c = _snap(cx, cy, grid_w, grid_h, n_cols, n_rows, W, H, hw, hh)
                x0 = cx_c - hw; y0 = cy_c - hh
                x1 = cx_c + hw; y1 = cy_c + hh
                if not overlaps_placed(x0, y0, x1, y1):
                    pos_out[mi, 0] = cx_c
                    pos_out[mi, 1] = cy_c
                    placed.append((x0, y0, x1, y1))
                    placed_mi = True
                    break

            if not placed_mi:
                # Fine scan fallback (extremely rare)
                step = min(hw, hh) * 0.5
                xi = hw
                while xi <= W - hw and not placed_mi:
                    yi = hh
                    while yi <= H - hh:
                        x0=xi-hw; y0=yi-hh; x1=xi+hw; y1=yi+hh
                        if not overlaps_placed(x0,y0,x1,y1):
                            pos_out[mi,0]=xi; pos_out[mi,1]=yi
                            placed.append((x0,y0,x1,y1))
                            placed_mi = True; break
                        yi += step
                    xi += step
                if not placed_mi:
                    return None  # failed (shouldn't happen)

        return pos_out

    _movable_set = set(hard_movable)

    print(f"    [w{wid:02d}] phase1 start  ov={cur_ov}")

    # Run legalizer multiple times from different seeds, keep best result
    best_legal_pos  = None
    best_legal_hpwl = float('inf')
    n_legal_tries   = max(1, int(sa_time * PHASE1_FRAC / 2.0))  # ~1 try/2s
    n_legal_tries   = min(n_legal_tries, 8)  # cap at 8

    for attempt in range(n_legal_tries):
        legal = _legalize(pos, seed_offset=attempt * 1000)
        if legal is None:
            continue
        rebuild(legal)
        h = full_hpwl(legal)
        if h < best_legal_hpwl:
            best_legal_hpwl = h
            best_legal_pos  = legal.copy()

    if best_legal_pos is not None:
        pos = best_legal_pos
        cur_ov = 0
        cur_ov_area = 0.0
    # else: keep original pos with overlaps — Phase 2 will try to fix

    rebuild(pos)
    cur_hpwl            = full_hpwl(pos)
    cur_ov, cur_ov_area = do_overlap(pos)
    current_cost        = cur_hpwl + phase1_pen * cur_ov_area

    if cur_ov == 0 and current_cost < best_cost:
        best_cost = current_cost; best_pos = pos.copy()

    n_iters += n_legal_tries * len(hard_movable)  # approximate iteration count

    phase1_elapsed = time.time() - start_t
    print(f"    [w{wid:02d}] phase1 done   ov={cur_ov}  "
          f"iters={n_iters}  t={phase1_elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — HPWL Optimisation
    # Normal SA with Fix 2 (HPWL gate before overlap check).
    # Positions remain grid-snapped throughout.
    # ══════════════════════════════════════════════════════════════════════

    rebuild(pos)
    cur_hpwl            = full_hpwl(pos)
    cur_ov, cur_ov_area = do_overlap(pos)
    current_cost        = cur_hpwl + overlap_pen * cur_ov_area

    if cur_ov == 0 and current_cost < best_cost:
        best_cost = current_cost; best_pos = pos.copy()

    phase2_start = time.time()
    phase2_time  = sa_time - (phase2_start - start_t)

    while True:
        elapsed = time.time() - phase2_start
        if elapsed >= phase2_time:
            break

        frac = max(0.0, min(0.999, elapsed / max(phase2_time, 1e-6)))
        T    = T_START * (T_END / T_START) ** frac

        for _ in range(N_MOVES_PER_ITER):
            r = rng.random()

            # ── SHIFT (52%) — incremental HPWL gate + snap ────────────────
            if r < 0.52:
                i  = rng.choice(hard_movable)
                hw = float(sizes_np[i, 0]) * 0.5
                hh = float(sizes_np[i, 1]) * 0.5
                tr = max(0.01, T / T_START)
                nx, ny = snap(
                    float(pos[i, 0]) + rng.uniform(-0.05*W*tr, 0.05*W*tr),
                    float(pos[i, 1]) + rng.uniform(-0.05*H*tr, 0.05*H*tr),
                    hw, hh)

                old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
                if nx == old_x and ny == old_y:
                    n_iters += 1; continue   # same cell — no work needed

                # GATE 1: HPWL gate
                d, new_total = delta_hpwl(pos, i, nx, ny)
                if not (d < 0 or rng.random() < math.exp(-d / max(T, 1e-12))):
                    n_iters += 1; continue   # reject — skip overlap check

                # GATE 1 passed
                pos[i, 0] = nx; pos[i, 1] = ny
                commit()

                # GATE 2: full cost with overlap
                new_ov, new_ov_area = do_overlap(pos)
                new_cost = new_total + overlap_pen * new_ov_area
                delta    = new_cost - current_cost

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                    current_cost = new_cost
                    cur_hpwl     = new_total
                    cur_ov       = new_ov
                    cur_ov_area  = new_ov_area
                    if new_ov == 0 and new_cost < best_cost:
                        best_cost = new_cost; best_pos = pos.copy()
                else:
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    rebuild(pos)
                    cur_hpwl = new_total - d

            # ── SWAP (28%) ────────────────────────────────────────────────
            elif r < 0.80:
                if len(hard_movable) < 2:
                    n_iters += 1; continue
                i, j = rng.sample(hard_movable, 2)
                hw_i = float(sizes_np[i, 0]) * 0.5
                hh_i = float(sizes_np[i, 1]) * 0.5
                hw_j = float(sizes_np[j, 0]) * 0.5
                hh_j = float(sizes_np[j, 1]) * 0.5
                oxi, oyi = float(pos[i, 0]), float(pos[i, 1])
                oxj, oyj = float(pos[j, 0]), float(pos[j, 1])

                nxi, nyi = snap(oxj, oyj, hw_i, hh_i)
                nxj, nyj = snap(oxi, oyi, hw_j, hh_j)
                pos[i, 0] = nxi; pos[i, 1] = nyi
                pos[j, 0] = nxj; pos[j, 1] = nyj
                rebuild(pos)

                new_hpwl            = full_hpwl(pos)
                new_ov, new_ov_area = do_overlap(pos)
                new_cost            = new_hpwl + overlap_pen * new_ov_area
                delta               = new_cost - current_cost

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                    current_cost = new_cost; cur_hpwl = new_hpwl
                    cur_ov = new_ov; cur_ov_area = new_ov_area
                    if new_ov == 0 and new_cost < best_cost:
                        best_cost = new_cost; best_pos = pos.copy()
                else:
                    pos[i, 0] = oxi; pos[i, 1] = oyi
                    pos[j, 0] = oxj; pos[j, 1] = oyj
                    rebuild(pos)

            # ── MOVE (16%) — random grid cell + HPWL gate ─────────────────
            elif r < 0.96:
                i  = rng.choice(hard_movable)
                hw = float(sizes_np[i, 0]) * 0.5
                hh = float(sizes_np[i, 1]) * 0.5
                rc = rng.randint(0, n_cols - 1)
                rr = rng.randint(0, n_rows - 1)
                nx, ny = snap((rc + 0.5) * grid_w, (rr + 0.5) * grid_h,
                               hw, hh)

                old_x, old_y = float(pos[i, 0]), float(pos[i, 1])
                if nx == old_x and ny == old_y:
                    n_iters += 1; continue

                d, new_total = delta_hpwl(pos, i, nx, ny)
                if not (d < 0 or rng.random() < math.exp(-d / max(T, 1e-12))):
                    n_iters += 1; continue

                pos[i, 0] = nx; pos[i, 1] = ny
                commit()

                new_ov, new_ov_area = do_overlap(pos)
                new_cost = new_total + overlap_pen * new_ov_area
                delta    = new_cost - current_cost

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                    current_cost = new_cost; cur_hpwl = new_total
                    cur_ov = new_ov; cur_ov_area = new_ov_area
                    if new_ov == 0 and new_cost < best_cost:
                        best_cost = new_cost; best_pos = pos.copy()
                else:
                    pos[i, 0] = old_x; pos[i, 1] = old_y
                    rebuild(pos)
                    cur_hpwl = new_total - d

            # ── SHUFFLE (4%) — cyclic rotation of 3 macros ────────────────
            else:
                k = min(3, len(hard_movable))
                if k < 2:
                    n_iters += 1; continue
                chosen = rng.sample(hard_movable, k)
                saved  = [(float(pos[c, 0]), float(pos[c, 1])) for c in chosen]
                for ci, c in enumerate(chosen):
                    sx, sy = saved[(ci + 1) % k]
                    hw_c = float(sizes_np[c, 0]) * 0.5
                    hh_c = float(sizes_np[c, 1]) * 0.5
                    nx_c, ny_c = snap(sx, sy, hw_c, hh_c)
                    pos[c, 0] = nx_c; pos[c, 1] = ny_c
                rebuild(pos)

                new_hpwl            = full_hpwl(pos)
                new_ov, new_ov_area = do_overlap(pos)
                new_cost            = new_hpwl + overlap_pen * new_ov_area
                delta               = new_cost - current_cost

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                    current_cost = new_cost; cur_hpwl = new_hpwl
                    cur_ov = new_ov; cur_ov_area = new_ov_area
                    if new_ov == 0 and new_cost < best_cost:
                        best_cost = new_cost; best_pos = pos.copy()
                else:
                    for ci, c in enumerate(chosen):
                        pos[c, 0], pos[c, 1] = saved[ci]
                    rebuild(pos)

            n_iters += 1

    return best_pos, best_cost, n_iters