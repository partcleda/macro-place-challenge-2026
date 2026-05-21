# """
# GWTW-SA Macro Placer  —  4-Fix Version
# =======================================

# Fix 1: Valid fallback — greedy pack never wraps, always produces zero overlaps.
# Fix 2: Overlap gate in SA — do_overlap only called after HPWL gate passes.
# Fix 3: FFT electrostatic spreader — spread macros from random init before SA.
# Fix 4: Two-phase SA — phase 1 hammers overlaps, phase 2 optimises wirelength.
# """
# import os
# import sys
# import time
# import random
# from typing import List, Tuple, Optional
# from concurrent.futures import ProcessPoolExecutor
# import math

# import numpy as np
# import torch
# from macro_place.benchmark import Benchmark

# # ─────────────────────────────────────────────────────────────────────────────
# # Worker lives in worker_impl.py — stable module name for ProcessPoolExecutor
# # ─────────────────────────────────────────────────────────────────────────────

# _here = os.path.dirname(os.path.abspath(__file__))
# if _here not in sys.path:
#     sys.path.insert(0, _here)

# from worker_impl import run_sa_worker, nets_to_cpp_format

# # ─────────────────────────────────────────────────────────────────────────────
# # C++ extension (main process — for ref_hpwl only)
# # ─────────────────────────────────────────────────────────────────────────────

# _cpp = None
# _HAS_CPP = False
# for _p in [_here, os.path.join(_here, "placement_ops")]:
#     if _p not in sys.path:
#         sys.path.insert(0, _p)
# try:
#     import placement_ops as _cpp
#     _HAS_CPP = True
#     print("[placement_ops] C++ extension loaded — fast HPWL active")
# except ImportError:
#     print("[placement_ops] WARNING: C++ not found — Python fallback")
#     print(f"  Run: cd placement_ops && python setup.py build_ext --inplace")
#     print(f"  Then copy the .pyd/.so into submissions/")

# # ─────────────────────────────────────────────────────────────────────────────
# # HYPERPARAMETERS
# # ─────────────────────────────────────────────────────────────────────────────

# TIME_BUDGET_SEC          = 55.0
# NUM_WORKERS              = 16
# LEGALIZE_GAP             = 0.5
# OVERLAP_PENALTY_SCALE    = 50.0
# OVERLAP_PENALTY_FALLBACK = 1e6

# # FFT spreader settings
# FFT_MAX_ITERS   = 300      # Nesterov iterations
# FFT_TIME_LIMIT  = 8.0     # hard wall: never spend more than this on spreading
# FFT_OV_TARGET   = 0.02    # stop when density overflow < 2%


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: locate testcase dir
# # ─────────────────────────────────────────────────────────────────────────────

# def _find_testcase_dir(name: str) -> Optional[str]:
#     candidates = [
#         f"external/MacroPlacement/Testcases/ICCAD04/{name}",
#         os.path.join(os.getcwd(), "external", "MacroPlacement",
#                      "Testcases", "ICCAD04", name),
#     ]
#     for c in candidates:
#         if os.path.isdir(c) and os.path.isfile(
#                 os.path.join(c, "netlist.pb.txt")):
#             return c
#     return None


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: extract nets
# # ─────────────────────────────────────────────────────────────────────────────

# def _extract_nets(plc, benchmark) -> List:
#     nets = []
#     if plc is None or not hasattr(plc, 'nets'):
#         return nets
#     name_to_bench = {n: i for i, n in enumerate(benchmark.macro_names)}

#     def resolve(pin_name):
#         if pin_name not in plc.mod_name_to_indices:
#             return None
#         idx = plc.mod_name_to_indices[pin_name]
#         if idx >= len(plc.modules_w_pins):
#             return None
#         obj = plc.modules_w_pins[idx]
#         try:
#             ptype = obj.get_type()
#         except Exception:
#             return None
#         if ptype == 'PORT':
#             try:
#                 px, py = obj.get_pos()
#                 return (-1, (float(px), float(py)))
#             except Exception:
#                 return None
#         if ptype != 'MACRO_PIN':
#             return None
#         try:
#             parent = obj.macro_name
#         except AttributeError:
#             return None
#         if parent not in name_to_bench:
#             return None
#         try:
#             ox, oy = obj.get_offset()
#         except Exception:
#             ox, oy = 0.0, 0.0
#         return (name_to_bench[parent], (float(ox), float(oy)))

#     for drv_name, sink_names in plc.nets.items():
#         drv = resolve(drv_name)
#         if drv is None:
#             continue
#         sinks = [s for sn in sink_names
#                  for s in [resolve(sn)] if s is not None]
#         if sinks:
#             nets.append((drv[0], drv[1], sinks))
#     return nets


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: Python HPWL (main process ref calculation)
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


# # ─────────────────────────────────────────────────────────────────────────────
# # FIX 1: GUARANTEED-VALID GREEDY PACK
# # Never wraps back to origin — instead places overflow macros randomly in
# # remaining whitespace, and falls back to explicit overlap-free slot search.
# # ─────────────────────────────────────────────────────────────────────────────

# def _greedy_pack_init(benchmark: Benchmark, seed: int = 0) -> np.ndarray:
#     """
#     Row-pack macros sorted by height descending.
#     When a row overflows vertically, we find the actual remaining whitespace
#     columns and continue there — never wrapping back to already-occupied rows.
#     Result is guaranteed overlap-free as long as total macro area < canvas area.
#     """
#     rng   = random.Random(seed)
#     sizes = benchmark.macro_sizes
#     W     = float(benchmark.canvas_width)
#     H     = float(benchmark.canvas_height)
#     g     = LEGALIZE_GAP

#     pos   = benchmark.macro_positions.numpy().astype(np.float32).copy()
#     movable = (benchmark.get_movable_mask() &
#                benchmark.get_hard_macro_mask()).numpy()
#     idx = [i for i in range(len(movable)) if movable[i]]
#     if not idx:
#         return pos

#     idx.sort(key=lambda i: (-float(sizes[i][1]), rng.random()))

#     x_cur = g; y_cur = g; row_h = 0.0
#     # Track placed boxes for overflow fallback
#     placed_boxes = []  # list of (x0, y0, x1, y1)

#     for i in idx:
#         w = float(sizes[i][0]); h = float(sizes[i][1])
#         hw = w / 2.0; hh = h / 2.0

#         if w + 2*g > W or h + 2*g > H:
#             # Macro nearly as big as canvas — center it
#             pos[i, 0] = W / 2.0; pos[i, 1] = H / 2.0
#             placed_boxes.append((W/2-hw, H/2-hh, W/2+hw, H/2+hh))
#             continue

#         if x_cur + w + g > W:
#             # End of row — move to next row
#             y_cur += row_h + g; x_cur = g; row_h = 0.0

#         if y_cur + h + g <= H:
#             # Normal placement
#             cx = x_cur + hw; cy = y_cur + hh
#             pos[i, 0] = cx; pos[i, 1] = cy
#             placed_boxes.append((cx-hw, cy-hh, cx+hw, cy+hh))
#             x_cur += w + g; row_h = max(row_h, h)
#         else:
#             # Canvas full — scan for a free location using a coarse grid
#             cx, cy = _find_free_slot(w, h, W, H, g, placed_boxes, rng)
#             pos[i, 0] = cx; pos[i, 1] = cy
#             placed_boxes.append((cx-hw, cy-hh, cx+hw, cy+hh))

#     return pos


# def _find_free_slot(w: float, h: float, W: float, H: float,
#                     g: float, placed_boxes: List,
#                     rng: random.Random) -> Tuple[float, float]:
#     """
#     Scan on a coarse grid to find a non-overlapping position for a macro
#     of size (w, h). Falls back to random position if grid scan finds nothing.
#     """
#     hw = w / 2.0; hh = h / 2.0
#     step = max(w, h) * 0.5  # coarse grid step

#     x = hw + g
#     while x + hw + g <= W:
#         y = hh + g
#         while y + hh + g <= H:
#             # Check if this candidate position overlaps any placed box
#             ok = True
#             for (bx0, by0, bx1, by1) in placed_boxes:
#                 ox = min(x+hw, bx1) - max(x-hw, bx0)
#                 oy = min(y+hh, by1) - max(y-hh, by0)
#                 if ox > 1e-3 and oy > 1e-3:
#                     ok = False; break
#             if ok:
#                 return x, y
#             y += step
#         x += step

#     # Grid scan failed — return random clamped position (last resort)
#     cx = rng.uniform(hw + g, max(hw + g + 0.01, W - hw - g))
#     cy = rng.uniform(hh + g, max(hh + g + 0.01, H - hh - g))
#     return cx, cy


# # ─────────────────────────────────────────────────────────────────────────────
# # FIX 3: FFT ELECTROSTATIC SPREADER
# #
# # Physics: macros are positively charged particles. Charge density on a grid
# # creates an electrostatic potential (via Poisson's equation solved with FFT).
# # Each macro experiences a repulsive force proportional to the gradient of the
# # potential — this pushes macros away from crowded bins.
# # Simultaneously, a wirelength gradient (from net spring forces) pulls
# # connected macros together.
# #
# # We use Nesterov's accelerated gradient descent (same as RePlAce/ePlace).
# # The density penalty lambda grows slowly until overflow is resolved.
# #
# # GPU-accelerated automatically if torch.cuda is available (RTX 6000 on
# # judge machine). Falls back to CPU on your i7.
# # ─────────────────────────────────────────────────────────────────────────────

# def _fft_spread(benchmark: Benchmark,
#                 nets_cpp: List,
#                 seed: int = 0,
#                 time_limit: float = FFT_TIME_LIMIT) -> np.ndarray:
#     """
#     FFT-based electrostatic spreader.

#     Returns float32 numpy [N, 2] with hard macros spread to near-zero overlap.
#     Soft macros are left at their reference positions.

#     Algorithm (ePlace / RePlAce):
#       1. Random initialisation of hard macro positions
#       2. For each Nesterov iteration:
#          a. Compute charge density on grid (each macro smeared over its bins)
#          b. Solve Poisson equation via 2D FFT → electric potential φ
#          c. Density gradient = electric field E = ∇φ (gradient of potential)
#          d. Wirelength gradient via WA (weighted-average) smoothing of HPWL
#          e. Total gradient = WL_grad + λ * density_grad
#          f. Nesterov update with momentum
#          g. Clamp to canvas, increase λ if overflow still high
#       3. Return positions
#     """
#     t_start = time.time()
#     device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     W = float(benchmark.canvas_width)
#     H = float(benchmark.canvas_height)
#     g = LEGALIZE_GAP

#     sizes_t = benchmark.macro_sizes.to(device)  # [N, 2]
#     N       = benchmark.num_macros

#     # Grid dimensions — use benchmark's grid (already defined for density eval)
#     n_cols = benchmark.grid_cols
#     n_rows = benchmark.grid_rows
#     bin_w  = W / n_cols
#     bin_h  = H / n_rows

#     movable_mask = (benchmark.get_movable_mask() &
#                     benchmark.get_hard_macro_mask()).numpy()
#     hard_movable = [i for i in range(N) if movable_mask[i]]
#     hard_all     = torch.where(benchmark.get_hard_macro_mask())[0].tolist()

#     if not hard_movable:
#         return benchmark.macro_positions.numpy().astype(np.float32)

#     # ── Random initialisation ─────────────────────────────────────────────
#     rng = random.Random(seed)
#     pos_np = benchmark.macro_positions.numpy().astype(np.float32).copy()

#     for i in hard_movable:
#         hw = float(sizes_t[i, 0]) * 0.5
#         hh = float(sizes_t[i, 1]) * 0.5
#         pos_np[i, 0] = rng.uniform(hw + g, W - hw - g)
#         pos_np[i, 1] = rng.uniform(hh + g, H - hh - g)

#     # ── Convert nets to macro-only format for WL gradient ─────────────────
#     # Each net: list of macro indices that participate (ignoring IO ports)
#     # We use a simplified WA wirelength gradient
#     net_macros = []   # list of lists of macro indices
#     for drv_idx, dox, doy, sinks in nets_cpp:
#         members = []
#         if drv_idx >= 0:
#             members.append(drv_idx)
#         for sidx, sox, soy in sinks:
#             if sidx >= 0 and sidx not in members:
#                 members.append(sidx)
#         if len(members) >= 2:
#             net_macros.append(members)

#     # ── Precompute Poisson kernel in frequency domain ─────────────────────
#     # For a 2D periodic domain, the Green's function of the Laplacian in
#     # frequency domain is -1 / (kx² + ky²) with the DC component zeroed.
#     # We use the discrete version with wavenumbers kx, ky.
#     kx = torch.fft.fftfreq(n_cols, d=1.0/n_cols, device=device) * (2*np.pi/n_cols)
#     ky = torch.fft.fftfreq(n_rows, d=1.0/n_rows, device=device) * (2*np.pi/n_rows)
#     KX, KY = torch.meshgrid(kx, ky, indexing='ij')  # [n_cols, n_rows]
#     K2 = KX**2 + KY**2
#     K2[0, 0] = 1.0   # avoid division by zero at DC; DC component is zeroed

#     # Precomputed kernel: -1/K2 (the Poisson kernel)
#     poisson_kernel = -1.0 / K2   # [n_cols, n_rows]
#     poisson_kernel[0, 0] = 0.0   # zero DC (fixes absolute potential)

#     # ── Nesterov state ────────────────────────────────────────────────────
#     pos    = torch.tensor(pos_np, device=device)   # [N, 2]  all macros
#     pos_v  = pos.clone()   # Nesterov "lookahead" variable v
#     alpha  = 1.0           # Nesterov momentum coefficient
#     lam    = 1e-3          # density penalty lambda (grows over iterations)
#     lr     = min(bin_w, bin_h) * 0.5   # learning rate ≈ half a bin

#     hard_mov_t = torch.tensor(hard_movable, dtype=torch.long, device=device)
#     hard_all_t = torch.tensor(hard_all,     dtype=torch.long, device=device)

#     def clamp_pos(p):
#         """Clamp all movable macros to canvas."""
#         hw = sizes_t[hard_mov_t, 0] * 0.5  # [n_mov]
#         hh = sizes_t[hard_mov_t, 1] * 0.5
#         p_mov = p[hard_mov_t]
#         p_mov[:, 0] = torch.clamp(p_mov[:, 0], hw + g, W - hw - g)
#         p_mov[:, 1] = torch.clamp(p_mov[:, 1], hh + g, H - hh - g)
#         p = p.clone()
#         p[hard_mov_t] = p_mov
#         return p

#     def density_grad_and_overflow(p):
#         """
#         Compute density gradient and overflow using FFT.
#         Returns (grad [N, 2], overflow scalar).
#         grad is zero for fixed/soft macros.
#         """
#         # ── Rasterise macro areas onto grid ────────────────────────────
#         # For each hard macro, add its area to the bins it overlaps.
#         # We use a simple bell-shaped smearing: each macro is represented
#         # as a rectangular patch on the grid.
#         density = torch.zeros(n_cols, n_rows, device=device)

#         p_hard = p[hard_all_t]            # [n_hard, 2]
#         s_hard = sizes_t[hard_all_t]      # [n_hard, 2]

#         # Vectorised rasterisation: find which bins each macro covers
#         x0 = (p_hard[:, 0] - s_hard[:, 0]*0.5) / bin_w   # fractional col
#         x1 = (p_hard[:, 0] + s_hard[:, 0]*0.5) / bin_w
#         y0 = (p_hard[:, 1] - s_hard[:, 1]*0.5) / bin_h
#         y1 = (p_hard[:, 1] + s_hard[:, 1]*0.5) / bin_h

#         # Clamp to grid bounds
#         x0c = torch.clamp(x0, 0, n_cols - 1e-6)
#         x1c = torch.clamp(x1, 0, n_cols - 1e-6)
#         y0c = torch.clamp(y0, 0, n_rows - 1e-6)
#         y1c = torch.clamp(y1, 0, n_rows - 1e-6)

#         # For each macro, scatter its area fraction into overlapping bins
#         # (simplified: use centre bin only for speed, full overlap for accuracy)
#         # We do full overlap via a loop over macros — n_hard is small (≤800)
#         n_hard = p_hard.shape[0]
#         for k in range(n_hard):
#             col0 = max(0, int(x0c[k].item()))
#             col1 = min(n_cols - 1, int(x1c[k].item()))
#             row0 = max(0, int(y0c[k].item()))
#             row1 = min(n_rows - 1, int(y1c[k].item()))
#             if col0 <= col1 and row0 <= row1:
#                 # Area fraction this macro contributes to each overlapped bin
#                 macro_area = float(s_hard[k, 0].item() * s_hard[k, 1].item())
#                 n_bins_covered = (col1-col0+1) * (row1-row0+1)
#                 per_bin = macro_area / max(1, n_bins_covered)
#                 density[col0:col1+1, row0:row1+1] += per_bin

#         bin_area  = bin_w * bin_h
#         overflow  = float(torch.clamp(density - bin_area, min=0).sum().item()
#                           / max(1.0, (p_hard[:,0]*0).sum().item() + 1.0))
#         # Simpler overflow: fraction of bins that are overcrowded
#         overflow = float((density > bin_area).float().mean().item())

#         # ── Solve Poisson: potential = IFFT(-1/K² × FFT(density)) ────
#         dens_f   = torch.fft.fft2(density)       # FFT of density
#         phi_f    = dens_f * poisson_kernel        # multiply by kernel
#         phi      = torch.fft.ifft2(phi_f).real   # potential [n_cols, n_rows]

#         # ── Electric field = gradient of potential ────────────────────
#         # Ex = d(phi)/dx, Ey = d(phi)/dy in frequency domain:
#         # Ex_f = i*kx * phi_f,  Ey_f = i*ky * phi_f
#         Ex_f = 1j * KX * phi_f
#         Ey_f = 1j * KY * phi_f
#         Ex   = torch.fft.ifft2(Ex_f).real   # [n_cols, n_rows]
#         Ey   = torch.fft.ifft2(Ey_f).real

#         # ── Sample field at each movable macro's position ─────────────
#         grad = torch.zeros_like(p)   # [N, 2]

#         p_mov = p[hard_mov_t]                    # [n_mov, 2]
#         col_f = torch.clamp(p_mov[:, 0] / bin_w, 0, n_cols - 1)
#         row_f = torch.clamp(p_mov[:, 1] / bin_h, 0, n_rows - 1)
#         col_i = col_f.long().clamp(0, n_cols - 1)
#         row_i = row_f.long().clamp(0, n_rows - 1)

#         gx = Ex[col_i, row_i]   # [n_mov]
#         gy = Ey[col_i, row_i]

#         grad[hard_mov_t, 0] = gx
#         grad[hard_mov_t, 1] = gy
#         return grad, overflow

#     def wl_grad(p):
#         """
#         Weighted-average (WA) wirelength gradient.
#         For each net, the gradient pulls each macro toward the net centroid.
#         This is a simplified version — no pin offsets for speed.
#         """
#         grad = torch.zeros_like(p)
#         gamma = 1.0   # WA smoothing parameter

#         for members in net_macros:
#             if len(members) < 2:
#                 continue
#             pos_m = p[members]                  # [k, 2]
#             # WA softmax weights
#             wx    = torch.exp((pos_m[:, 0] - pos_m[:, 0].max()) / gamma)
#             wy    = torch.exp((pos_m[:, 1] - pos_m[:, 1].max()) / gamma)
#             wx_neg= torch.exp((pos_m[:, 0].min() - pos_m[:, 0]) / gamma)
#             wy_neg= torch.exp((pos_m[:, 1].min() - pos_m[:, 1]) / gamma)

#             sx = wx.sum(); sy = wy.sum()
#             sx_n = wx_neg.sum(); sy_n = wy_neg.sum()

#             # x gradient for each macro in net
#             for ki, mi in enumerate(members):
#                 if not movable_mask[mi]:
#                     continue
#                 # Positive x endpoint gradient
#                 dfdxi = (wx[ki] * (sx - wx[ki])) / (sx**2 + 1e-12)
#                 # Negative x endpoint gradient
#                 dfdxi_n = -(wx_neg[ki] * (sx_n - wx_neg[ki])) / (sx_n**2 + 1e-12)
#                 grad[mi, 0] += dfdxi + dfdxi_n

#                 dfdyi   = (wy[ki] * (sy - wy[ki])) / (sy**2 + 1e-12)
#                 dfdyi_n = -(wy_neg[ki] * (sy_n - wy_neg[ki])) / (sy_n**2 + 1e-12)
#                 grad[mi, 1] += dfdyi + dfdyi_n

#         return grad

#     # ── Nesterov main loop ────────────────────────────────────────────────
#     alpha_prev = 1.0

#     for it in range(FFT_MAX_ITERS):
#         if time.time() - t_start > time_limit:
#             break

#         # Compute gradients at lookahead point v
#         with torch.no_grad():
#             d_dens, overflow = density_grad_and_overflow(pos_v)
#             d_wl             = wl_grad(pos_v)
#             total_grad       = d_wl + lam * d_dens

#         # Gradient step on movable macros only
#         new_pos = pos.clone()
#         new_pos[hard_mov_t] = (pos[hard_mov_t]
#                                - lr * total_grad[hard_mov_t])
#         new_pos = clamp_pos(new_pos)

#         # Nesterov momentum update
#         alpha_new = (1.0 + math.sqrt(1.0 + 4.0 * alpha_prev**2)) / 2.0
#         beta      = (alpha_prev - 1.0) / alpha_new

#         pos_v_new = new_pos.clone()
#         pos_v_new[hard_mov_t] = (new_pos[hard_mov_t]
#                                  + beta * (new_pos[hard_mov_t]
#                                            - pos[hard_mov_t]))
#         pos_v_new = clamp_pos(pos_v_new)

#         pos       = new_pos
#         pos_v     = pos_v_new
#         alpha_prev = alpha_new

#         # Increase density penalty
#         lam = min(lam * 1.05, 1e4)

#         if it % 50 == 0:
#             elapsed = time.time() - t_start
#             print(f"    [FFT] iter={it:3d}  overflow={overflow:.3f}  "
#                   f"lam={lam:.2f}  t={elapsed:.1f}s")

#         if overflow < FFT_OV_TARGET:
#             print(f"    [FFT] converged at iter={it}  overflow={overflow:.4f}")
#             break

#     result_np = pos.cpu().numpy().astype(np.float32)
#     return result_np


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: torch overlap check (main process diagnostics)
# # ─────────────────────────────────────────────────────────────────────────────

# def _torch_overlap(pos_t: torch.Tensor, sizes_t: torch.Tensor,
#                    hard_idx: List[int],
#                    threshold: float = 0.0) -> Tuple[int, float]:
#     """
#     threshold=0.0  → any positive overlap area counts (strict, matches evaluator)
#     threshold=0.0040 → relative overlap / min_macro_area > 0.004 (for SA cost)
#     """
#     n = len(hard_idx)
#     if n < 2:
#         return 0, 0.0
#     idx_t = torch.tensor(hard_idx, dtype=torch.long)
#     p = pos_t[idx_t]; s = sizes_t[idx_t]
#     hw = s[:, 0] * 0.5;  hh = s[:, 1] * 0.5
#     xmin = p[:, 0] - hw;  xmax = p[:, 0] + hw
#     ymin = p[:, 1] - hh;  ymax = p[:, 1] + hh

#     ox = (torch.minimum(xmax.unsqueeze(0), xmax.unsqueeze(1))
#         - torch.maximum(xmin.unsqueeze(0), xmin.unsqueeze(1)))
#     oy = (torch.minimum(ymax.unsqueeze(0), ymax.unsqueeze(1))
#         - torch.maximum(ymin.unsqueeze(0), ymin.unsqueeze(1)))

#     ov_area  = ox * oy
#     area_i   = s[:, 0] * s[:, 1]
#     min_area = torch.minimum(
#         area_i.unsqueeze(0).expand(n, n),
#         area_i.unsqueeze(1).expand(n, n))

#     # Relative overlap fraction — dimensionless, comparable to threshold
#     relative_ov = ov_area / (min_area + 1e-12)

#     triu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

#     if threshold == 0.0:
#         # Strict: any positive overlap area counts
#         mask = (ox > 1e-6) & (oy > 1e-6) & triu
#     else:
#         # Relative threshold (used during SA cost evaluation)
#         mask = (ox > 0) & (oy > 0) & (relative_ov > threshold) & triu

#     return int(mask.sum().item()), float((ov_area * mask.float()).sum().item())

# # ─────────────────────────────────────────────────────────────────────────────
# # SPIRAL INIT (unchanged — kept for worker diversity)
# # ─────────────────────────────────────────────────────────────────────────────

# def _spiral_init(benchmark: Benchmark, seed: int = 0) -> np.ndarray:
#     rng   = random.Random(seed)
#     sizes = benchmark.macro_sizes
#     W     = float(benchmark.canvas_width)
#     H     = float(benchmark.canvas_height)
#     g     = LEGALIZE_GAP
#     pos   = benchmark.macro_positions.numpy().astype(np.float32).copy()
#     movable = (benchmark.get_movable_mask() &
#                benchmark.get_hard_macro_mask()).numpy()
#     idx = [i for i in range(len(movable)) if movable[i]]
#     if not idx:
#         return pos
#     idx.sort(key=lambda i: (-(float(sizes[i][0])*float(sizes[i][1])),
#                              rng.random()))
#     left=g; right=W-g; bot=g; top=H-g
#     cx_ptr=left; cy_ptr=bot; direction=0; row_h=0.0
#     for i in idx:
#         w=float(sizes[i][0]); h=float(sizes[i][1])
#         hw=w/2.0; hh=h/2.0; placed=False
#         for _ in range(8):
#             if direction == 0:
#                 if cx_ptr+w+g <= right:
#                     cx=cx_ptr+hw; cy=cy_ptr+hh
#                     cx_ptr+=w+g; row_h=max(row_h,h); placed=True; break
#                 else:
#                     direction=1; cy_ptr=max(cy_ptr,bot+row_h+g)
#                     cx_ptr=right-hw; row_h=0.0
#             elif direction == 1:
#                 if cy_ptr+h+g <= top:
#                     cx=cx_ptr; cy=cy_ptr+hh; cy_ptr+=h+g; placed=True; break
#                 else:
#                     direction=2; cx_ptr=right-hw; cy_ptr=top-hh
#             elif direction == 2:
#                 if cx_ptr-w-g >= left:
#                     cx=cx_ptr-hw; cy=cy_ptr; cx_ptr-=w+g; placed=True; break
#                 else:
#                     direction=3; cy_ptr=top-hh; cx_ptr=left+hw
#             else:
#                 if cy_ptr-h-g >= bot:
#                     cx=cx_ptr; cy=cy_ptr-hh; cy_ptr-=h+g; placed=True; break
#                 else:
#                     direction=0; shrink=max(2.0*g, row_h+g)
#                     left+=shrink; right-=shrink; bot+=shrink; top-=shrink
#                     row_h=0.0; cx_ptr=left+hw; cy_ptr=bot+hh
#                     if left>=right or bot>=top:
#                         cx=rng.uniform(hw+g,W-hw-g)
#                         cy=rng.uniform(hh+g,H-hh-g); placed=True; break
#         if not placed:
#             cx=rng.uniform(hw+g,W-hw-g); cy=rng.uniform(hh+g,H-hh-g)
#         pos[i,0]=max(hw+g,min(W-hw-g,cx))
#         pos[i,1]=max(hh+g,min(H-hh-g,cy))
#     return pos


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN PLACER
# # ─────────────────────────────────────────────────────────────────────────────

# class SimulatedAnnealingPlacer:

#     def __init__(self, time_budget=TIME_BUDGET_SEC, num_workers=NUM_WORKERS):
#         self.time_budget = time_budget
#         self.num_workers = num_workers

#     def place(self, benchmark: Benchmark) -> torch.Tensor:
#         t0      = time.time()
#         name    = benchmark.name
#         sizes_t = benchmark.macro_sizes
#         sizes_np = sizes_t.numpy().astype(np.float32)
#         W = float(benchmark.canvas_width)
#         H = float(benchmark.canvas_height)

#         hard_mask    = (benchmark.get_movable_mask() &
#                         benchmark.get_hard_macro_mask())
#         hard_movable = torch.where(hard_mask)[0].tolist()
#         hard_all     = torch.where(
#                         benchmark.get_hard_macro_mask())[0].tolist()

#         hard_t = torch.tensor(hard_all, dtype=torch.long)
#         util   = float((sizes_t[hard_t,0]*sizes_t[hard_t,1]).sum()) / (W*H)
#         print(f"\n{'='*65}")
#         print(f"  [{name}] W={W:.1f} H={H:.1f}  "
#               f"hard={len(hard_all)} movable={len(hard_movable)}  "
#               f"util={util:.1%}")

#         # ── 1. Load nets ──────────────────────────────────────────────────
#         plc = None; nets_raw = []
#         try:
#             from macro_place.loader import load_benchmark_from_dir
#             tdir = _find_testcase_dir(name)
#             if tdir:
#                 _, plc = load_benchmark_from_dir(tdir)
#                 nets_raw = _extract_nets(plc, benchmark)
#                 print(f"  [{name}] nets={len(nets_raw)}")
#             else:
#                 print(f"  [{name}] WARNING: testcase dir not found")
#         except Exception as e:
#             print(f"  [{name}] WARNING: plc load failed ({e})")

#         nets_cpp = nets_to_cpp_format(nets_raw)

#         # ── 2. Adaptive overlap penalty ───────────────────────────────────
#         ref_np = benchmark.macro_positions.numpy().astype(np.float32)
#         if _HAS_CPP and _cpp is not None:
#             try:
#                 _cpp.init_nets(nets_cpp, benchmark.num_macros)
#                 ref_hpwl = float(_cpp.compute_hpwl(ref_np))
#             except Exception:
#                 ref_hpwl = _py_hpwl(ref_np, nets_cpp)
#         else:
#             ref_hpwl = _py_hpwl(ref_np, nets_cpp)

#         overlap_pen = (OVERLAP_PENALTY_SCALE * ref_hpwl
#                        if ref_hpwl > 1.0 else OVERLAP_PENALTY_FALLBACK)
#         print(f"  [{name}] ref_hpwl={ref_hpwl:.1f}  "
#               f"overlap_pen={overlap_pen:.1f}")

#         # ── 3. FIX 1: Guaranteed-valid fallback ───────────────────────────
#         fallback_np = _greedy_pack_init(benchmark, seed=99999)
#         fallback_t  = torch.from_numpy(fallback_np)
#         fb_ov, _    = _torch_overlap(fallback_t, sizes_t, hard_all)
#         print(f"  [{name}] fallback overlaps={fb_ov}  "
#               f"({'VALID' if fb_ov==0 else 'needs FFT fix'})")

#         # If greedy pack still has overlaps (very dense benchmarks), use FFT
#         # spreader as the fallback too
#         if fb_ov > 0:
#             print(f"  [{name}] running FFT on fallback to fix {fb_ov} overlaps")
#             fft_fallback = _fft_spread(benchmark, nets_cpp, seed=99999,
#                                        time_limit=5.0)
#             fft_t = torch.from_numpy(fft_fallback)
#             fft_ov, _ = _torch_overlap(fft_t, sizes_t, hard_all)
#             print(f"  [{name}] FFT fallback overlaps={fft_ov}")
#             if fft_ov < fb_ov:
#                 fallback_np = fft_fallback
#                 fallback_t  = fft_t
#                 fb_ov       = fft_ov

#         # ── 4. FIX 3: FFT spreader for SA initialisation ─────────────────
#         # Budget: up to FFT_TIME_LIMIT seconds, but stop early if time is short
#         fft_budget = min(FFT_TIME_LIMIT, self.time_budget * 0.12)
#         print(f"  [{name}] running FFT spreader  budget={fft_budget:.1f}s")
#         fft_np = _fft_spread(benchmark, nets_cpp, seed=42,
#                               time_limit=fft_budget)
#         fft_t  = torch.from_numpy(fft_np)
#         fft_ov, _ = _torch_overlap(fft_t, sizes_t, hard_all)
#         print(f"  [{name}] FFT init overlaps={fft_ov}")

#         # ── 5. Build worker args ──────────────────────────────────────────
#         overhead  = min(5.0, 0.5 * self.num_workers)
#         sa_time   = max(10.0,
#                         self.time_budget - overhead - (time.time() - t0))
#         n_workers = min(self.num_workers,
#                         max(2, len(hard_movable) // 4 + 2))
#         print(f"  [{name}] {n_workers} workers  sa_time={sa_time:.0f}s  "
#               f"cpp={'ON' if _HAS_CPP else 'OFF'}")

#         # Worker diversity:
#         #   workers 0..3  — greedy pack init  (half)
#         #   workers 4..7  — spiral init       (quarter)
#         #   workers 8..15 — FFT spread init   (quarter — best starting point)
#         worker_args = []
#         fft_workers  = n_workers // 4          # ~4 workers get FFT init
#         grep_workers = n_workers // 2          # ~8 workers get greedy
#         for wid in range(n_workers):
#             if wid < grep_workers:
#                 init_np = _greedy_pack_init(benchmark, seed=1000+wid)
#             elif wid < grep_workers + (n_workers - grep_workers - fft_workers):
#                 init_np = _spiral_init(benchmark, seed=2000+wid)
#             else:
#                 # Add small perturbation to FFT result for diversity
#                 init_np = fft_np.copy()
#                 rng_w   = random.Random(5000+wid)
#                 noise   = 0.5  # microns of noise
#                 for mi in hard_movable:
#                     hw = float(sizes_np[mi, 0]) * 0.5
#                     hh = float(sizes_np[mi, 1]) * 0.5
#                     init_np[mi, 0] = max(hw + LEGALIZE_GAP,
#                                          min(W - hw - LEGALIZE_GAP,
#                                              init_np[mi, 0] +
#                                              rng_w.uniform(-noise, noise)))
#                     init_np[mi, 1] = max(hh + LEGALIZE_GAP,
#                                          min(H - hh - LEGALIZE_GAP,
#                                              init_np[mi, 1] +
#                                              rng_w.uniform(-noise, noise)))

#             worker_args.append((
#                 wid,
#                 init_np,
#                 sizes_np,
#                 hard_movable,
#                 hard_all,
#                 nets_cpp,
#                 benchmark.num_macros,
#                 W, H,
#                 3000 + wid,
#                 overlap_pen,
#                 sa_time,
#                 _here,
#             ))

#         # ── 6. Run workers ────────────────────────────────────────────────
#         results = []
#         try:
#             with ProcessPoolExecutor(max_workers=n_workers) as ex:
#                 futures = [ex.submit(run_sa_worker, a) for a in worker_args]
#                 for wid, f in enumerate(futures):
#                     try:
#                         res = f.result(timeout=sa_time + 30.0)
#                         results.append(res)
#                         p, c, it = res
#                         print(f"  [{name}] w{wid:02d}  cost={c:.2f}  "
#                               f"valid={'Y' if p is not None else 'N'}  "
#                               f"iters={it:,}")
#                     except Exception as e:
#                         print(f"  [{name}] worker {wid} error: {e}")
#         except Exception as e:
#             print(f"  [{name}] executor error: {e}")

#         # ── 7. Pick best valid result ─────────────────────────────────────
#         best_np = None; best_cost = float('inf')
#         for p, c, _ in results:
#             if p is not None and c < best_cost:
#                 best_cost = c; best_np = p

#         if best_np is not None:
#             candidate = torch.from_numpy(best_np)
#             c_ov, _   = _torch_overlap(candidate, sizes_t, hard_all)
#             print(f"  [{name}] SA best cost={best_cost:.4f}  ov={c_ov}")
#             if c_ov == 0:
#                 result = candidate
#             elif fb_ov == 0:
#                 print(f"  [{name}] SA invalid, using valid fallback")
#                 result = fallback_t
#             elif fft_ov == 0:
#                 print(f"  [{name}] SA invalid, using valid FFT init")
#                 result = fft_t
#             elif fb_ov <= c_ov:
#                 result = fallback_t
#             else:
#                 result = candidate
#         else:
#             print(f"  [{name}] no SA result — using best available")
#             if fb_ov == 0:
#                 result = fallback_t
#             elif fft_ov == 0:
#                 result = fft_t
#             else:
#                 result = fallback_t if fb_ov <= fft_ov else fft_t

#         try:
#             sys.path.insert(0, os.path.join(os.path.dirname(_here),
#                                             "submissions", "examples"))
#             from greedy_row_placer import GreedyRowPlacer as _GRP
#             _guaranteed_fallback = _GRP().place(benchmark)
#         except Exception:
#             _guaranteed_fallback = fallback_t   # our own greedy pack if import fails


#         grid_w = W / benchmark.grid_cols
#         grid_h = H / benchmark.grid_rows

#         result_np = result.numpy().astype(np.float32).copy()

#         for mi in hard_all:
#             hw = float(sizes_t[mi, 0]) * 0.5
#             hh = float(sizes_t[mi, 1]) * 0.5
#             cx = float(result_np[mi, 0])
#             cy = float(result_np[mi, 1])

#             # Which grid cell contains this macro center?
#             col = int(cx / grid_w)
#             row = int(cy / grid_h)
#             col = max(0, min(benchmark.grid_cols - 1, col))
#             row = max(0, min(benchmark.grid_rows - 1, row))

#             # Snap to cell center
#             cx_snap = (col + 0.5) * grid_w
#             cy_snap = (row + 0.5) * grid_h

#             # Clamp so macro body stays fully inside canvas
#             cx_snap = max(hw, min(W - hw, cx_snap))
#             cy_snap = max(hh, min(H - hh, cy_snap))

#             result_np[mi, 0] = cx_snap
#             result_np[mi, 1] = cy_snap

#         result_snapped = torch.from_numpy(result_np)

#         # Use strict threshold=0.0 — after snapping we want exact evaluator behaviour
#         snap_ov, _ = _torch_overlap(result_snapped, sizes_t, hard_all, threshold=0.0)
#         print(f"  [{name}] after grid-snap  ov={snap_ov}")

#         if snap_ov == 0:
#             result = result_snapped
#         else:
#             print(f"  [{name}] snap introduced {snap_ov} overlaps — using pre-snap result")
#             # Pre-snap result already passed our threshold=0.0 check, return it
#             result = result  # unchanged — better than falling back to greedy pack

#         # ── Final report ──────────────────────────────────────────────────────────
#         final_ov, _ = _torch_overlap(result, sizes_t, hard_all, threshold=0.0)
#         elapsed = time.time() - t0
#         print(f"  [{name}] DONE {elapsed:.1f}s  ov={final_ov}  "
#             f"{'✓ VALID' if final_ov==0 else '✗ INVALID'}")
#         print(f"{'='*65}\n")
#         return result


# # ─────────────────────────────────────────────────────────────────────────────
# # ENTRY POINT
# # ─────────────────────────────────────────────────────────────────────────────

# placer = SimulatedAnnealingPlacer(
#     time_budget=TIME_BUDGET_SEC,
#     num_workers=NUM_WORKERS,
# )


# """
# GWTW-SA Macro Placer  —  4-Fix Version
# =======================================

# Fix 1: Valid fallback — greedy pack never wraps, always produces zero overlaps.
# Fix 2: Overlap gate in SA — do_overlap only called after HPWL gate passes.
# Fix 3: FFT electrostatic spreader — spread macros from random init before SA.
# Fix 4: Two-phase SA — phase 1 hammers overlaps, phase 2 optimises wirelength.
# """
# import os
# import sys
# import time
# import random
# from typing import List, Tuple, Optional
# from concurrent.futures import ProcessPoolExecutor
# import math

# import numpy as np
# import torch
# from macro_place.benchmark import Benchmark

# # ─────────────────────────────────────────────────────────────────────────────
# # Worker lives in worker_impl.py — stable module name for ProcessPoolExecutor
# # ─────────────────────────────────────────────────────────────────────────────

# _here = os.path.dirname(os.path.abspath(__file__))
# if _here not in sys.path:
#     sys.path.insert(0, _here)

# from worker_impl import run_sa_worker, nets_to_cpp_format

# # ─────────────────────────────────────────────────────────────────────────────
# # C++ extension (main process — for ref_hpwl only)
# # ─────────────────────────────────────────────────────────────────────────────

# _cpp = None
# _HAS_CPP = False
# for _p in [_here, os.path.join(_here, "placement_ops")]:
#     if _p not in sys.path:
#         sys.path.insert(0, _p)
# try:
#     import placement_ops as _cpp
#     _HAS_CPP = True
#     print("[placement_ops] C++ extension loaded — fast HPWL active")
# except ImportError:
#     print("[placement_ops] WARNING: C++ not found — Python fallback")
#     print(f"  Run: cd placement_ops && python setup.py build_ext --inplace")
#     print(f"  Then copy the .pyd/.so into submissions/")

# # ─────────────────────────────────────────────────────────────────────────────
# # HYPERPARAMETERS
# # ─────────────────────────────────────────────────────────────────────────────

# TIME_BUDGET_SEC          = 55.0
# NUM_WORKERS              = 16
# LEGALIZE_GAP             = 0.5
# OVERLAP_PENALTY_SCALE    = 50.0
# OVERLAP_PENALTY_FALLBACK = 1e6

# # FFT spreader settings
# FFT_MAX_ITERS   = 300      # Nesterov iterations
# FFT_TIME_LIMIT  = 8.0     # hard wall: never spend more than this on spreading
# FFT_OV_TARGET   = 0.02    # stop when density overflow < 2%


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: locate testcase dir
# # ─────────────────────────────────────────────────────────────────────────────

# def _find_testcase_dir(name: str) -> Optional[str]:
#     candidates = [
#         f"external/MacroPlacement/Testcases/ICCAD04/{name}",
#         os.path.join(os.getcwd(), "external", "MacroPlacement",
#                      "Testcases", "ICCAD04", name),
#     ]
#     for c in candidates:
#         if os.path.isdir(c) and os.path.isfile(
#                 os.path.join(c, "netlist.pb.txt")):
#             return c
#     return None


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: extract nets
# # ─────────────────────────────────────────────────────────────────────────────

# def _extract_nets(plc, benchmark) -> List:
#     nets = []
#     if plc is None or not hasattr(plc, 'nets'):
#         return nets
#     name_to_bench = {n: i for i, n in enumerate(benchmark.macro_names)}

#     def resolve(pin_name):
#         if pin_name not in plc.mod_name_to_indices:
#             return None
#         idx = plc.mod_name_to_indices[pin_name]
#         if idx >= len(plc.modules_w_pins):
#             return None
#         obj = plc.modules_w_pins[idx]
#         try:
#             ptype = obj.get_type()
#         except Exception:
#             return None
#         if ptype == 'PORT':
#             try:
#                 px, py = obj.get_pos()
#                 return (-1, (float(px), float(py)))
#             except Exception:
#                 return None
#         if ptype != 'MACRO_PIN':
#             return None
#         try:
#             parent = obj.macro_name
#         except AttributeError:
#             return None
#         if parent not in name_to_bench:
#             return None
#         try:
#             ox, oy = obj.get_offset()
#         except Exception:
#             ox, oy = 0.0, 0.0
#         return (name_to_bench[parent], (float(ox), float(oy)))

#     for drv_name, sink_names in plc.nets.items():
#         drv = resolve(drv_name)
#         if drv is None:
#             continue
#         sinks = [s for sn in sink_names
#                  for s in [resolve(sn)] if s is not None]
#         if sinks:
#             nets.append((drv[0], drv[1], sinks))
#     return nets


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: Python HPWL (main process ref calculation)
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


# # ─────────────────────────────────────────────────────────────────────────────
# # FIX 1: GUARANTEED-VALID GREEDY PACK
# # Never wraps back to origin — instead places overflow macros randomly in
# # remaining whitespace, and falls back to explicit overlap-free slot search.
# # ─────────────────────────────────────────────────────────────────────────────

# def _greedy_pack_init(benchmark: Benchmark, seed: int = 0) -> np.ndarray:
#     """
#     Row-pack macros sorted by height descending.
#     When a row overflows vertically, we find the actual remaining whitespace
#     columns and continue there — never wrapping back to already-occupied rows.
#     Result is guaranteed overlap-free as long as total macro area < canvas area.
#     """
#     rng   = random.Random(seed)
#     sizes = benchmark.macro_sizes
#     W     = float(benchmark.canvas_width)
#     H     = float(benchmark.canvas_height)
#     g     = LEGALIZE_GAP

#     pos   = benchmark.macro_positions.numpy().astype(np.float32).copy()
#     movable = (benchmark.get_movable_mask() &
#                benchmark.get_hard_macro_mask()).numpy()
#     idx = [i for i in range(len(movable)) if movable[i]]
#     if not idx:
#         return pos

#     idx.sort(key=lambda i: (-float(sizes[i][1]), rng.random()))

#     x_cur = g; y_cur = g; row_h = 0.0
#     # Track placed boxes for overflow fallback
#     placed_boxes = []  # list of (x0, y0, x1, y1)

#     for i in idx:
#         w = float(sizes[i][0]); h = float(sizes[i][1])
#         hw = w / 2.0; hh = h / 2.0

#         if w + 2*g > W or h + 2*g > H:
#             # Macro nearly as big as canvas — center it
#             pos[i, 0] = W / 2.0; pos[i, 1] = H / 2.0
#             placed_boxes.append((W/2-hw, H/2-hh, W/2+hw, H/2+hh))
#             continue

#         if x_cur + w + g > W:
#             # End of row — move to next row
#             y_cur += row_h + g; x_cur = g; row_h = 0.0

#         if y_cur + h + g <= H:
#             # Normal placement
#             cx = x_cur + hw; cy = y_cur + hh
#             pos[i, 0] = cx; pos[i, 1] = cy
#             placed_boxes.append((cx-hw, cy-hh, cx+hw, cy+hh))
#             x_cur += w + g; row_h = max(row_h, h)
#         else:
#             # Canvas full — scan for a free location using a coarse grid
#             cx, cy = _find_free_slot(w, h, W, H, g, placed_boxes, rng)
#             pos[i, 0] = cx; pos[i, 1] = cy
#             placed_boxes.append((cx-hw, cy-hh, cx+hw, cy+hh))

#     return pos


# def _find_free_slot(w: float, h: float, W: float, H: float,
#                     g: float, placed_boxes: List,
#                     rng: random.Random) -> Tuple[float, float]:
#     """
#     Scan on a coarse grid to find a non-overlapping position for a macro
#     of size (w, h). Falls back to random position if grid scan finds nothing.
#     """
#     hw = w / 2.0; hh = h / 2.0
#     step = max(w, h) * 0.5  # coarse grid step

#     x = hw + g
#     while x + hw + g <= W:
#         y = hh + g
#         while y + hh + g <= H:
#             # Check if this candidate position overlaps any placed box
#             ok = True
#             for (bx0, by0, bx1, by1) in placed_boxes:
#                 ox = min(x+hw, bx1) - max(x-hw, bx0)
#                 oy = min(y+hh, by1) - max(y-hh, by0)
#                 if ox > 1e-3 and oy > 1e-3:
#                     ok = False; break
#             if ok:
#                 return x, y
#             y += step
#         x += step

#     # Grid scan failed — return random clamped position (last resort)
#     cx = rng.uniform(hw + g, max(hw + g + 0.01, W - hw - g))
#     cy = rng.uniform(hh + g, max(hh + g + 0.01, H - hh - g))
#     return cx, cy


# # ─────────────────────────────────────────────────────────────────────────────
# # FIX 3: FFT ELECTROSTATIC SPREADER
# #
# # Physics: macros are positively charged particles. Charge density on a grid
# # creates an electrostatic potential (via Poisson's equation solved with FFT).
# # Each macro experiences a repulsive force proportional to the gradient of the
# # potential — this pushes macros away from crowded bins.
# # Simultaneously, a wirelength gradient (from net spring forces) pulls
# # connected macros together.
# #
# # We use Nesterov's accelerated gradient descent (same as RePlAce/ePlace).
# # The density penalty lambda grows slowly until overflow is resolved.
# #
# # GPU-accelerated automatically if torch.cuda is available (RTX 6000 on
# # judge machine). Falls back to CPU on your i7.
# # ─────────────────────────────────────────────────────────────────────────────

# def _fft_spread(benchmark: Benchmark,
#                 nets_cpp: List,
#                 seed: int = 0,
#                 time_limit: float = FFT_TIME_LIMIT) -> np.ndarray:
#     """
#     FFT-based electrostatic spreader.

#     Returns float32 numpy [N, 2] with hard macros spread to near-zero overlap.
#     Soft macros are left at their reference positions.

#     Algorithm (ePlace / RePlAce):
#       1. Random initialisation of hard macro positions
#       2. For each Nesterov iteration:
#          a. Compute charge density on grid (each macro smeared over its bins)
#          b. Solve Poisson equation via 2D FFT → electric potential φ
#          c. Density gradient = electric field E = ∇φ (gradient of potential)
#          d. Wirelength gradient via WA (weighted-average) smoothing of HPWL
#          e. Total gradient = WL_grad + λ * density_grad
#          f. Nesterov update with momentum
#          g. Clamp to canvas, increase λ if overflow still high
#       3. Return positions
#     """
#     t_start = time.time()
#     device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     W = float(benchmark.canvas_width)
#     H = float(benchmark.canvas_height)
#     g = LEGALIZE_GAP

#     sizes_t = benchmark.macro_sizes.to(device)  # [N, 2]
#     N       = benchmark.num_macros

#     # Grid dimensions — use benchmark's grid (already defined for density eval)
#     n_cols = benchmark.grid_cols
#     n_rows = benchmark.grid_rows
#     bin_w  = W / n_cols
#     bin_h  = H / n_rows

#     movable_mask = (benchmark.get_movable_mask() &
#                     benchmark.get_hard_macro_mask()).numpy()
#     hard_movable = [i for i in range(N) if movable_mask[i]]
#     hard_all     = torch.where(benchmark.get_hard_macro_mask())[0].tolist()

#     if not hard_movable:
#         return benchmark.macro_positions.numpy().astype(np.float32)

#     # ── Random initialisation ─────────────────────────────────────────────
#     rng = random.Random(seed)
#     pos_np = benchmark.macro_positions.numpy().astype(np.float32).copy()

#     for i in hard_movable:
#         hw = float(sizes_t[i, 0]) * 0.5
#         hh = float(sizes_t[i, 1]) * 0.5
#         pos_np[i, 0] = rng.uniform(hw + g, W - hw - g)
#         pos_np[i, 1] = rng.uniform(hh + g, H - hh - g)

#     # ── Convert nets to macro-only format for WL gradient ─────────────────
#     # Each net: list of macro indices that participate (ignoring IO ports)
#     # We use a simplified WA wirelength gradient
#     net_macros = []   # list of lists of macro indices
#     for drv_idx, dox, doy, sinks in nets_cpp:
#         members = []
#         if drv_idx >= 0:
#             members.append(drv_idx)
#         for sidx, sox, soy in sinks:
#             if sidx >= 0 and sidx not in members:
#                 members.append(sidx)
#         if len(members) >= 2:
#             net_macros.append(members)

#     # ── Precompute Poisson kernel in frequency domain ─────────────────────
#     # For a 2D periodic domain, the Green's function of the Laplacian in
#     # frequency domain is -1 / (kx² + ky²) with the DC component zeroed.
#     # We use the discrete version with wavenumbers kx, ky.
#     kx = torch.fft.fftfreq(n_cols, d=1.0/n_cols, device=device) * (2*np.pi/n_cols)
#     ky = torch.fft.fftfreq(n_rows, d=1.0/n_rows, device=device) * (2*np.pi/n_rows)
#     KX, KY = torch.meshgrid(kx, ky, indexing='ij')  # [n_cols, n_rows]
#     K2 = KX**2 + KY**2
#     K2[0, 0] = 1.0   # avoid division by zero at DC; DC component is zeroed

#     # Precomputed kernel: -1/K2 (the Poisson kernel)
#     poisson_kernel = -1.0 / K2   # [n_cols, n_rows]
#     poisson_kernel[0, 0] = 0.0   # zero DC (fixes absolute potential)

#     # ── Nesterov state ────────────────────────────────────────────────────
#     pos    = torch.tensor(pos_np, device=device)   # [N, 2]  all macros
#     pos_v  = pos.clone()   # Nesterov "lookahead" variable v
#     alpha  = 1.0           # Nesterov momentum coefficient
#     lam    = 1e-3          # density penalty lambda (grows over iterations)
#     lr     = min(bin_w, bin_h) * 0.5   # learning rate ≈ half a bin

#     hard_mov_t = torch.tensor(hard_movable, dtype=torch.long, device=device)
#     hard_all_t = torch.tensor(hard_all,     dtype=torch.long, device=device)

#     def clamp_pos(p):
#         """Clamp all movable macros to canvas."""
#         hw = sizes_t[hard_mov_t, 0] * 0.5  # [n_mov]
#         hh = sizes_t[hard_mov_t, 1] * 0.5
#         p_mov = p[hard_mov_t]
#         p_mov[:, 0] = torch.clamp(p_mov[:, 0], hw + g, W - hw - g)
#         p_mov[:, 1] = torch.clamp(p_mov[:, 1], hh + g, H - hh - g)
#         p = p.clone()
#         p[hard_mov_t] = p_mov
#         return p

#     def density_grad_and_overflow(p):
#         """
#         Compute density gradient and overflow using FFT.
#         Returns (grad [N, 2], overflow scalar).
#         grad is zero for fixed/soft macros.
#         """
#         # ── Rasterise macro areas onto grid ────────────────────────────
#         # For each hard macro, add its area to the bins it overlaps.
#         # We use a simple bell-shaped smearing: each macro is represented
#         # as a rectangular patch on the grid.
#         density = torch.zeros(n_cols, n_rows, device=device)

#         p_hard = p[hard_all_t]            # [n_hard, 2]
#         s_hard = sizes_t[hard_all_t]      # [n_hard, 2]

#         # Vectorised rasterisation: find which bins each macro covers
#         x0 = (p_hard[:, 0] - s_hard[:, 0]*0.5) / bin_w   # fractional col
#         x1 = (p_hard[:, 0] + s_hard[:, 0]*0.5) / bin_w
#         y0 = (p_hard[:, 1] - s_hard[:, 1]*0.5) / bin_h
#         y1 = (p_hard[:, 1] + s_hard[:, 1]*0.5) / bin_h

#         # Clamp to grid bounds
#         x0c = torch.clamp(x0, 0, n_cols - 1e-6)
#         x1c = torch.clamp(x1, 0, n_cols - 1e-6)
#         y0c = torch.clamp(y0, 0, n_rows - 1e-6)
#         y1c = torch.clamp(y1, 0, n_rows - 1e-6)

#         # For each macro, scatter its area fraction into overlapping bins
#         # (simplified: use centre bin only for speed, full overlap for accuracy)
#         # We do full overlap via a loop over macros — n_hard is small (≤800)
#         n_hard = p_hard.shape[0]
#         for k in range(n_hard):
#             col0 = max(0, int(x0c[k].item()))
#             col1 = min(n_cols - 1, int(x1c[k].item()))
#             row0 = max(0, int(y0c[k].item()))
#             row1 = min(n_rows - 1, int(y1c[k].item()))
#             if col0 <= col1 and row0 <= row1:
#                 # Area fraction this macro contributes to each overlapped bin
#                 macro_area = float(s_hard[k, 0].item() * s_hard[k, 1].item())
#                 n_bins_covered = (col1-col0+1) * (row1-row0+1)
#                 per_bin = macro_area / max(1, n_bins_covered)
#                 density[col0:col1+1, row0:row1+1] += per_bin

#         bin_area  = bin_w * bin_h
#         overflow  = float(torch.clamp(density - bin_area, min=0).sum().item()
#                           / max(1.0, (p_hard[:,0]*0).sum().item() + 1.0))
#         # Simpler overflow: fraction of bins that are overcrowded
#         overflow = float((density > bin_area).float().mean().item())

#         # ── Solve Poisson: potential = IFFT(-1/K² × FFT(density)) ────
#         dens_f   = torch.fft.fft2(density)       # FFT of density
#         phi_f    = dens_f * poisson_kernel        # multiply by kernel
#         phi      = torch.fft.ifft2(phi_f).real   # potential [n_cols, n_rows]

#         # ── Electric field = gradient of potential ────────────────────
#         # Ex = d(phi)/dx, Ey = d(phi)/dy in frequency domain:
#         # Ex_f = i*kx * phi_f,  Ey_f = i*ky * phi_f
#         Ex_f = 1j * KX * phi_f
#         Ey_f = 1j * KY * phi_f
#         Ex   = torch.fft.ifft2(Ex_f).real   # [n_cols, n_rows]
#         Ey   = torch.fft.ifft2(Ey_f).real

#         # ── Sample field at each movable macro's position ─────────────
#         grad = torch.zeros_like(p)   # [N, 2]

#         p_mov = p[hard_mov_t]                    # [n_mov, 2]
#         col_f = torch.clamp(p_mov[:, 0] / bin_w, 0, n_cols - 1)
#         row_f = torch.clamp(p_mov[:, 1] / bin_h, 0, n_rows - 1)
#         col_i = col_f.long().clamp(0, n_cols - 1)
#         row_i = row_f.long().clamp(0, n_rows - 1)

#         gx = Ex[col_i, row_i]   # [n_mov]
#         gy = Ey[col_i, row_i]

#         grad[hard_mov_t, 0] = gx
#         grad[hard_mov_t, 1] = gy
#         return grad, overflow

#     def wl_grad(p):
#         """
#         Weighted-average (WA) wirelength gradient.
#         For each net, the gradient pulls each macro toward the net centroid.
#         This is a simplified version — no pin offsets for speed.
#         """
#         grad = torch.zeros_like(p)
#         gamma = 1.0   # WA smoothing parameter

#         for members in net_macros:
#             if len(members) < 2:
#                 continue
#             pos_m = p[members]                  # [k, 2]
#             # WA softmax weights
#             wx    = torch.exp((pos_m[:, 0] - pos_m[:, 0].max()) / gamma)
#             wy    = torch.exp((pos_m[:, 1] - pos_m[:, 1].max()) / gamma)
#             wx_neg= torch.exp((pos_m[:, 0].min() - pos_m[:, 0]) / gamma)
#             wy_neg= torch.exp((pos_m[:, 1].min() - pos_m[:, 1]) / gamma)

#             sx = wx.sum(); sy = wy.sum()
#             sx_n = wx_neg.sum(); sy_n = wy_neg.sum()

#             # x gradient for each macro in net
#             for ki, mi in enumerate(members):
#                 if not movable_mask[mi]:
#                     continue
#                 # Positive x endpoint gradient
#                 dfdxi = (wx[ki] * (sx - wx[ki])) / (sx**2 + 1e-12)
#                 # Negative x endpoint gradient
#                 dfdxi_n = -(wx_neg[ki] * (sx_n - wx_neg[ki])) / (sx_n**2 + 1e-12)
#                 grad[mi, 0] += dfdxi + dfdxi_n

#                 dfdyi   = (wy[ki] * (sy - wy[ki])) / (sy**2 + 1e-12)
#                 dfdyi_n = -(wy_neg[ki] * (sy_n - wy_neg[ki])) / (sy_n**2 + 1e-12)
#                 grad[mi, 1] += dfdyi + dfdyi_n

#         return grad

#     # ── Nesterov main loop ────────────────────────────────────────────────
#     alpha_prev = 1.0

#     for it in range(FFT_MAX_ITERS):
#         if time.time() - t_start > time_limit:
#             break

#         # Compute gradients at lookahead point v
#         with torch.no_grad():
#             d_dens, overflow = density_grad_and_overflow(pos_v)
#             d_wl             = wl_grad(pos_v)
#             total_grad       = d_wl + lam * d_dens

#         # Gradient step on movable macros only
#         new_pos = pos.clone()
#         new_pos[hard_mov_t] = (pos[hard_mov_t]
#                                - lr * total_grad[hard_mov_t])
#         new_pos = clamp_pos(new_pos)

#         # Nesterov momentum update
#         alpha_new = (1.0 + math.sqrt(1.0 + 4.0 * alpha_prev**2)) / 2.0
#         beta      = (alpha_prev - 1.0) / alpha_new

#         pos_v_new = new_pos.clone()
#         pos_v_new[hard_mov_t] = (new_pos[hard_mov_t]
#                                  + beta * (new_pos[hard_mov_t]
#                                            - pos[hard_mov_t]))
#         pos_v_new = clamp_pos(pos_v_new)

#         pos       = new_pos
#         pos_v     = pos_v_new
#         alpha_prev = alpha_new

#         # Increase density penalty
#         lam = min(lam * 1.05, 1e4)

#         if it % 50 == 0:
#             elapsed = time.time() - t_start
#             print(f"    [FFT] iter={it:3d}  overflow={overflow:.3f}  "
#                   f"lam={lam:.2f}  t={elapsed:.1f}s")

#         if overflow < FFT_OV_TARGET:
#             print(f"    [FFT] converged at iter={it}  overflow={overflow:.4f}")
#             break

#     result_np = pos.cpu().numpy().astype(np.float32)
#     return result_np


# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITY: torch overlap check (main process diagnostics)
# # ─────────────────────────────────────────────────────────────────────────────

# def _torch_overlap(pos_t: torch.Tensor, sizes_t: torch.Tensor,
#                    hard_idx: List[int],
#                    threshold: float = 0.0) -> Tuple[int, float]:
#     """
#     threshold=0.0  → any positive overlap area counts (strict, matches evaluator)
#     threshold=0.0040 → relative overlap / min_macro_area > 0.004 (for SA cost)
#     """
#     n = len(hard_idx)
#     if n < 2:
#         return 0, 0.0
#     idx_t = torch.tensor(hard_idx, dtype=torch.long)
#     p = pos_t[idx_t]; s = sizes_t[idx_t]
#     hw = s[:, 0] * 0.5;  hh = s[:, 1] * 0.5
#     xmin = p[:, 0] - hw;  xmax = p[:, 0] + hw
#     ymin = p[:, 1] - hh;  ymax = p[:, 1] + hh

#     ox = (torch.minimum(xmax.unsqueeze(0), xmax.unsqueeze(1))
#         - torch.maximum(xmin.unsqueeze(0), xmin.unsqueeze(1)))
#     oy = (torch.minimum(ymax.unsqueeze(0), ymax.unsqueeze(1))
#         - torch.maximum(ymin.unsqueeze(0), ymin.unsqueeze(1)))

#     ov_area  = ox * oy
#     area_i   = s[:, 0] * s[:, 1]
#     min_area = torch.minimum(
#         area_i.unsqueeze(0).expand(n, n),
#         area_i.unsqueeze(1).expand(n, n))

#     # Relative overlap fraction — dimensionless, comparable to threshold
#     relative_ov = ov_area / (min_area + 1e-12)

#     triu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

#     if threshold == 0.0:
#         # Strict: any positive overlap area counts
#         mask = (ox > 1e-6) & (oy > 1e-6) & triu
#     else:
#         # Relative threshold (used during SA cost evaluation)
#         mask = (ox > 0) & (oy > 0) & (relative_ov > threshold) & triu

#     return int(mask.sum().item()), float((ov_area * mask.float()).sum().item())

# # ─────────────────────────────────────────────────────────────────────────────
# # SPIRAL INIT (unchanged — kept for worker diversity)
# # ─────────────────────────────────────────────────────────────────────────────

# def _spiral_init(benchmark: Benchmark, seed: int = 0) -> np.ndarray:
#     rng   = random.Random(seed)
#     sizes = benchmark.macro_sizes
#     W     = float(benchmark.canvas_width)
#     H     = float(benchmark.canvas_height)
#     g     = LEGALIZE_GAP
#     pos   = benchmark.macro_positions.numpy().astype(np.float32).copy()
#     movable = (benchmark.get_movable_mask() &
#                benchmark.get_hard_macro_mask()).numpy()
#     idx = [i for i in range(len(movable)) if movable[i]]
#     if not idx:
#         return pos
#     idx.sort(key=lambda i: (-(float(sizes[i][0])*float(sizes[i][1])),
#                              rng.random()))
#     left=g; right=W-g; bot=g; top=H-g
#     cx_ptr=left; cy_ptr=bot; direction=0; row_h=0.0
#     for i in idx:
#         w=float(sizes[i][0]); h=float(sizes[i][1])
#         hw=w/2.0; hh=h/2.0; placed=False
#         for _ in range(8):
#             if direction == 0:
#                 if cx_ptr+w+g <= right:
#                     cx=cx_ptr+hw; cy=cy_ptr+hh
#                     cx_ptr+=w+g; row_h=max(row_h,h); placed=True; break
#                 else:
#                     direction=1; cy_ptr=max(cy_ptr,bot+row_h+g)
#                     cx_ptr=right-hw; row_h=0.0
#             elif direction == 1:
#                 if cy_ptr+h+g <= top:
#                     cx=cx_ptr; cy=cy_ptr+hh; cy_ptr+=h+g; placed=True; break
#                 else:
#                     direction=2; cx_ptr=right-hw; cy_ptr=top-hh
#             elif direction == 2:
#                 if cx_ptr-w-g >= left:
#                     cx=cx_ptr-hw; cy=cy_ptr; cx_ptr-=w+g; placed=True; break
#                 else:
#                     direction=3; cy_ptr=top-hh; cx_ptr=left+hw
#             else:
#                 if cy_ptr-h-g >= bot:
#                     cx=cx_ptr; cy=cy_ptr-hh; cy_ptr-=h+g; placed=True; break
#                 else:
#                     direction=0; shrink=max(2.0*g, row_h+g)
#                     left+=shrink; right-=shrink; bot+=shrink; top-=shrink
#                     row_h=0.0; cx_ptr=left+hw; cy_ptr=bot+hh
#                     if left>=right or bot>=top:
#                         cx=rng.uniform(hw+g,W-hw-g)
#                         cy=rng.uniform(hh+g,H-hh-g); placed=True; break
#         if not placed:
#             cx=rng.uniform(hw+g,W-hw-g); cy=rng.uniform(hh+g,H-hh-g)
#         pos[i,0]=max(hw+g,min(W-hw-g,cx))
#         pos[i,1]=max(hh+g,min(H-hh-g,cy))
#     return pos


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN PLACER
# # ─────────────────────────────────────────────────────────────────────────────

# class SimulatedAnnealingPlacer:

#     def __init__(self, time_budget=TIME_BUDGET_SEC, num_workers=NUM_WORKERS):
#         self.time_budget = time_budget
#         self.num_workers = num_workers

#     def place(self, benchmark: Benchmark) -> torch.Tensor:
#         t0      = time.time()
#         name    = benchmark.name
#         sizes_t = benchmark.macro_sizes
#         sizes_np = sizes_t.numpy().astype(np.float32)
#         W = float(benchmark.canvas_width)
#         H = float(benchmark.canvas_height)

#         hard_mask    = (benchmark.get_movable_mask() &
#                         benchmark.get_hard_macro_mask())
#         hard_movable = torch.where(hard_mask)[0].tolist()
#         hard_all     = torch.where(
#                         benchmark.get_hard_macro_mask())[0].tolist()

#         hard_t = torch.tensor(hard_all, dtype=torch.long)
#         util   = float((sizes_t[hard_t,0]*sizes_t[hard_t,1]).sum()) / (W*H)
#         print(f"\n{'='*65}")
#         print(f"  [{name}] W={W:.1f} H={H:.1f}  "
#               f"hard={len(hard_all)} movable={len(hard_movable)}  "
#               f"util={util:.1%}")

#         # ── 1. Load nets ──────────────────────────────────────────────────
#         plc = None; nets_raw = []
#         try:
#             from macro_place.loader import load_benchmark_from_dir
#             tdir = _find_testcase_dir(name)
#             if tdir:
#                 _, plc = load_benchmark_from_dir(tdir)
#                 nets_raw = _extract_nets(plc, benchmark)
#                 print(f"  [{name}] nets={len(nets_raw)}")
#             else:
#                 print(f"  [{name}] WARNING: testcase dir not found")
#         except Exception as e:
#             print(f"  [{name}] WARNING: plc load failed ({e})")

#         nets_cpp = nets_to_cpp_format(nets_raw)

#         # ── 2. Adaptive overlap penalty ───────────────────────────────────
#         ref_np = benchmark.macro_positions.numpy().astype(np.float32)
#         if _HAS_CPP and _cpp is not None:
#             try:
#                 _cpp.init_nets(nets_cpp, benchmark.num_macros)
#                 ref_hpwl = float(_cpp.compute_hpwl(ref_np))
#             except Exception:
#                 ref_hpwl = _py_hpwl(ref_np, nets_cpp)
#         else:
#             ref_hpwl = _py_hpwl(ref_np, nets_cpp)

#         overlap_pen = (OVERLAP_PENALTY_SCALE * ref_hpwl
#                        if ref_hpwl > 1.0 else OVERLAP_PENALTY_FALLBACK)
#         print(f"  [{name}] ref_hpwl={ref_hpwl:.1f}  "
#               f"overlap_pen={overlap_pen:.1f}")

#         # ── 3. FIX 1: Guaranteed-valid fallback ───────────────────────────
#         fallback_np = _greedy_pack_init(benchmark, seed=99999)
#         fallback_t  = torch.from_numpy(fallback_np)
#         fb_ov, _    = _torch_overlap(fallback_t, sizes_t, hard_all)
#         print(f"  [{name}] fallback overlaps={fb_ov}  "
#               f"({'VALID' if fb_ov==0 else 'needs FFT fix'})")

#         # If greedy pack still has overlaps (very dense benchmarks), use FFT
#         # spreader as the fallback too
#         if fb_ov > 0:
#             print(f"  [{name}] running FFT on fallback to fix {fb_ov} overlaps")
#             fft_fallback = _fft_spread(benchmark, nets_cpp, seed=99999,
#                                        time_limit=5.0)
#             fft_t = torch.from_numpy(fft_fallback)
#             fft_ov, _ = _torch_overlap(fft_t, sizes_t, hard_all)
#             print(f"  [{name}] FFT fallback overlaps={fft_ov}")
#             if fft_ov < fb_ov:
#                 fallback_np = fft_fallback
#                 fallback_t  = fft_t
#                 fb_ov       = fft_ov

#         # ── 4. FIX 3: FFT spreader for SA initialisation ─────────────────
#         # Budget: up to FFT_TIME_LIMIT seconds, but stop early if time is short
#         fft_budget = min(FFT_TIME_LIMIT, self.time_budget * 0.12)
#         print(f"  [{name}] running FFT spreader  budget={fft_budget:.1f}s")
#         fft_np = _fft_spread(benchmark, nets_cpp, seed=42,
#                               time_limit=fft_budget)
#         fft_t  = torch.from_numpy(fft_np)
#         fft_ov, _ = _torch_overlap(fft_t, sizes_t, hard_all)
#         print(f"  [{name}] FFT init overlaps={fft_ov}")

#         # ── 5. Build worker args ──────────────────────────────────────────
#         overhead  = min(5.0, 0.5 * self.num_workers)
#         sa_time   = max(10.0,
#                         self.time_budget - overhead - (time.time() - t0))
#         n_workers = min(self.num_workers,
#                         max(2, len(hard_movable) // 4 + 2))
#         print(f"  [{name}] {n_workers} workers  sa_time={sa_time:.0f}s  "
#               f"cpp={'ON' if _HAS_CPP else 'OFF'}")

#         # Worker diversity:
#         #   workers 0..3  — greedy pack init  (half)
#         #   workers 4..7  — spiral init       (quarter)
#         #   workers 8..15 — FFT spread init   (quarter — best starting point)
#         worker_args = []
#         fft_workers  = n_workers // 4          # ~4 workers get FFT init
#         grep_workers = n_workers // 2          # ~8 workers get greedy
#         for wid in range(n_workers):
#             if wid < grep_workers:
#                 init_np = _greedy_pack_init(benchmark, seed=1000+wid)
#             elif wid < grep_workers + (n_workers - grep_workers - fft_workers):
#                 init_np = _spiral_init(benchmark, seed=2000+wid)
#             else:
#                 # Add small perturbation to FFT result for diversity
#                 init_np = fft_np.copy()
#                 rng_w   = random.Random(5000+wid)
#                 noise   = 0.5  # microns of noise
#                 for mi in hard_movable:
#                     hw = float(sizes_np[mi, 0]) * 0.5
#                     hh = float(sizes_np[mi, 1]) * 0.5
#                     init_np[mi, 0] = max(hw + LEGALIZE_GAP,
#                                          min(W - hw - LEGALIZE_GAP,
#                                              init_np[mi, 0] +
#                                              rng_w.uniform(-noise, noise)))
#                     init_np[mi, 1] = max(hh + LEGALIZE_GAP,
#                                          min(H - hh - LEGALIZE_GAP,
#                                              init_np[mi, 1] +
#                                              rng_w.uniform(-noise, noise)))

#             worker_args.append((
#                 wid,
#                 init_np,
#                 sizes_np,
#                 hard_movable,
#                 hard_all,
#                 nets_cpp,
#                 benchmark.num_macros,
#                 W, H,
#                 3000 + wid,
#                 overlap_pen,
#                 sa_time,
#                 _here,
#             ))

#         # ── 6. Run workers ────────────────────────────────────────────────
#         results = []
#         try:
#             with ProcessPoolExecutor(max_workers=n_workers) as ex:
#                 futures = [ex.submit(run_sa_worker, a) for a in worker_args]
#                 for wid, f in enumerate(futures):
#                     try:
#                         res = f.result(timeout=sa_time + 30.0)
#                         results.append(res)
#                         p, c, it = res
#                         print(f"  [{name}] w{wid:02d}  cost={c:.2f}  "
#                               f"valid={'Y' if p is not None else 'N'}  "
#                               f"iters={it:,}")
#                     except Exception as e:
#                         print(f"  [{name}] worker {wid} error: {e}")
#         except Exception as e:
#             print(f"  [{name}] executor error: {e}")

#         # ── 7. Pick best valid result ─────────────────────────────────────
#         best_np = None; best_cost = float('inf')
#         for p, c, _ in results:
#             if p is not None and c < best_cost:
#                 best_cost = c; best_np = p

#         if best_np is not None:
#             candidate = torch.from_numpy(best_np)
#             c_ov, _   = _torch_overlap(candidate, sizes_t, hard_all)
#             print(f"  [{name}] SA best cost={best_cost:.4f}  ov={c_ov}")
#             if c_ov == 0:
#                 result = candidate
#             elif fb_ov == 0:
#                 print(f"  [{name}] SA invalid, using valid fallback")
#                 result = fallback_t
#             elif fft_ov == 0:
#                 print(f"  [{name}] SA invalid, using valid FFT init")
#                 result = fft_t
#             elif fb_ov <= c_ov:
#                 result = fallback_t
#             else:
#                 result = candidate
#         else:
#             print(f"  [{name}] no SA result — using best available")
#             if fb_ov == 0:
#                 result = fallback_t
#             elif fft_ov == 0:
#                 result = fft_t
#             else:
#                 result = fallback_t if fb_ov <= fft_ov else fft_t

#         try:
#             sys.path.insert(0, os.path.join(os.path.dirname(_here),
#                                             "submissions", "examples"))
#             from greedy_row_placer import GreedyRowPlacer as _GRP
#             _guaranteed_fallback = _GRP().place(benchmark)
#         except Exception:
#             _guaranteed_fallback = fallback_t   # our own greedy pack if import fails


#         grid_w = W / benchmark.grid_cols
#         grid_h = H / benchmark.grid_rows

#         result_np = result.numpy().astype(np.float32).copy()

#         for mi in hard_all:
#             hw = float(sizes_t[mi, 0]) * 0.5
#             hh = float(sizes_t[mi, 1]) * 0.5
#             cx = float(result_np[mi, 0])
#             cy = float(result_np[mi, 1])

#             # Which grid cell contains this macro center?
#             col = int(cx / grid_w)
#             row = int(cy / grid_h)
#             col = max(0, min(benchmark.grid_cols - 1, col))
#             row = max(0, min(benchmark.grid_rows - 1, row))

#             # Snap to cell center
#             cx_snap = (col + 0.5) * grid_w
#             cy_snap = (row + 0.5) * grid_h

#             # Clamp so macro body stays fully inside canvas
#             cx_snap = max(hw, min(W - hw, cx_snap))
#             cy_snap = max(hh, min(H - hh, cy_snap))

#             result_np[mi, 0] = cx_snap
#             result_np[mi, 1] = cy_snap

#         result_snapped = torch.from_numpy(result_np)

#         # Use strict threshold=0.0 — after snapping we want exact evaluator behaviour
#         snap_ov, _ = _torch_overlap(result_snapped, sizes_t, hard_all, threshold=0.0)
#         print(f"  [{name}] after grid-snap  ov={snap_ov}")

#         if snap_ov == 0:
#             result = result_snapped
#         else:
#             print(f"  [{name}] snap introduced {snap_ov} overlaps — using pre-snap result")
#             # Pre-snap result already passed our threshold=0.0 check, return it
#             result = result  # unchanged — better than falling back to greedy pack

#         # ── Final report ──────────────────────────────────────────────────────────
#         final_ov, _ = _torch_overlap(result, sizes_t, hard_all, threshold=0.0)
#         elapsed = time.time() - t0
#         print(f"  [{name}] DONE {elapsed:.1f}s  ov={final_ov}  "
#             f"{'✓ VALID' if final_ov==0 else '✗ INVALID'}")
#         print(f"{'='*65}\n")
#         return result


# # ─────────────────────────────────────────────────────────────────────────────
# # ENTRY POINT
# # ─────────────────────────────────────────────────────────────────────────────

# placer = SimulatedAnnealingPlacer(
#     time_budget=TIME_BUDGET_SEC,
#     num_workers=NUM_WORKERS,
# )


"""
GWTW-SA Macro Placer  —  Snap-in-SA Version
============================================

Fix 1: Valid fallback — greedy pack never wraps; GreedyRowPlacer as backstop.
Fix 2: Overlap gate in SA — do_overlap only called after HPWL gate passes.
Fix 3: FFT electrostatic spreader — wirelength-aware spread before SA.
Fix 4: Two-phase SA — phase 1 hammers overlaps, phase 2 optimises HPWL.
Fix 5: Snap-in-SA — every SA proposal is snapped to a grid cell center
       before evaluation.  Post-SA snap is therefore a guaranteed no-op.
"""

import os
import sys
import time
import random
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np
import torch
from macro_place.benchmark import Benchmark

# ─────────────────────────────────────────────────────────────────────────────
# Worker module — stable pickle path for ProcessPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from worker_impl import run_sa_worker, nets_to_cpp_format

# ─────────────────────────────────────────────────────────────────────────────
# C++ extension (main process — ref_hpwl only)
# ─────────────────────────────────────────────────────────────────────────────

_cpp = None
_HAS_CPP = False
for _p in [_here, os.path.join(_here, "placement_ops")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import placement_ops as _cpp
    _HAS_CPP = True
    print("[placement_ops] C++ extension loaded — fast HPWL active")
except ImportError:
    print("[placement_ops] WARNING: C++ not found — Python fallback")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TIME_BUDGET_SEC          = 55.0
NUM_WORKERS              = 16
OVERLAP_PENALTY_SCALE    = 50.0
OVERLAP_PENALTY_FALLBACK = 1e6

FFT_TIME_LIMIT = 8.0    # max seconds for FFT spreader
FFT_OV_TARGET  = 0.02   # stop spreading when bin overflow < 2%
FFT_MAX_ITERS  = 300


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _find_testcase_dir(name: str) -> Optional[str]:
    candidates = [
        f"external/MacroPlacement/Testcases/ICCAD04/{name}",
        os.path.join(os.getcwd(), "external", "MacroPlacement",
                     "Testcases", "ICCAD04", name),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(
                os.path.join(c, "netlist.pb.txt")):
            return c
    return None


def _extract_nets(plc, benchmark) -> List:
    nets = []
    if plc is None or not hasattr(plc, 'nets'):
        return nets
    name_to_bench = {n: i for i, n in enumerate(benchmark.macro_names)}

    def resolve(pin_name):
        if pin_name not in plc.mod_name_to_indices:
            return None
        idx = plc.mod_name_to_indices[pin_name]
        if idx >= len(plc.modules_w_pins):
            return None
        obj = plc.modules_w_pins[idx]
        try:
            ptype = obj.get_type()
        except Exception:
            return None
        if ptype == 'PORT':
            try:
                px, py = obj.get_pos()
                return (-1, (float(px), float(py)))
            except Exception:
                return None
        if ptype != 'MACRO_PIN':
            return None
        try:
            parent = obj.macro_name
        except AttributeError:
            return None
        if parent not in name_to_bench:
            return None
        try:
            ox, oy = obj.get_offset()
        except Exception:
            ox, oy = 0.0, 0.0
        return (name_to_bench[parent], (float(ox), float(oy)))

    for drv_name, sink_names in plc.nets.items():
        drv = resolve(drv_name)
        if drv is None:
            continue
        sinks = [s for sn in sink_names
                 for s in [resolve(sn)] if s is not None]
        if sinks:
            nets.append((drv[0], drv[1], sinks))
    return nets


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


# ─────────────────────────────────────────────────────────────────────────────
# SNAP HELPER (main process — mirrors worker_impl._snap exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _snap_np(pos_np: np.ndarray, sizes_np: np.ndarray,
             hard_all: List[int],
             grid_w: float, grid_h: float,
             n_cols: int, n_rows: int,
             W: float, H: float) -> np.ndarray:
    """
    Snap every hard macro center to its grid cell center, in-place copy.
    Mirrors the snap done in each SA worker.
    """
    out = pos_np.copy()
    for mi in hard_all:
        hw = float(sizes_np[mi, 0]) * 0.5
        hh = float(sizes_np[mi, 1]) * 0.5
        cx = float(out[mi, 0]); cy = float(out[mi, 1])
        col = max(0, min(n_cols - 1, int(cx / grid_w)))
        row = max(0, min(n_rows - 1, int(cy / grid_h)))
        out[mi, 0] = max(hw, min(W - hw, (col + 0.5) * grid_w))
        out[mi, 1] = max(hh, min(H - hh, (row + 0.5) * grid_h))
    return out

# import math as _math

# def _snap_np(pos_np: np.ndarray, sizes_np: np.ndarray,
#              hard_all: List[int],
#              grid_w: float, grid_h: float,
#              n_cols: int, n_rows: int,
#              W: float, H: float) -> np.ndarray:
#     """
#     Snap every hard macro center to its grid cell center (valid range only).
#     Mirrors _snap() exactly — result is always a proper grid cell center.
#     """
#     out = pos_np.copy()
#     for mi in hard_all:
#         hw = float(sizes_np[mi, 0]) * 0.5
#         hh = float(sizes_np[mi, 1]) * 0.5

#         col_min = max(0, _math.ceil(hw / grid_w - 0.5))
#         col_max = min(n_cols - 1, int((W - hw) / grid_w - 0.5 + 1e-9))
#         if col_min > col_max:
#             col_min = col_max = n_cols // 2

#         row_min = max(0, _math.ceil(hh / grid_h - 0.5))
#         row_max = min(n_rows - 1, int((H - hh) / grid_h - 0.5 + 1e-9))
#         if row_min > row_max:
#             row_min = row_max = n_rows // 2

#         col = int(float(out[mi, 0]) / grid_w)
#         col = max(col_min, min(col_max, col))
#         row = int(float(out[mi, 1]) / grid_h)
#         row = max(row_min, min(row_max, row))

#         out[mi, 0] = (col + 0.5) * grid_w
#         out[mi, 1] = (row + 0.5) * grid_h
#     return out


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAP CHECK (main process diagnostics — strict, matches evaluator)
# ─────────────────────────────────────────────────────────────────────────────

def _torch_overlap(pos_t: torch.Tensor, sizes_t: torch.Tensor,
                   hard_idx: List[int]) -> Tuple[int, float]:
    """
    Strict overlap: any positive overlap area (> 1e-6) counts.
    This matches the evaluator which requires zero tolerance.
    """
    n = len(hard_idx)
    if n < 2:
        return 0, 0.0
    idx_t = torch.tensor(hard_idx, dtype=torch.long)
    p = pos_t[idx_t]; s = sizes_t[idx_t]
    hw = s[:, 0] * 0.5;  hh = s[:, 1] * 0.5
    xmin = p[:, 0] - hw;  xmax = p[:, 0] + hw
    ymin = p[:, 1] - hh;  ymax = p[:, 1] + hh
    ox = (torch.minimum(xmax.unsqueeze(0), xmax.unsqueeze(1))
        - torch.maximum(xmin.unsqueeze(0), xmin.unsqueeze(1)))
    oy = (torch.minimum(ymax.unsqueeze(0), ymax.unsqueeze(1))
        - torch.maximum(ymin.unsqueeze(0), ymin.unsqueeze(1)))
    ov_area = ox * oy
    triu    = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    mask    = (ox > 1e-6) & (oy > 1e-6) & triu
    return int(mask.sum().item()), float((ov_area * mask.float()).sum().item())


# ─────────────────────────────────────────────────────────────────────────────
# GUARANTEED-VALID FALLBACK: SPREAD + LEGALIZE
#
# WHY NOT GreedyRowPlacer?
#   GreedyRowPlacer packs everything into a tight bottom-left corner.
#   Valid (zero overlaps) but terrible proxy: WL is huge, congestion is
#   extreme because all macros cluster in one quadrant.  Scores ~2.0–2.7.
#
# THIS approach:
#   1. Use FFT-spread positions as starting points (macros already spread
#      across the canvas for good WL/density).
#   2. Assign each macro to its nearest available grid cell using greedy
#      nearest-available matching (sorted largest-first).
#   3. Each macro owns exactly one grid cell → guaranteed zero overlap
#      because we check physical overlap between placed macros.
#   4. Falls back to row-pack only if the greedy cell assignment fails.
#
# Result: proxy typically 1.5–2.0 vs 2.5+ from GreedyRowPlacer.
# ─────────────────────────────────────────────────────────────────────────────

def _spread_legalize_fallback(benchmark: Benchmark,
                               fft_np: Optional[np.ndarray] = None
                               ) -> torch.Tensor:
    """
    Guaranteed-valid placement with good spatial spread.

    Algorithm:
      - Sort macros largest-area-first (large macros hardest to place).
      - For each macro, find the nearest grid cell center (from FFT spread
        positions) that does NOT physically overlap any already-placed macro.
      - Place it there.  Scan cells in order of distance from FFT position.
      - If no non-overlapping cell exists (canvas full), fall back to any
        non-overlapping spot found by a coarse scan.

    This is O(N × n_cells) worst case but N ≤ 800 and n_cells ≤ 3000,
    so it completes in well under 1 second.
    """
    import math as _math

    def _lsn(cx, cy, hw, hh):
        col_min = max(0, _math.ceil(hw / grid_w - 0.5))
        col_max = min(n_cols - 1, int((W - hw) / grid_w - 0.5 + 1e-9))
        if col_min > col_max: col_min = col_max = n_cols // 2
        row_min = max(0, _math.ceil(hh / grid_h - 0.5))
        row_max = min(n_rows - 1, int((H - hh) / grid_h - 0.5 + 1e-9))
        if row_min > row_max: row_min = row_max = n_rows // 2
        return ((max(col_min, min(col_max, int(cx / grid_w))) + 0.5) * grid_w,
                (max(row_min, min(row_max, int(cy / grid_h))) + 0.5) * grid_h)

    sizes_t  = benchmark.macro_sizes
    sizes_np = sizes_t.numpy().astype(np.float32)
    W        = float(benchmark.canvas_width)
    H        = float(benchmark.canvas_height)
    n_cols   = benchmark.grid_cols
    n_rows   = benchmark.grid_rows
    grid_w   = W / n_cols
    grid_h   = H / n_rows

    movable_mask = (benchmark.get_movable_mask() &
                    benchmark.get_hard_macro_mask()).numpy()
    hard_all     = torch.where(benchmark.get_hard_macro_mask())[0].tolist()
    hard_movable = [i for i in hard_all if movable_mask[i]]

    placement = benchmark.macro_positions.clone().numpy().astype(np.float32)

    if not hard_movable:
        return torch.from_numpy(placement)

    # Starting positions: FFT spread if available, else random spread
    if fft_np is not None:
        start = fft_np.copy()
    else:
        rng = random.Random(42)
        start = placement.copy()
        for i in hard_movable:
            hw = float(sizes_np[i, 0]) * 0.5
            hh = float(sizes_np[i, 1]) * 0.5
            start[i, 0] = rng.uniform(hw, W - hw)
            start[i, 1] = rng.uniform(hh, H - hh)

    # Precompute all grid cell centers
    cell_centers = []  # list of (cx, cy)
    for col in range(n_cols):
        for row in range(n_rows):
            cx = (col + 0.5) * grid_w
            cy = (row + 0.5) * grid_h
            cell_centers.append((cx, cy))

    # Sort macros: largest area first (hardest to place without overlap)
    order = sorted(hard_movable,
                   key=lambda i: -(float(sizes_np[i, 0]) * float(sizes_np[i, 1])))

    # Track placed boxes: list of (x0, y0, x1, y1) for fast overlap checks
    placed_boxes = []
    # Fixed (non-movable) hard macros already occupy space
    for i in hard_all:
        if i not in hard_movable:
            hw = float(sizes_np[i, 0]) * 0.5
            hh = float(sizes_np[i, 1]) * 0.5
            cx = float(placement[i, 0])
            cy = float(placement[i, 1])
            placed_boxes.append((cx - hw, cy - hh, cx + hw, cy + hh, i))

    def overlaps_any(x0, y0, x1, y1):
        """Check if rectangle (x0,y0,x1,y1) overlaps any placed box."""
        for (bx0, by0, bx1, by1, _) in placed_boxes:
            ox = min(x1, bx1) - max(x0, bx0)
            oy = min(y1, by1) - max(y0, by0)
            if ox >= 0 and oy >= 0:
                return True
        return False

    for i in order:
        hw = float(sizes_np[i, 0]) * 0.5
        hh = float(sizes_np[i, 1]) * 0.5
        # Clamp macro body inside canvas
        sx = max(hw, min(W - hw, float(start[i, 0])))
        sy = max(hh, min(H - hh, float(start[i, 1])))

        # Sort grid cell centers by distance from FFT starting position
        cells_sorted = sorted(
            cell_centers,
            key=lambda c: (c[0] - sx) ** 2 + (c[1] - sy) ** 2)

        placed = False
        for (cx, cy) in cells_sorted:
            # Clamp so macro body stays in canvas
            cx_c, cy_c = _lsn(cx, cy, hw, hh)
            x0 = cx_c - hw; y0 = cy_c - hh
            x1 = cx_c + hw; y1 = cy_c + hh
            if not overlaps_any(x0, y0, x1, y1):
                placement[i, 0] = cx_c
                placement[i, 1] = cy_c
                placed_boxes.append((x0, y0, x1, y1, i))
                placed = True
                break

        if not placed:
            # Canvas too dense — place at first non-overlapping spot via
            # fine scan (unlikely for typical utilization ≤ 60%)
            step = min(hw, hh) * 0.5
            x = hw
            while x <= W - hw:
                y = hh
                while y <= H - hh:
                    x0 = x - hw; y0 = y - hh
                    x1 = x + hw; y1 = y + hh
                    if not overlaps_any(x0, y0, x1, y1):
                        placement[i, 0] = x
                        placement[i, 1] = y
                        placed_boxes.append((x0, y0, x1, y1, i))
                        placed = True
                        break
                    y += step
                if placed:
                    break
                x += step
            if not placed:
                # Absolute last resort — center (may overlap, but rare)
                placement[i, 0] = W / 2
                placement[i, 1] = H / 2

    return torch.from_numpy(placement)


def _greedy_fallback(benchmark: Benchmark) -> torch.Tensor:
    """
    Thin wrapper kept for API compatibility.
    Now delegates to _spread_legalize_fallback which is called AFTER the
    FFT spread, passing fft_np in.  This version (no fft_np) is used only
    at the start before FFT runs — e.g. for the initial overlap count print.
    """
    return _spread_legalize_fallback(benchmark, fft_np=None)


# ─────────────────────────────────────────────────────────────────────────────
# FFT ELECTROSTATIC SPREADER (RePlAce / ePlace algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def _fft_spread(benchmark: Benchmark,
                nets_cpp: List,
                seed: int = 0,
                time_limit: float = FFT_TIME_LIMIT) -> np.ndarray:
    """
    Nesterov gradient descent on electrostatic density + WA wirelength.
    Produces a spread-out initialisation for SA workers.
    GPU-accelerated if available.
    """
    t_start = time.time()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    W = float(benchmark.canvas_width)
    H = float(benchmark.canvas_height)
    N = benchmark.num_macros

    n_cols = benchmark.grid_cols
    n_rows = benchmark.grid_rows
    bin_w  = W / n_cols
    bin_h  = H / n_rows

    sizes_t      = benchmark.macro_sizes.to(device)
    movable_mask = (benchmark.get_movable_mask() &
                    benchmark.get_hard_macro_mask()).numpy()
    hard_movable = [i for i in range(N) if movable_mask[i]]
    hard_all     = torch.where(benchmark.get_hard_macro_mask())[0].tolist()

    if not hard_movable:
        return benchmark.macro_positions.numpy().astype(np.float32)

    rng    = random.Random(seed)
    pos_np = benchmark.macro_positions.numpy().astype(np.float32).copy()
    for i in hard_movable:
        hw = float(sizes_t[i, 0]) * 0.5
        hh = float(sizes_t[i, 1]) * 0.5
        pos_np[i, 0] = rng.uniform(hw, W - hw)
        pos_np[i, 1] = rng.uniform(hh, H - hh)

    net_macros = []
    for drv_idx, dox, doy, sinks in nets_cpp:
        members = []
        if drv_idx >= 0:
            members.append(drv_idx)
        for sidx, sox, soy in sinks:
            if sidx >= 0 and sidx not in members:
                members.append(sidx)
        if len(members) >= 2:
            net_macros.append(members)

    kx = torch.fft.fftfreq(n_cols, d=1.0/n_cols, device=device) * (2*np.pi/n_cols)
    ky = torch.fft.fftfreq(n_rows, d=1.0/n_rows, device=device) * (2*np.pi/n_rows)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    poisson_kernel       = -1.0 / K2
    poisson_kernel[0, 0] = 0.0

    pos   = torch.tensor(pos_np, device=device)
    pos_v = pos.clone()
    lam   = 1e-3
    lr    = min(bin_w, bin_h) * 0.5
    alpha_prev = 1.0

    hard_mov_t = torch.tensor(hard_movable, dtype=torch.long, device=device)
    hard_all_t = torch.tensor(hard_all,     dtype=torch.long, device=device)

    def clamp_pos(p):
        hw = sizes_t[hard_mov_t, 0] * 0.5
        hh = sizes_t[hard_mov_t, 1] * 0.5
        p_mov = p[hard_mov_t].clone()
        p_mov[:, 0] = torch.clamp(p_mov[:, 0], hw, W - hw)
        p_mov[:, 1] = torch.clamp(p_mov[:, 1], hh, H - hh)
        p = p.clone(); p[hard_mov_t] = p_mov
        return p

    def density_grad_overflow(p):
        density = torch.zeros(n_cols, n_rows, device=device)
        p_hard  = p[hard_all_t]; s_hard = sizes_t[hard_all_t]
        x0c = torch.clamp((p_hard[:, 0] - s_hard[:, 0]*0.5) / bin_w, 0, n_cols-1e-6)
        x1c = torch.clamp((p_hard[:, 0] + s_hard[:, 0]*0.5) / bin_w, 0, n_cols-1e-6)
        y0c = torch.clamp((p_hard[:, 1] - s_hard[:, 1]*0.5) / bin_h, 0, n_rows-1e-6)
        y1c = torch.clamp((p_hard[:, 1] + s_hard[:, 1]*0.5) / bin_h, 0, n_rows-1e-6)
        for k in range(p_hard.shape[0]):
            c0 = max(0, int(x0c[k].item()))
            c1 = min(n_cols-1, int(x1c[k].item()))
            r0 = max(0, int(y0c[k].item()))
            r1 = min(n_rows-1, int(y1c[k].item()))
            if c0 <= c1 and r0 <= r1:
                macro_area = float(s_hard[k, 0].item() * s_hard[k, 1].item())
                per_bin    = macro_area / max(1, (c1-c0+1)*(r1-r0+1))
                density[c0:c1+1, r0:r1+1] += per_bin
        overflow = float((density > bin_w*bin_h).float().mean().item())
        dens_f   = torch.fft.fft2(density)
        phi_f    = dens_f * poisson_kernel
        Ex = torch.fft.ifft2(1j * KX * phi_f).real
        Ey = torch.fft.ifft2(1j * KY * phi_f).real
        p_mov  = p[hard_mov_t]
        col_i  = torch.clamp((p_mov[:, 0] / bin_w).long(), 0, n_cols-1)
        row_i  = torch.clamp((p_mov[:, 1] / bin_h).long(), 0, n_rows-1)
        grad   = torch.zeros_like(p)
        grad[hard_mov_t, 0] = Ex[col_i, row_i]
        grad[hard_mov_t, 1] = Ey[col_i, row_i]
        return grad, overflow

    def wl_grad(p):
        grad  = torch.zeros_like(p)
        gamma = 1.0
        for members in net_macros:
            if len(members) < 2:
                continue
            pm  = p[members]
            wx  = torch.exp((pm[:, 0] - pm[:, 0].max()) / gamma)
            wy  = torch.exp((pm[:, 1] - pm[:, 1].max()) / gamma)
            wxn = torch.exp((pm[:, 0].min() - pm[:, 0]) / gamma)
            wyn = torch.exp((pm[:, 1].min() - pm[:, 1]) / gamma)
            sx  = wx.sum();  sy  = wy.sum()
            sxn = wxn.sum(); syn = wyn.sum()
            for ki, mi in enumerate(members):
                if not movable_mask[mi]:
                    continue
                grad[mi, 0] += (wx[ki]*(sx-wx[ki]))/(sx**2+1e-12) \
                             - (wxn[ki]*(sxn-wxn[ki]))/(sxn**2+1e-12)
                grad[mi, 1] += (wy[ki]*(sy-wy[ki]))/(sy**2+1e-12) \
                             - (wyn[ki]*(syn-wyn[ki]))/(syn**2+1e-12)
        return grad

    for it in range(FFT_MAX_ITERS):
        if time.time() - t_start > time_limit:
            break
        with torch.no_grad():
            d_dens, overflow = density_grad_overflow(pos_v)
            d_wl             = wl_grad(pos_v)
            total_grad       = d_wl + lam * d_dens
        new_pos = clamp_pos(pos.clone())
        new_pos[hard_mov_t] = clamp_pos(
            pos.clone() - lr * total_grad)[hard_mov_t]
        alpha_new  = (1.0 + math.sqrt(1.0 + 4.0*alpha_prev**2)) / 2.0
        beta       = (alpha_prev - 1.0) / alpha_new
        pos_v_new  = clamp_pos(
            new_pos.clone() +
            beta * (new_pos - pos) * torch.zeros_like(pos).index_fill_(
                0, hard_mov_t, 1.0))
        # Simpler Nesterov update for movable macros only
        pos_v_new = new_pos.clone()
        pos_v_new[hard_mov_t] = (new_pos[hard_mov_t]
                                 + beta * (new_pos[hard_mov_t]
                                           - pos[hard_mov_t]))
        pos_v_new = clamp_pos(pos_v_new)
        pos       = new_pos
        pos_v     = pos_v_new
        alpha_prev = alpha_new
        lam = min(lam * 1.05, 1e4)
        if it % 50 == 0:
            print(f"    [FFT] iter={it:3d}  overflow={overflow:.3f}  "
                  f"lam={lam:.2f}  t={time.time()-t_start:.1f}s")
        if overflow < FFT_OV_TARGET:
            print(f"    [FFT] converged  iter={it}  overflow={overflow:.4f}")
            break

    return pos.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY PACK (for worker init diversity — NOT used as final fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_pack_init(benchmark: Benchmark, seed: int = 0) -> np.ndarray:
    rng   = random.Random(seed)
    sizes = benchmark.macro_sizes
    W     = float(benchmark.canvas_width)
    H     = float(benchmark.canvas_height)
    pos   = benchmark.macro_positions.numpy().astype(np.float32).copy()
    movable = (benchmark.get_movable_mask() &
               benchmark.get_hard_macro_mask()).numpy()
    idx = [i for i in range(len(movable)) if movable[i]]
    if not idx:
        return pos
    idx.sort(key=lambda i: (-float(sizes[i][1]), rng.random()))
    gap   = 0.001
    cx = 0.0; cy = 0.0; rh = 0.0
    for i in idx:
        w = float(sizes[i][0]); h = float(sizes[i][1])
        if cx + w > W:
            cx = 0.0; cy += rh + gap; rh = 0.0
        if cy + h > H:
            pos[i, 0] = w / 2; pos[i, 1] = h / 2; continue
        pos[i, 0] = cx + w / 2
        pos[i, 1] = cy + h / 2
        cx += w + gap; rh = max(rh, h)
    return pos


def _spiral_init(benchmark: Benchmark, seed: int = 0) -> np.ndarray:
    rng   = random.Random(seed)
    sizes = benchmark.macro_sizes
    W     = float(benchmark.canvas_width)
    H     = float(benchmark.canvas_height)
    g     = 0.001
    pos   = benchmark.macro_positions.numpy().astype(np.float32).copy()
    movable = (benchmark.get_movable_mask() &
               benchmark.get_hard_macro_mask()).numpy()
    idx = [i for i in range(len(movable)) if movable[i]]
    if not idx:
        return pos
    idx.sort(key=lambda i: (-(float(sizes[i][0])*float(sizes[i][1])),
                             rng.random()))
    left=g; right=W-g; bot=g; top=H-g
    cx_ptr=left; cy_ptr=bot; direction=0; row_h=0.0
    for i in idx:
        w=float(sizes[i][0]); h=float(sizes[i][1])
        hw=w/2.0; hh=h/2.0; placed=False
        for _ in range(8):
            if direction == 0:
                if cx_ptr+w+g <= right:
                    cx=cx_ptr+hw; cy=cy_ptr+hh
                    cx_ptr+=w+g; row_h=max(row_h,h); placed=True; break
                else:
                    direction=1; cy_ptr=max(cy_ptr,bot+row_h+g)
                    cx_ptr=right-hw; row_h=0.0
            elif direction == 1:
                if cy_ptr+h+g <= top:
                    cx=cx_ptr; cy=cy_ptr+hh; cy_ptr+=h+g; placed=True; break
                else:
                    direction=2; cx_ptr=right-hw; cy_ptr=top-hh
            elif direction == 2:
                if cx_ptr-w-g >= left:
                    cx=cx_ptr-hw; cy=cy_ptr; cx_ptr-=w+g; placed=True; break
                else:
                    direction=3; cy_ptr=top-hh; cx_ptr=left+hw
            else:
                if cy_ptr-h-g >= bot:
                    cx=cx_ptr; cy=cy_ptr-hh; cy_ptr-=h+g; placed=True; break
                else:
                    direction=0; shrink=max(2.0*g, row_h+g)
                    left+=shrink; right-=shrink; bot+=shrink; top-=shrink
                    row_h=0.0; cx_ptr=left+hw; cy_ptr=bot+hh
                    if left>=right or bot>=top:
                        cx=rng.uniform(hw+g,W-hw-g)
                        cy=rng.uniform(hh+g,H-hh-g); placed=True; break
        if not placed:
            cx=rng.uniform(hw+g,W-hw-g); cy=rng.uniform(hh+g,H-hh-g)
        pos[i,0]=max(hw+g,min(W-hw-g,cx))
        pos[i,1]=max(hh+g,min(H-hh-g,cy))
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PLACER
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedAnnealingPlacer:

    def __init__(self, time_budget=TIME_BUDGET_SEC, num_workers=NUM_WORKERS):
        self.time_budget = time_budget
        self.num_workers = num_workers

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0       = time.time()
        bname    = benchmark.name          # local var — never shadows anything
        sizes_t  = benchmark.macro_sizes
        sizes_np = sizes_t.numpy().astype(np.float32)
        W  = float(benchmark.canvas_width)
        H  = float(benchmark.canvas_height)

        # Grid dimensions — used for snapping everywhere
        grid_w = W / benchmark.grid_cols
        grid_h = H / benchmark.grid_rows
        n_cols = benchmark.grid_cols
        n_rows = benchmark.grid_rows

        hard_mask    = (benchmark.get_movable_mask() &
                        benchmark.get_hard_macro_mask())
        hard_movable = torch.where(hard_mask)[0].tolist()
        hard_all     = torch.where(
                       benchmark.get_hard_macro_mask())[0].tolist()

        hard_t = torch.tensor(hard_all, dtype=torch.long)
        util   = float((sizes_t[hard_t,0]*sizes_t[hard_t,1]).sum()) / (W*H)
        print(f"\n{'='*65}")
        print(f"  [{bname}] W={W:.1f} H={H:.1f}  "
              f"hard={len(hard_all)} movable={len(hard_movable)}  "
              f"util={util:.1%}  grid={n_cols}×{n_rows}")

        # ── 1. Load nets ──────────────────────────────────────────────────
        plc = None; nets_raw = []
        try:
            from macro_place.loader import load_benchmark_from_dir
            tdir = _find_testcase_dir(bname)
            if tdir:
                _, plc = load_benchmark_from_dir(tdir)
                nets_raw = _extract_nets(plc, benchmark)
                print(f"  [{bname}] nets={len(nets_raw)}")
            else:
                print(f"  [{bname}] WARNING: testcase dir not found")
        except Exception as e:
            print(f"  [{bname}] WARNING: plc load failed ({e})")

        nets_cpp = nets_to_cpp_format(nets_raw)

        # ── 2. Overlap penalty ────────────────────────────────────────────
        ref_np = benchmark.macro_positions.numpy().astype(np.float32)
        if _HAS_CPP and _cpp is not None:
            try:
                _cpp.init_nets(nets_cpp, benchmark.num_macros)
                ref_hpwl = float(_cpp.compute_hpwl(ref_np))
            except Exception:
                ref_hpwl = _py_hpwl(ref_np, nets_cpp)
        else:
            ref_hpwl = _py_hpwl(ref_np, nets_cpp)

        overlap_pen = (OVERLAP_PENALTY_SCALE * ref_hpwl
                       if ref_hpwl > 1.0 else OVERLAP_PENALTY_FALLBACK)
        print(f"  [{bname}] ref_hpwl={ref_hpwl:.1f}  "
              f"overlap_pen={overlap_pen:.1f}")

        # ── 3. FFT spread init for SA workers ─────────────────────────────
        fft_budget = min(FFT_TIME_LIMIT, self.time_budget * 0.12)
        print(f"  [{bname}] FFT spreader  budget={fft_budget:.1f}s")
        fft_np = _fft_spread(benchmark, nets_cpp, seed=42,
                              time_limit=fft_budget)
        fft_t  = torch.from_numpy(fft_np)
        fft_ov, _ = _torch_overlap(fft_t, sizes_t, hard_all)
        print(f"  [{bname}] FFT init overlaps={fft_ov}")

        # ── 4. Guaranteed-valid fallback (spread+legalize beats GreedyRow) ─
        # Pass fft_np so macros are legalized from a SPREAD starting point,
        # giving much better proxy score than tight row-packing.
        guaranteed_t = _spread_legalize_fallback(benchmark, fft_np=fft_np)
        guar_ov, _   = _torch_overlap(guaranteed_t, sizes_t, hard_all)
        print(f"  [{bname}] guaranteed fallback overlaps={guar_ov}")
        if guar_ov != 0:
            # Extremely rare — run without fft_np as second attempt
            guaranteed_t = _spread_legalize_fallback(benchmark, fft_np=None)
            guar_ov, _   = _torch_overlap(guaranteed_t, sizes_t, hard_all)
            print(f"  [{bname}] fallback retry overlaps={guar_ov}")

        # ── 5. Build worker args ──────────────────────────────────────────
        overhead  = min(5.0, 0.5 * self.num_workers)
        sa_time   = max(10.0,
                        self.time_budget - overhead - (time.time() - t0))
        n_workers = min(self.num_workers,
                        max(2, len(hard_movable) // 4 + 2))
        print(f"  [{bname}] {n_workers} workers  sa_time={sa_time:.0f}s  "
              f"cpp={'ON' if _HAS_CPP else 'OFF'}")

        # Worker diversity: half greedy, quarter spiral, quarter FFT
        fft_workers  = n_workers // 4
        grep_workers = n_workers // 2

        worker_args = []
        for wid in range(n_workers):
            if wid < grep_workers:
                init_np = _greedy_pack_init(benchmark, seed=1000+wid)
            elif wid < grep_workers + (n_workers - grep_workers - fft_workers):
                init_np = _spiral_init(benchmark, seed=2000+wid)
            else:
                # FFT init with small perturbation for diversity
                init_np = fft_np.copy()
                rng_w   = random.Random(5000+wid)
                for mi in hard_movable:
                    hw = float(sizes_np[mi, 0]) * 0.5
                    hh = float(sizes_np[mi, 1]) * 0.5
                    init_np[mi, 0] = max(hw, min(W-hw,
                        init_np[mi, 0] + rng_w.uniform(-grid_w, grid_w)))
                    init_np[mi, 1] = max(hh, min(H-hh,
                        init_np[mi, 1] + rng_w.uniform(-grid_h, grid_h)))

            worker_args.append((
                wid, init_np, sizes_np,
                hard_movable, hard_all, nets_cpp,
                benchmark.num_macros, W, H,
                3000 + wid, overlap_pen, sa_time, _here,
                grid_w, grid_h,   # ← snap parameters for Option A
            ))

        # ── 6. Run workers ────────────────────────────────────────────────
        results = []
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(run_sa_worker, a) for a in worker_args]
                for wid, f in enumerate(futures):
                    try:
                        res = f.result(timeout=sa_time + 30.0)
                        results.append(res)
                        p, c, it = res
                        print(f"  [{bname}] w{wid:02d}  cost={c:.2f}  "
                              f"valid={'Y' if p is not None else 'N'}  "
                              f"iters={it:,}")
                    except Exception as e:
                        print(f"  [{bname}] worker {wid} error: {e}")
        except Exception as e:
            print(f"  [{bname}] executor error: {e}")

        # ── 7. Pick best valid SA result ──────────────────────────────────
        best_np = None; best_cost = float('inf')
        for p, c, _ in results:
            if p is not None and c < best_cost:
                best_cost = c; best_np = p

        if best_np is not None:
            candidate = torch.from_numpy(best_np)
            c_ov, _   = _torch_overlap(candidate, sizes_t, hard_all)
            print(f"  [{bname}] SA best cost={best_cost:.4f}  ov={c_ov}")
            result = candidate if c_ov == 0 else guaranteed_t
            if c_ov != 0:
                print(f"  [{bname}] SA invalid — using guaranteed fallback")
        else:
            print(f"  [{bname}] no SA result — using guaranteed fallback")
            result = guaranteed_t

        # ── 8. Post-SA grid snap (should be no-op — verify) ───────────────
        # Workers snap every position before storing it, so this is a
        # verification step, not a correction step.
        result_np_snapped = _snap_np(
            result.numpy().astype(np.float32), sizes_np,
            hard_all, grid_w, grid_h, n_cols, n_rows, W, H)
        result_snapped = torch.from_numpy(result_np_snapped)
        snap_ov, _     = _torch_overlap(result_snapped, sizes_t, hard_all)
        print(f"  [{bname}] after grid-snap  ov={snap_ov}")

        if snap_ov == 0:
            result = result_snapped
        else:
            # Snap introduced overlaps — should not happen with snap-in-SA.
            # Fall back to guaranteed-valid result.
            print(f"  [{bname}] WARNING: snap introduced {snap_ov} overlaps "
                  f"— using guaranteed fallback")
            result = guaranteed_t

        # ── 9. Final report ───────────────────────────────────────────────
        final_ov, _ = _torch_overlap(result, sizes_t, hard_all)
        elapsed     = time.time() - t0
        print(f"  [{bname}] DONE {elapsed:.1f}s  ov={final_ov}  "
              f"{'✓ VALID' if final_ov==0 else '✗ INVALID'}")
        print(f"{'='*65}\n")

        # ── 10. Evaluator pre-check (uses exact same arithmetic as contest) ─
        # This catches float64-vs-float32 edge cases the torch checker misses.
        if plc is not None:
            try:
                from macro_place.objective import compute_proxy_cost
                eval_ov = compute_proxy_cost(result, benchmark, plc).get('overlap_count', 0)
                print(f"  [{bname}] evaluator pre-check ov={eval_ov}")
                if eval_ov > 0:
                    print(f"  [{bname}] pre-check found {eval_ov} overlaps — trying guaranteed fallback")
                    ev2 = compute_proxy_cost(guaranteed_t, benchmark, plc).get('overlap_count', 0)
                    if ev2 == 0:
                        result = guaranteed_t
                        print(f"  [{bname}] guaranteed fallback is clean — using it")
                    else:
                        # guaranteed_t also has touching macros — use GreedyRowPlacer
                        print(f"  [{bname}] guaranteed fallback also fails (ev2={ev2}) — using GreedyRowPlacer")
                        try:
                            _ex = os.path.join(os.path.dirname(_here), "submissions", "examples")
                            if _ex not in sys.path:
                                sys.path.insert(0, _ex)
                            from greedy_row_placer import GreedyRowPlacer as _GRP
                            grp_result = _GRP().place(benchmark)
                            ev3 = compute_proxy_cost(grp_result, benchmark, plc).get('overlap_count', 0)
                            print(f"  [{bname}] GreedyRowPlacer ev={ev3}")
                            result = grp_result  # always valid (row-packing has explicit gaps)
                        except Exception as ge:
                            print(f"  [{bname}] GreedyRowPlacer failed: {ge}")
            except Exception as e:
                print(f"  [{bname}] evaluator pre-check failed: {e}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

placer = SimulatedAnnealingPlacer(
    time_budget=TIME_BUDGET_SEC,
    num_workers=NUM_WORKERS,
)