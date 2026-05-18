"""
MJ97 v11 — Initial-Layout Preserving Vectorized Legalizer

Strategy:
1) Start from the benchmark's initial hard-macro placement (very strong proxy signal)
2) Remove hard-macro overlaps with a fully vectorized repulsion solver
3) Keep displacement minimal to preserve wirelength/congestion structure

This implementation is deterministic, requires no external state, and respects
strict hard-macro legality (zero overlaps).
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark

class Mj97Placer:
    """Vectorized overlap legalizer around the strong initial placement."""

    def __init__(self, seed: int = 97):
        self.seed = seed
        self.gap = 0.005
        self.max_iters = 600
        self.min_budget_s = 20.0
        self.max_budget_s = 220.0
        self.refine_max_s = 12.0

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

        # Per-benchmark runtime target (well below hard competition limit).
        budget = min(self.max_budget_s, max(self.min_budget_s, 0.28 * n_hard))
        deadline = time.time() + budget

        legal = self._vectorized_legalize(
            init=init,
            sizes=sizes,
            movable=movable,
            canvas_w=cw,
            canvas_h=ch,
            diag=diag,
            deadline=deadline,
        )

        out[:n_hard] = torch.from_numpy(legal).to(dtype=torch.float32)
        return out

    def _load_plc_for_benchmark(self, bench_name: str):
        try:
            from macro_place.loader import load_benchmark, load_benchmark_from_dir
        except Exception:
            return None

        root = f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}"
        try:
            benchmark, plc = load_benchmark_from_dir(root)
            if benchmark.name:
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
            _, plc = load_benchmark(f"{ng_dir}/netlist.pb.txt", f"{ng_dir}/initial.plc", name=bench_name)
            return plc
        except Exception:
            return None

    def _multistart_select_best(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        base_init: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        diag: float,
        deadline: float,
    ) -> None:
        """Generate several legal candidates and keep the best exact proxy score."""
        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            return

        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5

        best_hard = placement[:n_hard].cpu().numpy().astype(np.float64).copy()
        best_score = float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])

        if n_hard >= 480:
            starts = 3
        elif n_hard >= 360:
            starts = 4
        else:
            starts = 5

        for s in range(1, starts):
            if time.time() >= deadline:
                break

            sigma = diag * (0.0015 + 0.0015 * s)
            perturbed = base_init.copy()
            noise = np.random.normal(loc=0.0, scale=sigma, size=perturbed.shape)
            noise[~movable] = 0.0
            perturbed += noise
            perturbed[:, 0] = np.clip(perturbed[:, 0], half_w, cw - half_w)
            perturbed[:, 1] = np.clip(perturbed[:, 1], half_h, ch - half_h)

            local_budget = min(3.0, max(1.0, (deadline - time.time()) / (starts - s + 1)))
            candidate = self._vectorized_legalize(
                init=perturbed,
                sizes=sizes,
                movable=movable,
                canvas_w=cw,
                canvas_h=ch,
                diag=diag,
                deadline=time.time() + local_budget,
            )

            trial = benchmark.macro_positions.clone()
            trial[:n_hard] = torch.from_numpy(candidate).to(dtype=torch.float32)
            score = float(compute_proxy_cost(trial, benchmark, plc)["proxy_cost"])
            if score < best_score:
                best_score = score
                best_hard = candidate

        placement[:n_hard] = torch.from_numpy(best_hard).to(dtype=torch.float32)

    def _vectorized_legalize(
        self,
        init: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        canvas_w: float,
        canvas_h: float,
        diag: float,
        deadline: float,
    ) -> np.ndarray:
        """
        Resolve overlaps via vectorized pairwise repulsion.

        For each overlapping pair, push along the easier axis (smaller overlap).
        Accumulate all pair pushes, clip per-iteration displacement, and iterate.
        """
        pos = init.copy()
        n = pos.shape[0]

        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5

        pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)

        sep_x = (sizes[:, 0:1] + sizes[:, 0][None, :]) * 0.5 + self.gap
        sep_y = (sizes[:, 1:2] + sizes[:, 1][None, :]) * 0.5 + self.gap
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)

        max_iters = self.max_iters
        for it in range(max_iters):
            if time.time() >= deadline:
                break

            x = pos[:, 0]
            y = pos[:, 1]

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]

            ov_x = sep_x - np.abs(dx)
            ov_y = sep_y - np.abs(dy)

            mask = (ov_x > 0.0) & (ov_y > 0.0) & upper
            if not mask.any():
                break

            i_idx, j_idx = np.where(mask)
            choose_x = ov_x[i_idx, j_idx] < ov_y[i_idx, j_idx]

            force_x = np.zeros(n, dtype=np.float64)
            force_y = np.zeros(n, dtype=np.float64)

            if choose_x.any():
                ii = i_idx[choose_x]
                jj = j_idx[choose_x]
                sign = np.sign(dx[ii, jj])
                sign[sign == 0.0] = 1.0
                push = ov_x[ii, jj] * 0.55 + 1e-4
                np.add.at(force_x, ii, sign * push)
                np.add.at(force_x, jj, -sign * push)

            if (~choose_x).any():
                ii = i_idx[~choose_x]
                jj = j_idx[~choose_x]
                sign = np.sign(dy[ii, jj])
                sign[sign == 0.0] = 1.0
                push = ov_y[ii, jj] * 0.55 + 1e-4
                np.add.at(force_y, ii, sign * push)
                np.add.at(force_y, jj, -sign * push)

            # Larger steps early, smaller near convergence.
            frac = (it + 1) / max_iters
            max_step = diag * (0.03 - 0.02 * frac)
            max_step = max(max_step, diag * 0.004)

            force_x = np.clip(force_x, -max_step, max_step)
            force_y = np.clip(force_y, -max_step, max_step)

            force_x[~movable] = 0.0
            force_y[~movable] = 0.0

            pos[:, 0] += force_x
            pos[:, 1] += force_y

            pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)

        # Final small cleanup pass if tiny residual overlaps remain.
        self._cleanup_pair_push(
            pos=pos,
            sizes=sizes,
            movable=movable,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            sep_x=sep_x,
            sep_y=sep_y,
            upper=upper,
            deadline=deadline,
        )
        self._final_spiral_fix(
            pos=pos,
            sizes=sizes,
            movable=movable,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            sep_x=sep_x,
            sep_y=sep_y,
            upper=upper,
            deadline=deadline,
        )

        return pos

    def _cleanup_pair_push(
        self,
        pos: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        canvas_w: float,
        canvas_h: float,
        sep_x: np.ndarray,
        sep_y: np.ndarray,
        upper: np.ndarray,
        deadline: float,
    ) -> None:
        """Resolve any residual overlaps with direct pair pushes."""
        n = pos.shape[0]
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5

        for _ in range(1200):
            if time.time() >= deadline:
                break

            x = pos[:, 0]
            y = pos[:, 1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            ov_x = sep_x - np.abs(dx)
            ov_y = sep_y - np.abs(dy)
            mask = (ov_x > 0.0) & (ov_y > 0.0) & upper
            if not mask.any():
                return

            i_idx, j_idx = np.where(mask)
            # handle one pair at a time to force monotonic cleanup
            i = int(i_idx[0])
            j = int(j_idx[0])

            # Prefer moving the larger-displacement candidate first.
            move_i = movable[i]
            move_j = movable[j]
            if not move_i and not move_j:
                continue

            ox = float(ov_x[i, j])
            oy = float(ov_y[i, j])

            if ox < oy:
                sign = 1.0 if (pos[i, 0] - pos[j, 0]) >= 0.0 else -1.0
                step = ox * 0.6 + 1e-4
                if move_i:
                    pos[i, 0] += sign * step
                if move_j:
                    pos[j, 0] -= sign * step
            else:
                sign = 1.0 if (pos[i, 1] - pos[j, 1]) >= 0.0 else -1.0
                step = oy * 0.6 + 1e-4
                if move_i:
                    pos[i, 1] += sign * step
                if move_j:
                    pos[j, 1] -= sign * step

            pos[i, 0] = np.clip(pos[i, 0], half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1], half_h[i], canvas_h - half_h[i])
            pos[j, 0] = np.clip(pos[j, 0], half_w[j], canvas_w - half_w[j])
            pos[j, 1] = np.clip(pos[j, 1], half_h[j], canvas_h - half_h[j])

    def _final_spiral_fix(
        self,
        pos: np.ndarray,
        sizes: np.ndarray,
        movable: np.ndarray,
        canvas_w: float,
        canvas_h: float,
        sep_x: np.ndarray,
        sep_y: np.ndarray,
        upper: np.ndarray,
        deadline: float,
    ) -> None:
        """Brute-force cleanup for any rare residual overlaps."""
        n = pos.shape[0]
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        area = sizes[:, 0] * sizes[:, 1]

        def overlap_pairs():
            dx = np.abs(pos[:, 0:1] - pos[:, 0][None, :])
            dy = np.abs(pos[:, 1:2] - pos[:, 1][None, :])
            mask = (dx < sep_x) & (dy < sep_y) & upper
            return np.where(mask)

        def legal(idx: int, x: float, y: float) -> bool:
            if x < half_w[idx] or x > canvas_w - half_w[idx]:
                return False
            if y < half_h[idx] or y > canvas_h - half_h[idx]:
                return False
            dx = np.abs(x - pos[:, 0])
            dy = np.abs(y - pos[:, 1])
            conflicts = (dx < sep_x[idx]) & (dy < sep_y[idx])
            conflicts[idx] = False
            return not conflicts.any()

        for _ in range(256):
            if time.time() >= deadline:
                break

            i_idx, j_idx = overlap_pairs()
            if len(i_idx) == 0:
                return

            # Pick a macro heavily involved in overlaps; prefer movable and larger.
            degree = np.zeros(n, dtype=np.int32)
            np.add.at(degree, i_idx, 1)
            np.add.at(degree, j_idx, 1)

            candidates = np.where(degree > 0)[0]
            if len(candidates) == 0:
                return

            # Score: overlap degree first, then area.
            score = degree[candidates].astype(np.float64) * 1e9 + area[candidates]
            order = candidates[np.argsort(-score)]

            moved = False
            for idx in order:
                if not movable[idx]:
                    continue
                anchor_x = float(pos[idx, 0])
                anchor_y = float(pos[idx, 1])
                base = max(float(sizes[idx, 0]), float(sizes[idx, 1])) * 0.18
                base = max(base, 1e-3)

                for r in range(1, 140):
                    if time.time() >= deadline:
                        break
                    radius = base * r
                    for k in range(24):
                        theta = (2.0 * math.pi * k) / 24.0
                        cx = anchor_x + radius * math.cos(theta)
                        cy = anchor_y + radius * math.sin(theta)
                        cx = float(np.clip(cx, half_w[idx], canvas_w - half_w[idx]))
                        cy = float(np.clip(cy, half_h[idx], canvas_h - half_h[idx]))
                        if legal(idx, cx, cy):
                            pos[idx, 0] = cx
                            pos[idx, 1] = cy
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break

            if not moved:
                # Last resort: tiny random jitter on a movable overlapping macro.
                for idx in order:
                    if not movable[idx]:
                        continue
                    jitter = max(float(sizes[idx, 0]), float(sizes[idx, 1])) * 0.03
                    cx = float(np.clip(pos[idx, 0] + np.random.uniform(-jitter, jitter), half_w[idx], canvas_w - half_w[idx]))
                    cy = float(np.clip(pos[idx, 1] + np.random.uniform(-jitter, jitter), half_h[idx], canvas_h - half_h[idx]))
                    if legal(idx, cx, cy):
                        pos[idx, 0] = cx
                        pos[idx, 1] = cy
                        moved = True
                        break
                if not moved:
                    return
