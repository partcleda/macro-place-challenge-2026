"""
Will's Seed Attempt — Force-Directed + Simulated Annealing Hybrid

Strategy:
1. Build macro connectivity graph from nets
2. Force-directed placement: net springs + overlap repulsion (vectorized numpy)
3. Legalize: greedy snap to non-overlapping positions
4. SA refinement: connectivity-guided moves with incremental cost updates

Usage:
    uv run evaluate submissions/will_seed/placer.py
    uv run evaluate submissions/will_seed/placer.py --all
    uv run evaluate submissions/will_seed/placer.py --ng45
"""

import math
import random
import torch
import numpy as np
from typing import List, Tuple

from macro_place.benchmark import Benchmark


class WillSeedPlacer:
    """Force-directed placement with SA refinement."""

    def __init__(self, seed: int = 42, fd_iters: int = 150, sa_iters: int = 3000):
        self.seed = seed
        self.fd_iters = fd_iters
        self.sa_iters = sa_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)

        n = benchmark.num_macros
        movable = benchmark.get_movable_mask().numpy()
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        # Build connectivity: sparse edge list
        edges, edge_weights = self._build_edges(benchmark)

        # Phase 1: Force-directed placement (vectorized)
        pos = self._force_directed(benchmark, edges, edge_weights, movable, sizes, cw, ch)

        # Phase 2: Legalize (remove overlaps)
        pos = self._legalize(pos, movable, sizes, cw, ch)

        # Phase 3: SA refinement with incremental cost
        pos = self._sa_refine(pos, benchmark, edges, edge_weights, movable, sizes, cw, ch)

        return torch.tensor(pos, dtype=torch.float32)

    def _build_edges(self, benchmark: Benchmark) -> Tuple[np.ndarray, np.ndarray]:
        """Build sparse edge list from net connectivity."""
        n = benchmark.num_macros
        edge_dict = {}

        if benchmark.net_nodes and len(benchmark.net_nodes) > 0:
            weights = benchmark.net_weights.numpy() if benchmark.net_weights is not None else None
            for net_idx, net in enumerate(benchmark.net_nodes):
                if net is None or len(net) == 0:
                    continue
                nodes = net.numpy() if isinstance(net, torch.Tensor) else np.array(net)
                macro_nodes = nodes[nodes < n]
                if len(macro_nodes) < 2:
                    continue
                w = 1.0 / (len(macro_nodes) - 1)
                if weights is not None and net_idx < len(weights):
                    w *= weights[net_idx]
                for ki in range(len(macro_nodes)):
                    for kj in range(ki + 1, len(macro_nodes)):
                        pair = (int(macro_nodes[ki]), int(macro_nodes[kj]))
                        if pair[0] > pair[1]:
                            pair = (pair[1], pair[0])
                        edge_dict[pair] = edge_dict.get(pair, 0) + w

        if not edge_dict:
            # No net data — create weak uniform edges
            edges_list = []
            weights_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges_list.append((i, j))
                    weights_list.append(0.01)
            return np.array(edges_list, dtype=np.int32), np.array(weights_list, dtype=np.float64)

        edges_list = list(edge_dict.keys())
        weights_list = [edge_dict[e] for e in edges_list]
        return np.array(edges_list, dtype=np.int32), np.array(weights_list, dtype=np.float64)

    def _force_directed(self, benchmark, edges, edge_weights, movable, sizes, cw, ch):
        """Vectorized force-directed placement."""
        n = benchmark.num_macros
        pos = benchmark.macro_positions.numpy().copy().astype(np.float64)

        # Scatter movable macros around center
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        for i in range(n):
            if not movable[i]:
                continue
            pos[i, 0] = np.clip(
                cw / 2 + np.random.uniform(-cw * 0.3, cw * 0.3),
                half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(
                ch / 2 + np.random.uniform(-ch * 0.3, ch * 0.3),
                half_h[i], ch - half_h[i])

        movable_mask = movable.astype(np.float64)  # [N]
        ei = edges[:, 0]  # source indices
        ej = edges[:, 1]  # target indices

        for iteration in range(self.fd_iters):
            t = 1.0 - iteration / self.fd_iters
            lr = max(0.3, 3.0 * t)

            forces = np.zeros((n, 2), dtype=np.float64)

            # --- Attraction (vectorized over edges) ---
            dx = pos[ej, 0] - pos[ei, 0]
            dy = pos[ej, 1] - pos[ei, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            dist = np.maximum(dist, 0.1)
            f = edge_weights * 0.02  # spring constant
            fx = f * dx / dist
            fy = f * dy / dist
            np.add.at(forces[:, 0], ei, fx * movable_mask[ei])
            np.add.at(forces[:, 1], ei, fy * movable_mask[ei])
            np.add.at(forces[:, 0], ej, -fx * movable_mask[ej])
            np.add.at(forces[:, 1], ej, -fy * movable_mask[ej])

            # --- Repulsion (vectorized, all pairs) ---
            # Compute pairwise distances
            dx_all = pos[:, 0:1] - pos[:, 0:1].T  # [N, N]
            dy_all = pos[:, 1:2] - pos[:, 1:2].T
            sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
            sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
            overlap_x = np.maximum(0, sep_x - np.abs(dx_all))
            overlap_y = np.maximum(0, sep_y - np.abs(dy_all))
            overlapping = (overlap_x > 0) & (overlap_y > 0)
            np.fill_diagonal(overlapping, False)

            # Overlap repulsion
            dist_all = np.sqrt(dx_all ** 2 + dy_all ** 2)
            dist_all = np.maximum(dist_all, 0.01)
            overlap_area = overlap_x * overlap_y
            max_area = (sizes[:, 0] * sizes[:, 1]).max()
            repel_mag = 30.0 * overlap_area / max_area * overlapping
            repel_fx = repel_mag * dx_all / dist_all
            repel_fy = repel_mag * dy_all / dist_all
            forces[:, 0] += (repel_fx.sum(axis=1)) * movable_mask
            forces[:, 1] += (repel_fy.sum(axis=1)) * movable_mask

            # Mild density spreading for non-overlapping nearby macros
            nearby = (~overlapping) & (dist_all < max(cw, ch) * 0.25)
            np.fill_diagonal(nearby, False)
            spread_mag = 0.3 * t / np.maximum(dist_all, 0.1) * nearby
            spread_fx = spread_mag * dx_all / dist_all
            spread_fy = spread_mag * dy_all / dist_all
            forces[:, 0] += spread_fx.sum(axis=1) * movable_mask
            forces[:, 1] += spread_fy.sum(axis=1) * movable_mask

            # Center gravity
            forces[:, 0] += (cw / 2 - pos[:, 0]) * 0.002 * movable_mask
            forces[:, 1] += (ch / 2 - pos[:, 1]) * 0.002 * movable_mask

            # Clamp and apply
            f_mag = np.sqrt(forces[:, 0] ** 2 + forces[:, 1] ** 2)
            max_step = lr * np.maximum(sizes[:, 0], sizes[:, 1])
            scale = np.where(f_mag > max_step, max_step / np.maximum(f_mag, 1e-10), 1.0)
            forces[:, 0] *= scale
            forces[:, 1] *= scale

            pos[:, 0] += forces[:, 0]
            pos[:, 1] += forces[:, 1]
            pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

        return pos

    def _legalize(self, pos, movable, sizes, cw, ch):
        """Remove overlaps with greedy displacement — largest first."""
        n = len(pos)
        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal_pos = pos.copy()
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue

            # Check current position
            if not self._has_overlap_vec(idx, legal_pos[idx], legal_pos, sizes, placed):
                placed[idx] = True
                continue

            # Spiral search for nearby legal position
            w, h = sizes[idx]
            step = max(w, h) * 0.4
            best_pos = legal_pos[idx].copy()
            best_dist = float('inf')

            for radius in range(1, 80):
                found = False
                offsets = []
                r = radius
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) == r or abs(dy) == r:  # Only ring perimeter
                            offsets.append((dx * step, dy * step))

                for dx_off, dy_off in offsets:
                    cx = np.clip(pos[idx, 0] + dx_off, half_w[idx], cw - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dy_off, half_h[idx], ch - half_h[idx])
                    candidate = np.array([cx, cy])
                    if not self._has_overlap_vec(idx, candidate, legal_pos, sizes, placed):
                        dist = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = candidate.copy()
                            found = True
                if found:
                    break

            legal_pos[idx] = best_pos
            placed[idx] = True

        return legal_pos

    def _has_overlap_vec(self, idx, candidate, all_pos, sizes, placed):
        """Vectorized overlap check with small gap for float safety."""
        gap = 0.01
        dx = np.abs(candidate[0] - all_pos[:, 0])
        dy = np.abs(candidate[1] - all_pos[:, 1])
        sep_x = (sizes[idx, 0] + sizes[:, 0]) / 2 + gap
        sep_y = (sizes[idx, 1] + sizes[:, 1]) / 2 + gap
        overlaps = (dx < sep_x) & (dy < sep_y) & placed
        overlaps[idx] = False
        return overlaps.any()

    def _sa_refine(self, pos, benchmark, edges, edge_weights, movable, sizes, cw, ch):
        """SA refinement with incremental cost."""
        n = benchmark.num_macros
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        pos = pos.copy()
        legalized_pos = pos.copy()  # Save overlap-free starting point
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2

        # Build per-macro edge lists for fast incremental cost
        macro_edges: List[List[Tuple[int, int, float]]] = [[] for _ in range(n)]
        for k, (i, j) in enumerate(edges):
            macro_edges[i].append((j, k, edge_weights[k]))
            macro_edges[j].append((i, k, edge_weights[k]))

        def wirelength_cost():
            dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
            dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
            return (edge_weights * (dx + dy)).sum()

        # Precompute separation matrices for overlap cost
        sep_x_mat = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y_mat = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        def overlap_cost():
            dx = np.abs(pos[:, 0:1] - pos[:, 0:1].T)
            dy = np.abs(pos[:, 1:2] - pos[:, 1:2].T)
            ox = np.maximum(0, sep_x_mat - dx)
            oy = np.maximum(0, sep_y_mat - dy)
            ov = ox * oy
            np.fill_diagonal(ov, 0)
            return ov.sum() * 250.0  # Each pair counted twice

        current_wl = wirelength_cost()
        current_ov = overlap_cost()
        current_cost = current_wl + current_ov
        best_pos = legalized_pos.copy()
        best_cost = float('inf')  # Only accept overlap-free solutions

        T_start = max(cw, ch) * 0.3
        T_end = max(cw, ch) * 0.0005

        for step in range(self.sa_iters):
            frac = step / self.sa_iters
            T = T_start * (T_end / T_start) ** frac

            move_type = random.random()
            old_pos = pos.copy()

            if move_type < 0.5:
                # SHIFT
                i = random.choice(movable_idx)
                dx = random.gauss(0, T)
                dy = random.gauss(0, T)
                pos[i, 0] = np.clip(pos[i, 0] + dx, half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + dy, half_h[i], ch - half_h[i])
            elif move_type < 0.8:
                # SWAP
                i = random.choice(movable_idx)
                j = random.choice(movable_idx)
                if i != j:
                    pi = np.clip(old_pos[j].copy(), [half_w[i], half_h[i]], [cw - half_w[i], ch - half_h[i]])
                    pj = np.clip(old_pos[i].copy(), [half_w[j], half_h[j]], [cw - half_w[j], ch - half_h[j]])
                    pos[i] = pi
                    pos[j] = pj
            else:
                # MOVE TOWARD NEIGHBOR
                i = random.choice(movable_idx)
                if macro_edges[i]:
                    j, _, _ = random.choice(macro_edges[i])
                    alpha = random.uniform(0.05, 0.4)
                    new_x = pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0])
                    new_y = pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1])
                    pos[i, 0] = np.clip(new_x, half_w[i], cw - half_w[i])
                    pos[i, 1] = np.clip(new_y, half_h[i], ch - half_h[i])

            new_wl = wirelength_cost()
            new_ov = overlap_cost()
            new_cost = new_wl + new_ov
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                current_wl = new_wl
                current_ov = new_ov
                if current_cost < best_cost and new_ov == 0:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos = old_pos

        # Final safety: if no overlap-free solution found, return legalized position
        if best_cost == float('inf'):
            return legalized_pos

        return best_pos
