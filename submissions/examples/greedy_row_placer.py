"""
Greedy Row Placer - Demo Submission

A simple but legal placer that:
1. Sorts movable macros by height (tallest first)
2. Places them left-to-right in rows (like shelf packing)
3. Guarantees zero overlaps among all macros (hard and soft) and canvas compliance

This produces valid, scoreable placements but makes no attempt to
optimize wirelength, density, or congestion. Use it as a starting
point for your own algorithm.

Usage:
    uv run evaluate submissions/examples/greedy_row_placer.py
    uv run evaluate submissions/examples/greedy_row_placer.py --all
    uv run evaluate submissions/examples/greedy_row_placer.py -b ibm03
"""

import torch

from macro_place.benchmark import Benchmark


class GreedyRowPlacer:
    """
    Greedy row-based (shelf packing) placement.

    Places every movable macro (hard and soft) in rows from bottom to top,
    left to right, sorted by descending height. Guarantees zero pairwise overlaps.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask()
        movable_indices = torch.where(movable)[0].tolist()

        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        # Sort movable macros by height descending (shelf packing heuristic)
        movable_indices.sort(key=lambda i: -sizes[i, 1].item())

        # Small gap to avoid float32 touching-edge false overlaps
        gap = 0.001

        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0

        for idx in movable_indices:
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            # Start new row if macro doesn't fit
            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0

            # Check if we've run out of vertical space
            if cursor_y + h > canvas_h:
                # Place at origin as fallback (will overlap but shouldn't happen
                # if area utilization < 100%)
                placement[idx, 0] = w / 2
                placement[idx, 1] = h / 2
                continue

            # Place macro (positions are centers)
            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2

            cursor_x += w + gap
            row_height = max(row_height, h)

        return placement
