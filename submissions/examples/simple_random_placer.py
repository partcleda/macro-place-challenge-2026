"""
Simple Random Placer - Example Submission

Places all movable macros at random positions within canvas bounds.
Respects fixed macros. This is a minimal baseline — it typically
performs worse than the initial placement.

Usage:
    uv run evaluate submissions/examples/simple_random_placer.py
    uv run evaluate submissions/examples/simple_random_placer.py --all
"""

import torch

from macro_place.benchmark import Benchmark


class SimpleRandomPlacer:
    """
    Simple random placement algorithm (baseline).

    Places all movable macros at random positions within canvas bounds.
    Respects fixed macros.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Generate random placement.

        Args:
            benchmark: Benchmark object with circuit data

        Returns:
            placement: [num_macros, 2] tensor of (x, y) center positions
        """
        torch.manual_seed(self.seed)

        placement = torch.zeros(benchmark.num_macros, 2)

        for i in range(benchmark.num_macros):
            if benchmark.macro_fixed[i]:
                continue

            w, h = benchmark.macro_sizes[i]

            x_min = w / 2
            x_max = benchmark.canvas_width - w / 2
            y_min = h / 2
            y_max = benchmark.canvas_height - h / 2

            x = torch.rand(1).item() * (x_max - x_min) + x_min
            y = torch.rand(1).item() * (y_max - y_min) + y_min

            placement[i, 0] = x
            placement[i, 1] = y

        # Restore fixed macro positions
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement
