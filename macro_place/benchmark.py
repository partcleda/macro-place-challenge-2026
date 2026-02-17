"""
Benchmark data structure for macro placement.

Pure PyTorch tensor representation of placement benchmarks.
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Benchmark:
    """
    Placement benchmark in pure PyTorch tensors.

    All coordinates are in microns.
    All indices are 0-based.
    """

    # Core data
    name: str

    # Canvas
    canvas_width: float
    canvas_height: float

    # Macros (hard macros only)
    num_macros: int
    macro_positions: torch.Tensor  # [num_macros, 2] - (x, y) centers
    macro_sizes: torch.Tensor  # [num_macros, 2] - (width, height)
    macro_fixed: torch.Tensor  # [num_macros] - bool, True if fixed
    macro_names: List[str]  # [num_macros] - names for debugging

    # Nets (hypergraph connectivity)
    num_nets: int
    net_nodes: List[torch.Tensor]  # List of [nodes_in_net_i] - node indices
    net_weights: torch.Tensor  # [num_nets] - net weights (default 1.0)

    # Grid (for metrics)
    grid_rows: int
    grid_cols: int

    # Routing parameters
    hroutes_per_micron: float = 11.285  # Horizontal routing tracks per micron
    vroutes_per_micron: float = 12.605  # Vertical routing tracks per micron

    # PlacementCost mapping (for objective computation)
    hard_macro_indices: List[int] = field(default_factory=list)  # Map tensor index → PlacementCost module index

    def __post_init__(self):
        """Validate tensor shapes."""
        assert self.macro_positions.shape == (self.num_macros, 2), (
            f"macro_positions shape {self.macro_positions.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_sizes.shape == (self.num_macros, 2), (
            f"macro_sizes shape {self.macro_sizes.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_fixed.shape == (self.num_macros,), (
            f"macro_fixed shape {self.macro_fixed.shape} != ({self.num_macros},)"
        )

        # Note: net_nodes may be empty even if num_nets > 0 (nets stored in PlacementCost only)
        if len(self.net_nodes) > 0:
            assert len(self.net_nodes) == self.num_nets, (
                f"len(net_nodes) {len(self.net_nodes)} != num_nets {self.num_nets}"
            )

        assert self.net_weights.shape == (self.num_nets,), (
            f"net_weights shape {self.net_weights.shape} != ({self.num_nets},)"
        )

    def save(self, path: str):
        """Save benchmark to .pt file."""
        torch.save(
            {
                "name": self.name,
                "canvas_width": self.canvas_width,
                "canvas_height": self.canvas_height,
                "num_macros": self.num_macros,
                "macro_positions": self.macro_positions,
                "macro_sizes": self.macro_sizes,
                "macro_fixed": self.macro_fixed,
                "macro_names": self.macro_names,
                "num_nets": self.num_nets,
                "net_nodes": self.net_nodes,
                "net_weights": self.net_weights,
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "hroutes_per_micron": self.hroutes_per_micron,
                "vroutes_per_micron": self.vroutes_per_micron,
                "hard_macro_indices": self.hard_macro_indices,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "Benchmark":
        """Load benchmark from .pt file."""
        data = torch.load(path, weights_only=False)
        return cls(**data)

    def get_movable_mask(self) -> torch.Tensor:
        """Return mask of movable macros (not fixed)."""
        return ~self.macro_fixed

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Benchmark(name='{self.name}', "
            f"num_macros={self.num_macros}, "
            f"num_nets={self.num_nets}, "
            f"canvas={self.canvas_width:.1f}x{self.canvas_height:.1f}um)"
        )
