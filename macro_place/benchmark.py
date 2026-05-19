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

    Tensors contain both hard macros (indices [0, num_hard_macros)) and
    soft macros (indices [num_hard_macros, num_macros)). Hard macros are
    the primary optimization targets; soft macros are standard cell clusters
    that should be co-optimized for best results.
    """

    # Core data
    name: str

    # Canvas
    canvas_width: float
    canvas_height: float

    # Macros (hard + soft, hard macros first)
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

    # I/O ports (pins on the chip boundary)
    port_positions: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 2))  # [num_ports, 2]

    # Hard macro pin offsets (relative to macro center)
    # List of [num_pins_i, 2] tensors, one per hard macro (indices [0, num_hard_macros))
    macro_pin_offsets: List[torch.Tensor] = field(default_factory=list)

    # Pin-level net connectivity (optional; empty list if not populated)
    # Each net_pin_nodes[i] is an int64 tensor of shape [num_pins_in_net_i, 2] where:
    #   column 0 = owner index:
    #     - hard macros:  [0, num_hard_macros)
    #     - soft macros:  [num_hard_macros, num_macros)
    #     - I/O ports:    [num_macros, num_macros + num_ports)
    #   column 1 = pin index within that owner:
    #     - hard macro:   index into macro_pin_offsets[owner]
    #     - soft macro:   always 0 (pins at macro center; soft macros carry no offsets)
    #     - port:         always 0 (port is a single point at port_positions[owner-num_macros])
    # Unlike net_nodes (which dedups to per-macro granularity), this preserves
    # every pin endpoint — multiple pins on the same macro appear as multiple rows.
    # Needed by placers computing pin-level HPWL for differentiable loss.
    net_pin_nodes: List[torch.Tensor] = field(default_factory=list)

    # Routing parameters
    hroutes_per_micron: float = 11.285  # Horizontal routing tracks per micron
    vroutes_per_micron: float = 12.605  # Vertical routing tracks per micron

    # PlacementCost mapping (tensor index → PlacementCost module index)
    hard_macro_indices: List[int] = field(default_factory=list)
    soft_macro_indices: List[int] = field(default_factory=list)

    # Counts
    num_hard_macros: int = 0
    num_soft_macros: int = 0

    def __post_init__(self):
        """Validate tensor shapes and set counts."""
        # Backwards compat: if num_hard_macros not set, all macros are hard
        if self.num_hard_macros == 0 and self.num_soft_macros == 0:
            self.num_hard_macros = self.num_macros
            self.num_soft_macros = 0

        assert self.num_macros == self.num_hard_macros + self.num_soft_macros, (
            f"num_macros {self.num_macros} != "
            f"num_hard {self.num_hard_macros} + num_soft {self.num_soft_macros}"
        )
        assert self.macro_positions.shape == (self.num_macros, 2), (
            f"macro_positions shape {self.macro_positions.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_sizes.shape == (self.num_macros, 2), (
            f"macro_sizes shape {self.macro_sizes.shape} != ({self.num_macros}, 2)"
        )
        assert self.macro_fixed.shape == (self.num_macros,), (
            f"macro_fixed shape {self.macro_fixed.shape} != ({self.num_macros},)"
        )

        if len(self.net_nodes) > 0:
            assert len(self.net_nodes) == self.num_nets, (
                f"len(net_nodes) {len(self.net_nodes)} != num_nets {self.num_nets}"
            )

        if len(self.net_pin_nodes) > 0:
            assert len(self.net_pin_nodes) == self.num_nets, (
                f"len(net_pin_nodes) {len(self.net_pin_nodes)} != num_nets {self.num_nets}"
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
                "num_hard_macros": self.num_hard_macros,
                "num_soft_macros": self.num_soft_macros,
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
                "port_positions": self.port_positions,
                "macro_pin_offsets": self.macro_pin_offsets,
                "net_pin_nodes": self.net_pin_nodes,
                "hard_macro_indices": self.hard_macro_indices,
                "soft_macro_indices": self.soft_macro_indices,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "Benchmark":
        """Load benchmark from .pt file."""
        data = torch.load(path, weights_only=False)
        # Backwards compat: old .pt files lack soft macro fields
        if "num_hard_macros" not in data:
            data["num_hard_macros"] = data["num_macros"]
            data["num_soft_macros"] = 0
        if "soft_macro_indices" not in data:
            data["soft_macro_indices"] = []
        if "port_positions" not in data:
            data["port_positions"] = torch.zeros(0, 2)
        if "macro_pin_offsets" not in data:
            data["macro_pin_offsets"] = []
        if "net_pin_nodes" not in data:
            data["net_pin_nodes"] = []
        return cls(**data)

    def get_movable_mask(self) -> torch.Tensor:
        """Return mask of movable macros (not fixed)."""
        return ~self.macro_fixed

    def get_hard_macro_mask(self) -> torch.Tensor:
        """Return mask that is True for hard macros (first num_hard_macros entries)."""
        mask = torch.zeros(self.num_macros, dtype=torch.bool)
        mask[: self.num_hard_macros] = True
        return mask

    def get_soft_macro_mask(self) -> torch.Tensor:
        """Return mask that is True for soft macros."""
        mask = torch.zeros(self.num_macros, dtype=torch.bool)
        mask[self.num_hard_macros :] = True
        return mask

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Benchmark(name='{self.name}', "
            f"hard_macros={self.num_hard_macros}, "
            f"soft_macros={self.num_soft_macros}, "
            f"num_nets={self.num_nets}, "
            f"canvas={self.canvas_width:.1f}x{self.canvas_height:.1f}um)"
        )
