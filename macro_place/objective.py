"""
Proxy cost computation using PlacementCost's ground truth evaluator.

Wraps PlacementCost methods to compute wirelength, density, and congestion costs.
Also computes overlap metrics for validation and analysis.
"""

import torch
import math
from typing import Dict, Optional

from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark


# Monkey-patch PlacementCost to fix boundary bug in __get_grid_cell_location
_original_get_grid_cell_location = PlacementCost._PlacementCost__get_grid_cell_location


def _patched_get_grid_cell_location(self, x_pos, y_pos):
    """Fixed version with bounds clamping."""
    self.grid_width = float(self.width / self.grid_col)
    self.grid_height = float(self.height / self.grid_row)
    row = math.floor(y_pos / self.grid_height)
    col = math.floor(x_pos / self.grid_width)

    # Clamp to valid range to fix boundary bug
    row = max(0, min(row, self.grid_row - 1))
    col = max(0, min(col, self.grid_col - 1))

    return row, col


PlacementCost._PlacementCost__get_grid_cell_location = _patched_get_grid_cell_location


def compute_overlap_metrics(
    placement: torch.Tensor, benchmark: Benchmark
) -> Dict[str, float]:
    """
    Compute overlap metrics for macro placement.

    Borrowed from intern_challenge placement.py and adapted for macro placement.

    Args:
        placement: [num_macros, 2] tensor of (x, y) center positions
        benchmark: Benchmark object with macro sizes

    Returns:
        Dictionary with:
            - overlap_count: Number of overlapping macro pairs
            - total_overlap_area: Total area of all overlaps (μm²)
            - max_overlap_area: Largest single overlap area (μm²)
            - num_macros_with_overlaps: Number of macros involved in at least one overlap
            - overlap_ratio: Fraction of macros with overlaps (0.0 = no overlaps, 1.0 = all overlap)
    """
    num_macros = placement.shape[0]

    if num_macros <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "num_macros_with_overlaps": 0,
            "overlap_ratio": 0.0,
        }

    # Extract positions and sizes
    positions = placement.cpu().detach().numpy()  # [N, 2]
    widths = benchmark.macro_sizes[:, 0].cpu().numpy()  # [N]
    heights = benchmark.macro_sizes[:, 1].cpu().numpy()  # [N]

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    macros_with_overlaps = set()

    # Check all pairs for overlap
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap (sum of half-widths/heights)
            min_sep_x = (widths[i] + widths[j]) / 2.0
            min_sep_y = (heights[i] + heights[j]) / 2.0

            # Calculate overlap amounts in each dimension
            overlap_x = max(0.0, min_sep_x - dx)
            overlap_y = max(0.0, min_sep_y - dy)

            # Overlap occurs only if BOTH x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
                macros_with_overlaps.add(i)
                macros_with_overlaps.add(j)

    num_macros_with_overlaps = len(macros_with_overlaps)
    overlap_ratio = num_macros_with_overlaps / num_macros if num_macros > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "num_macros_with_overlaps": num_macros_with_overlaps,
        "overlap_ratio": overlap_ratio,
    }


def compute_proxy_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc: PlacementCost,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute proxy cost using PlacementCost's ground truth evaluator.

    Args:
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object with circuit data
        plc: PlacementCost object (contains all netlist/placement data)
        weights: Optional cost weights {
            'wirelength': 1.0,
            'density': 0.5,
            'congestion': 0.5
        }

    Returns:
        {
            'proxy_cost': float,
            'wirelength_cost': float,
            'density_cost': float,
            'congestion_cost': float,
            'overlap_count': int,
            'total_overlap_area': float,
            'max_overlap_area': float,
            'num_macros_with_overlaps': int,
            'overlap_ratio': float,
        }
    """
    if weights is None:
        weights = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}

    # Set placement in PlacementCost object (if different from current)
    _set_placement(plc, placement, benchmark)

    # Compute costs using PlacementCost methods
    wirelength_cost = plc.get_cost()
    density_cost = plc.get_density_cost()
    congestion_cost = plc.get_congestion_cost()  # Fixed with monkey-patch above

    # Weighted sum (matching ISPD 2023 paper convention)
    proxy = (
        weights["wirelength"] * wirelength_cost
        + weights["density"] * density_cost
        + weights["congestion"] * congestion_cost
    )

    # Compute overlap metrics
    overlap_metrics = compute_overlap_metrics(placement, benchmark)

    return {
        "proxy_cost": proxy,
        "wirelength_cost": wirelength_cost,
        "density_cost": density_cost,
        "congestion_cost": congestion_cost,
        **overlap_metrics,  # Add all overlap metrics
    }


def _set_placement(plc: PlacementCost, placement: torch.Tensor, benchmark: Benchmark):
    """
    Set macro positions in PlacementCost object.

    Args:
        plc: PlacementCost object
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object with macro indices mapping
    """
    # Convert tensor to numpy for PlacementCost API
    placement_np = placement.cpu().numpy()

    # Set each macro's position
    for i, macro_idx in enumerate(benchmark.hard_macro_indices):
        x, y = placement_np[i]
        node = plc.modules_w_pins[macro_idx]
        node.set_pos(x, y)

    # Reinitialize congestion arrays with correct size
    # This is needed because the arrays may be incorrectly sized
    _ensure_congestion_arrays(plc)

    # Mark that costs need to be recomputed
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True


def _ensure_congestion_arrays(plc: PlacementCost):
    """
    Ensure congestion arrays are properly sized for current grid.

    Args:
        plc: PlacementCost object
    """
    expected_size = plc.grid_col * plc.grid_row
    current_size = len(plc.H_routing_cong)

    if current_size != expected_size:
        # Reinitialize with correct size
        plc.V_routing_cong = [0] * expected_size
        plc.H_routing_cong = [0] * expected_size
        plc.V_macro_routing_cong = [0] * expected_size
        plc.H_macro_routing_cong = [0] * expected_size
