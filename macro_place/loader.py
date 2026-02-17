"""
Benchmark loader - extracts data from PlacementCost into PyTorch tensors.

Leverages the existing MacroPlacement parser instead of reimplementing.
"""

import os
import torch
from typing import Optional, Tuple

from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark


def load_benchmark(
    netlist_file: str, plc_file: Optional[str] = None
) -> Tuple[Benchmark, PlacementCost]:
    """
    Load benchmark from ICCAD04 format using PlacementCost parser.

    Args:
        netlist_file: Path to netlist.pb.txt
        plc_file: Optional path to initial.plc (if None, uses default placement)

    Returns:
        Tuple of (Benchmark, PlacementCost) - Benchmark contains PyTorch tensors,
        PlacementCost object is needed for cost computation
    """
    # Initialize PlacementCost (parses netlist)
    plc = PlacementCost(netlist_file)

    # Optionally restore placement from .plc file
    if plc_file:
        plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

    # Extract benchmark name
    name = os.path.basename(os.path.dirname(netlist_file))

    # Extract canvas and grid info
    canvas_width, canvas_height = plc.get_canvas_width_height()
    grid_rows = plc.grid_row
    grid_cols = plc.grid_col
    hroutes_per_micron = plc.hroutes_per_micron
    vroutes_per_micron = plc.vroutes_per_micron

    # Extract hard macros only (movable + fixed blocks)
    hard_macro_indices = plc.hard_macro_indices
    num_macros = len(hard_macro_indices)

    macro_positions = []
    macro_sizes = []
    macro_fixed = []
    macro_names = []

    for idx in hard_macro_indices:
        node = plc.modules_w_pins[idx]

        x, y = node.get_pos()
        w = node.get_width()
        h = node.get_height()
        fixed = node.get_fix_flag()
        node_name = node.get_name()

        macro_positions.append([x, y])
        macro_sizes.append([w, h])
        macro_fixed.append(fixed)
        macro_names.append(node_name)

    # Convert to tensors
    macro_positions = torch.tensor(macro_positions, dtype=torch.float32)
    macro_sizes = torch.tensor(macro_sizes, dtype=torch.float32)
    macro_fixed = torch.tensor(macro_fixed, dtype=torch.bool)

    # Extract net connectivity
    # NOTE: PlacementCost has all the nets internally (net_cnt), but extracting them
    # into tensor format is complex due to pins/sinks structure. Since cost computation
    # is handled by PlacementCost directly, we don't strictly need them in tensor form.
    # For now, just capture the count for display purposes.
    num_nets = int(plc.net_cnt) if hasattr(plc, "net_cnt") else 0

    # TODO: If needed for participant algorithms, extract nets from PlacementCost:
    # - Iterate through hard_macro_pin_indices and soft_macro_pin_indices
    # - Build driver-sink relationships from pin.get_sink()
    # - Convert to List[Tensor] format for net_nodes
    net_nodes = []
    net_weights = []

    net_weights_tensor = (
        torch.tensor(net_weights, dtype=torch.float32)
        if net_weights
        else torch.zeros(num_nets, dtype=torch.float32)
    )

    # Create Benchmark object
    benchmark = Benchmark(
        name=name,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        num_macros=num_macros,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_fixed=macro_fixed,
        macro_names=macro_names,
        num_nets=num_nets,
        net_nodes=net_nodes,
        net_weights=net_weights_tensor,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        hroutes_per_micron=hroutes_per_micron,
        vroutes_per_micron=vroutes_per_micron,
        hard_macro_indices=hard_macro_indices,
    )

    return benchmark, plc


def load_benchmark_from_dir(benchmark_dir: str) -> Tuple[Benchmark, PlacementCost]:
    """
    Convenience wrapper to load from directory.

    Args:
        benchmark_dir: Path like "external/MacroPlacement/Testcases/ICCAD04/ibm01"

    Returns:
        Tuple of (Benchmark, PlacementCost)
    """
    netlist_file = os.path.join(benchmark_dir, "netlist.pb.txt")
    plc_file = os.path.join(benchmark_dir, "initial.plc")

    if not os.path.exists(netlist_file):
        raise FileNotFoundError(f"Netlist not found: {netlist_file}")

    if not os.path.exists(plc_file):
        print(f"Warning: No initial.plc found at {plc_file}, using default placement")
        plc_file = None

    return load_benchmark(netlist_file, plc_file)
