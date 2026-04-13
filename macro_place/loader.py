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
    netlist_file: str, plc_file: Optional[str] = None, name: Optional[str] = None
) -> Tuple[Benchmark, PlacementCost]:
    """
    Load benchmark from ICCAD04 format using PlacementCost parser.

    Args:
        netlist_file: Path to netlist.pb.txt
        plc_file: Optional path to initial.plc (if None, uses default placement)
        name: Optional benchmark name override (inferred from path if not given)

    Returns:
        Tuple of (Benchmark, PlacementCost) - Benchmark contains PyTorch tensors,
        PlacementCost object is needed for cost computation
    """
    # Initialize PlacementCost (parses netlist)
    plc = PlacementCost(netlist_file)

    # Optionally restore placement from .plc file
    if plc_file:
        plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

    # Extract benchmark name from path if not provided.
    # IBM paths: .../ibm01/netlist.pb.txt  -> "ibm01"
    # NG45 paths: .../ariane133/netlist/output_CT_Grouping/netlist.pb.txt -> "ariane133"
    if name is None:
        name = os.path.basename(os.path.dirname(netlist_file))
        # NG45 designs have extra subdirectory levels; walk up to find the design name
        if name in ("output_CT_Grouping", "output_CodeElement"):
            name = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(netlist_file)))
            )

    # Extract canvas and grid info
    canvas_width, canvas_height = plc.get_canvas_width_height()
    grid_rows = plc.grid_row
    grid_cols = plc.grid_col
    hroutes_per_micron = plc.hroutes_per_micron
    vroutes_per_micron = plc.vroutes_per_micron

    # Extract hard macros
    hard_macro_plc_indices = plc.hard_macro_indices
    num_hard = len(hard_macro_plc_indices)

    macro_positions = []
    macro_sizes = []
    macro_fixed = []
    macro_names = []

    for idx in hard_macro_plc_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        w = node.get_width()
        h = node.get_height()
        fixed = node.get_fix_flag()
        macro_positions.append([x, y])
        macro_sizes.append([w, h])
        macro_fixed.append(fixed)
        macro_names.append(node.get_name())

    # Extract soft macros (standard cell clusters)
    soft_macro_plc_indices = plc.soft_macro_indices
    num_soft = len(soft_macro_plc_indices)

    for idx in soft_macro_plc_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        w = node.get_width()
        h = node.get_height()
        fixed = node.get_fix_flag()
        macro_positions.append([x, y])
        macro_sizes.append([w, h])
        macro_fixed.append(fixed)
        macro_names.append(node.get_name())

    num_macros = num_hard + num_soft

    # Extract hard macro pin offsets (relative to macro center)
    macro_pin_offsets = []
    pin_map = {}
    for idx in plc.hard_macro_pin_indices:
        pin = plc.modules_w_pins[idx]
        pin_macro = pin.get_macro_name() if hasattr(pin, "get_macro_name") else None
        if pin_macro:
            pin_map.setdefault(pin_macro, []).append(
                [pin.x_offset, pin.y_offset]
            )
    for macro_idx in hard_macro_plc_indices:
        macro_name = plc.modules_w_pins[macro_idx].get_name()
        offsets = pin_map.get(macro_name, [])
        macro_pin_offsets.append(
            torch.tensor(offsets, dtype=torch.float32) if offsets else torch.zeros(0, 2)
        )

    # Extract I/O port positions
    port_pos_list = []
    for idx in plc.port_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        port_pos_list.append([x, y])
    port_positions = (
        torch.tensor(port_pos_list, dtype=torch.float32)
        if port_pos_list
        else torch.zeros(0, 2)
    )

    # Convert to tensors
    macro_positions = torch.tensor(macro_positions, dtype=torch.float32)
    macro_sizes = torch.tensor(macro_sizes, dtype=torch.float32)
    macro_fixed = torch.tensor(macro_fixed, dtype=torch.bool)

    # Extract net connectivity
    # Build mapping from module/port names to benchmark tensor indices:
    #   hard macros -> [0, num_hard), soft macros -> [num_hard, num_hard+num_soft)
    #   ports -> num_macros + port_index
    plc_idx_to_bench = {}
    for bench_idx, plc_idx in enumerate(hard_macro_plc_indices):
        plc_idx_to_bench[plc_idx] = bench_idx
    for bench_idx_offset, plc_idx in enumerate(soft_macro_plc_indices):
        plc_idx_to_bench[plc_idx] = num_hard + bench_idx_offset
    for port_offset, plc_idx in enumerate(plc.port_indices):
        plc_idx_to_bench[plc_idx] = num_macros + port_offset

    # Map pin/module names to benchmark indices via their parent macro/port
    name_to_bench = {}
    for plc_idx, bench_idx in plc_idx_to_bench.items():
        mod = plc.modules_w_pins[plc_idx]
        name_to_bench[mod.get_name()] = bench_idx

    num_nets = int(plc.net_cnt)
    net_nodes = []
    net_weights_list = []
    for driver, sinks in plc.nets.items():
        nodes_in_net = set()
        for pin_name in [driver] + sinks:
            # Pin names are "MACRO/PIN" for macro pins or just "PORT" for ports
            parent = pin_name.split("/")[0]
            if parent in name_to_bench:
                nodes_in_net.add(name_to_bench[parent])
        if nodes_in_net:
            net_nodes.append(torch.tensor(sorted(nodes_in_net), dtype=torch.long))
            net_weights_list.append(1.0)

    num_nets = len(net_nodes)
    net_weights_tensor = torch.tensor(net_weights_list, dtype=torch.float32) if net_weights_list else torch.zeros(0, dtype=torch.float32)

    # Create Benchmark object
    benchmark = Benchmark(
        name=name,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        num_macros=num_macros,
        num_hard_macros=num_hard,
        num_soft_macros=num_soft,
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
        port_positions=port_positions,
        macro_pin_offsets=macro_pin_offsets,
        hard_macro_indices=hard_macro_plc_indices,
        soft_macro_indices=soft_macro_plc_indices,
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
