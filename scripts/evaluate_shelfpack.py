#!/usr/bin/env python3
"""
Generate shelf-pack placements for NG45/ASAP7 benchmarks and evaluate with ORFS.

Usage:
    # Generate placement + proxy cost only (fast)
    python scripts/evaluate_shelfpack.py --benchmark ariane133_ng45

    # Generate placement + run full ORFS flow
    python scripts/evaluate_shelfpack.py --benchmark ariane133_ng45 --run-orfs

    # All NG45 benchmarks
    python scripts/evaluate_shelfpack.py --all --run-orfs
"""

import os
import sys
import shutil
import argparse
import subprocess
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import Benchmark
from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
from utils import validate_placement


class ShelfPackPlacer:
    """FFDH shelf-packing placer — zero overlaps guaranteed."""

    def __init__(self, halo_x: float = 0.0, halo_y: float = 0.0):
        """
        Args:
            halo_x: Horizontal clearance around each macro (μm). For ORFS NG45: 22.4
            halo_y: Vertical clearance around each macro (μm). For ORFS NG45: 15.12
        """
        self.halo_x = halo_x
        self.halo_y = halo_y

    def place(self, benchmark: Benchmark, canvas_override=None) -> torch.Tensor:
        """
        Args:
            benchmark: Benchmark object
            canvas_override: Optional (width, height) to override benchmark canvas dims.
                             Use ORFS CORE_AREA dimensions when targeting OpenROAD.
        """
        num_macros = benchmark.num_macros
        sizes = benchmark.macro_sizes
        canvas_w = canvas_override[0] if canvas_override else benchmark.canvas_width
        canvas_h = canvas_override[1] if canvas_override else benchmark.canvas_height
        EPS = 1e-3

        movable = [i for i in range(num_macros) if not benchmark.macro_fixed[i]]
        movable.sort(key=lambda i: sizes[i, 1].item(), reverse=True)

        placement = torch.zeros(num_macros, 2)
        shelves = []  # [y_bottom, shelf_height, current_x]

        for idx in movable:
            # Effective size includes halo on both sides + epsilon for float32
            w_eff = sizes[idx, 0].item() + 2 * self.halo_x + EPS
            h_eff = sizes[idx, 1].item() + 2 * self.halo_y + EPS
            w_real = sizes[idx, 0].item()
            h_real = sizes[idx, 1].item()

            placed = False
            for shelf in shelves:
                y_bot, sh, cx = shelf
                if h_eff <= sh + 1e-9 and cx + w_eff <= canvas_w + 1e-9:
                    # Place center of actual macro (inside the halo)
                    placement[idx, 0] = cx + self.halo_x + w_real / 2
                    placement[idx, 1] = y_bot + self.halo_y + h_real / 2
                    shelf[2] = cx + w_eff
                    placed = True
                    break

            if not placed:
                y_bot = shelves[-1][0] + shelves[-1][1] if shelves else 0.0
                placement[idx, 0] = self.halo_x + w_real / 2
                placement[idx, 1] = y_bot + self.halo_y + h_real / 2
                shelves.append([y_bot, h_eff, w_eff])

        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
        return placement


# Map benchmark names to source directories for loading plc
SOURCE_DIRS = {
    'ariane133_ng45': 'external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping',
    'ariane136_ng45': 'external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping',
    'nvdla_ng45': 'external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping',
    'mempool_tile_ng45': 'external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping',
}


def evaluate_one(benchmark_name: str, run_orfs: bool = False):
    print(f"\n{'='*60}")
    print(f"  {benchmark_name}")
    print(f"{'='*60}")

    # Load benchmark
    pt_file = Path(f"benchmarks/processed/public/{benchmark_name}.pt")
    if not pt_file.exists():
        print(f"  ERROR: {pt_file} not found")
        return

    benchmark = Benchmark.load(str(pt_file))
    print(f"  Loaded: {benchmark.num_macros} macros, canvas {benchmark.canvas_width:.1f}x{benchmark.canvas_height:.1f}")

    # Load PlacementCost for proxy cost computation
    source_dir = SOURCE_DIRS.get(benchmark_name)
    if not source_dir or not Path(source_dir).exists():
        print(f"  WARNING: Source dir not found for {benchmark_name}, skipping proxy cost")
        plc = None
    else:
        _, plc = load_benchmark_from_dir(source_dir)

    # Generate placement
    # ORFS CORE_AREA dimensions (larger than benchmark canvas) and macro halos
    # Halos must be >= ORFS MACRO_PLACE_HALO (22.4x15.12 for NG45) plus extra
    # margin for PDN power stripe channels between macro rows.
    ORFS_CONFIGS = {
        'ariane133_ng45': {'core_w': 2052.0, 'core_h': 2100.0, 'halo_x': 30.0, 'halo_y': 30.0},
        'ariane136_ng45': {'core_w': 2052.0, 'core_h': 2100.0, 'halo_x': 15.0, 'halo_y': 15.0},
        'nvdla_ng45':     {'core_w': 2052.0, 'core_h': 2100.0, 'halo_x': 30.0, 'halo_y': 30.0},
        'mempool_tile_ng45': {'core_w': 1990.0, 'core_h': 1990.0, 'halo_x': 30.0, 'halo_y': 30.0},
    }

    canvas_override = None
    if run_orfs and benchmark_name in ORFS_CONFIGS:
        cfg = ORFS_CONFIGS[benchmark_name]
        placer = ShelfPackPlacer(halo_x=cfg['halo_x'], halo_y=cfg['halo_y'])
        canvas_override = (cfg['core_w'], cfg['core_h'])
        print(f"  Using ORFS core area: {cfg['core_w']}x{cfg['core_h']}, halo: {cfg['halo_x']}x{cfg['halo_y']}")
    else:
        placer = ShelfPackPlacer()
    placement = placer.place(benchmark, canvas_override=canvas_override)

    # Validate
    is_valid, violations = validate_placement(placement, benchmark)
    print(f"  Valid: {is_valid}, Overlaps: {'none' if is_valid else violations[:3]}")

    # Proxy cost
    if plc is not None:
        costs = compute_proxy_cost(placement, benchmark, plc)
        print(f"  Proxy cost: {costs['proxy_cost']:.4f}  (WL={costs['wirelength_cost']:.4f}, D={costs['density_cost']:.4f}, C={costs['congestion_cost']:.4f})")
        print(f"  Overlaps: {costs['overlap_count']}")

    # Save placement tensor
    output_dir = Path("output/shelfpack")
    output_dir.mkdir(parents=True, exist_ok=True)
    placement_file = output_dir / f"{benchmark_name}_placement.pt"
    torch.save(placement, placement_file)
    print(f"  Saved placement: {placement_file}")

    # Run ORFS if requested
    if run_orfs:
        print(f"\n  Running ORFS evaluation...")
        cmd = [
            sys.executable, "scripts/evaluate_with_orfs.py",
            "--benchmark", benchmark_name,
            "--placement", str(placement_file),
            "--no-docker",
            "--output", "output/shelfpack_orfs",
        ]
        # Ensure system yosys/openroad are found by ORFS make
        env = dict(os.environ)
        env["YOSYS_EXE"] = shutil.which("yosys") or "yosys"
        env["OPENROAD_EXE"] = shutil.which("openroad") or "openroad"
        print(f"  $ {' '.join(cmd)}")
        print(f"  YOSYS_EXE={env['YOSYS_EXE']}  OPENROAD_EXE={env['OPENROAD_EXE']}")
        subprocess.run(cmd, timeout=14400, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Single benchmark name')
    parser.add_argument('--all', action='store_true', help='All NG45 benchmarks')
    parser.add_argument('--run-orfs', action='store_true', help='Run full ORFS flow')
    args = parser.parse_args()

    if args.all:
        benchmarks = list(SOURCE_DIRS.keys())
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        print("Specify --benchmark or --all")
        return 1

    for name in benchmarks:
        evaluate_one(name, run_orfs=args.run_orfs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
