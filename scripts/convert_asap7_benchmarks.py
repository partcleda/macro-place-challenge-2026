#!/usr/bin/env python3
"""Convert ASAP7 benchmarks to PyTorch tensor format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from loader import load_benchmark_from_dir

def main():
    flows_dir = Path("external/MacroPlacement/Flows/ASAP7")

    benchmarks = {
        'ariane136_asap7': flows_dir / "ariane136" / "netlist" / "output_CT_Grouping",
        'nvdla_asap7': flows_dir / "nvdla" / "netlist" / "output_CT_Grouping",
        'mempool_tile_asap7': flows_dir / "mempool_tile" / "netlist" / "output_CT_Grouping",
    }

    output_dir = Path("benchmarks/processed/public")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, benchmark_dir in benchmarks.items():
        if not benchmark_dir.exists():
            print(f"⚠️  {name} SKIPPED (directory not found)")
            continue

        try:
            benchmark, plc = load_benchmark_from_dir(str(benchmark_dir))
            output_file = output_dir / f"{name}.pt"
            benchmark.save(str(output_file))
            print(f"✓ {name:25} -> {output_file.name}")
        except Exception as e:
            print(f"✗ {name:25} FAILED: {e}")

if __name__ == "__main__":
    sys.exit(main())
