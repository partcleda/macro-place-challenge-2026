#!/usr/bin/env python3
"""
Convert IBM (ICCAD04) benchmarks to PyTorch tensor format.

These are classic academic benchmarks from the ICCAD 2004 competition.

Usage:
    python scripts/convert_ibm_benchmarks.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loader import load_benchmark_from_dir

def main():
    # Path to IBM benchmarks in MacroPlacement
    testcases_dir = Path("external/MacroPlacement/Testcases/ICCAD04")

    if not testcases_dir.exists():
        print(f"Error: {testcases_dir} not found")
        print("Run: git submodule update --init external/MacroPlacement")
        return 1

    # IBM benchmarks to convert
    ibm_benchmarks = [f"ibm{i:02d}" for i in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]

    # Output directory
    output_dir = Path("benchmarks/processed/public")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Converting IBM (ICCAD04) Benchmarks to PyTorch Tensor Format")
    print("=" * 80)
    print()

    success_count = 0
    total_count = len(ibm_benchmarks)

    for name in ibm_benchmarks:
        benchmark_dir = testcases_dir / name

        if not benchmark_dir.exists():
            print(f"⚠️  {name:20} SKIPPED (directory not found)")
            continue

        print(f"Converting {name}...", end=" ")

        try:
            # Load and convert
            benchmark, plc = load_benchmark_from_dir(str(benchmark_dir))

            # Save as .pt
            output_file = output_dir / f"{name}.pt"
            benchmark.save(str(output_file))

            print(f"✓ ({benchmark.num_macros:3} macros, {benchmark.num_nets:6} nets)")
            success_count += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print(f"✓ Converted {success_count}/{total_count} benchmarks to {output_dir}/")
    print("=" * 80)
    print()

    if success_count > 0:
        print("Next steps:")
        print("  1. Compute baselines: python scripts/compute_ibm_baselines.py")
        print("  2. Update leaderboard: python scripts/generate_leaderboard.py")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
