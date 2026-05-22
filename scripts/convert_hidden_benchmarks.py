#!/usr/bin/env python3
"""Convert hidden Tier 2 benchmark(s) to PyTorch tensor format.

Hidden benchmarks are stored in benchmarks/processed/hidden/ and are not
included in the public benchmark set. They are used for Tier 2 evaluation
to prevent overfitting to the public designs.

Current hidden designs:
  mempool_group_ng45 — full 4x4 mempool group, 324 hard macros, ~3373x3371 um
"""

from pathlib import Path

from macro_place.loader import load_benchmark_from_dir

def main():
    base = Path("external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW/test")

    benchmarks = {
        'mempool_group_ng45': base / "mempool_group_ng45",
    }

    output_dir = Path("benchmarks/processed/hidden")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, benchmark_dir in benchmarks.items():
        if not benchmark_dir.exists():
            print(f"  {name} SKIPPED (directory not found: {benchmark_dir})")
            continue
        try:
            benchmark, _ = load_benchmark_from_dir(str(benchmark_dir))
            output_file = output_dir / f"{name}.pt"
            benchmark.save(str(output_file))
            print(f"  {name:30} -> {output_file}  ({benchmark.num_hard_macros} hard macros)")
        except Exception as e:
            print(f"  {name:30} FAILED: {e}")

if __name__ == "__main__":
    import sys; sys.exit(main())
