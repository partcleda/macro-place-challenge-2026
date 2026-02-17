"""
Evaluation harness for macro placement submissions.

Loads a placer from a Python file, runs it on benchmarks, and prints results
with baseline comparisons.

Usage:
    uv run evaluate submissions/examples/greedy_row_placer.py
    uv run evaluate submissions/examples/greedy_row_placer.py --all
    uv run evaluate submissions/examples/greedy_row_placer.py -b ibm03
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement

# ── IBM ICCAD04 benchmark list ──────────────────────────────────────────────

BENCHMARKS = [
    "ibm01",
    "ibm02",
    "ibm03",
    "ibm04",
    "ibm06",
    "ibm07",
    "ibm08",
    "ibm09",
    "ibm10",
    "ibm11",
    "ibm12",
    "ibm13",
    "ibm14",
    "ibm15",
    "ibm16",
    "ibm17",
    "ibm18",
]

# ── Published baselines ─────────────────────────────────────────────────────

SA_BASELINES = {
    "ibm01": 1.3166,
    "ibm02": 1.9072,
    "ibm03": 1.7401,
    "ibm04": 1.5037,
    "ibm06": 2.5057,
    "ibm07": 2.0229,
    "ibm08": 1.9239,
    "ibm09": 1.3875,
    "ibm10": 2.1108,
    "ibm11": 1.7111,
    "ibm12": 2.8261,
    "ibm13": 1.9141,
    "ibm14": 2.2750,
    "ibm15": 2.3000,
    "ibm16": 2.2337,
    "ibm17": 3.6726,
    "ibm18": 2.7755,
}

REPLACE_BASELINES = {
    "ibm01": 0.9976,
    "ibm02": 1.8370,
    "ibm03": 1.3222,
    "ibm04": 1.3024,
    "ibm06": 1.6187,
    "ibm07": 1.4633,
    "ibm08": 1.4285,
    "ibm09": 1.1194,
    "ibm10": 1.5009,
    "ibm11": 1.1774,
    "ibm12": 1.7261,
    "ibm13": 1.3355,
    "ibm14": 1.5436,
    "ibm15": 1.5159,
    "ibm16": 1.4780,
    "ibm17": 1.6446,
    "ibm18": 1.7722,
}

# ── Placer loading ───────────────────────────────────────────────────────────


def _load_placer(path: Path):
    """Import a placer .py file and return an instance of its placer class.

    Convention: the first class defined in the file that has a ``place``
    method is treated as the placer.  It is instantiated with no arguments.
    """
    path = path.resolve()
    if spec := importlib.util.spec_from_file_location(path.stem, str(path)):
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        raise RuntimeError(f"Failed to load placer from {path}")

    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()

    raise RuntimeError(
        f"No placer class found in {path}.\n"
        "Expected a class with a  place(self, benchmark) -> Tensor  method."
    )


# ── Single-benchmark evaluation ─────────────────────────────────────────────


def evaluate_benchmark(placer, name: str, testcase_root: str) -> dict:
    """Run *placer* on a single benchmark and return a results dict."""
    benchmark_dir = f"{testcase_root}/{name}"
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    start = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - start

    is_valid, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)

    return {
        "name": name,
        "proxy_cost": costs["proxy_cost"],
        "wirelength": costs["wirelength_cost"],
        "density": costs["density_cost"],
        "congestion": costs["congestion_cost"],
        "overlaps": costs["overlap_count"],
        "runtime": runtime,
        "valid": is_valid,
        "sa_baseline": SA_BASELINES.get(name),
        "replace_baseline": REPLACE_BASELINES.get(name),
        "placement": placement,
        "benchmark": benchmark,
    }


# ── Pretty-printing ─────────────────────────────────────────────────────────


def _print_summary_table(results):
    """Print a multi-benchmark comparison table."""
    print()
    print("-" * 80)
    print(
        f"{'Benchmark':>8}  {'Proxy':>8}  {'SA':>8}  {'RePlAce':>8}"
        f"  {'vs SA':>8}  {'vs RePlAce':>10}  {'Overlaps':>8}"
    )
    print("-" * 80)

    for r in results:
        vs_sa = (
            ((r["sa_baseline"] - r["proxy_cost"]) / r["sa_baseline"] * 100)
            if r["sa_baseline"]
            else 0
        )
        vs_rep = (
            ((r["replace_baseline"] - r["proxy_cost"]) / r["replace_baseline"] * 100)
            if r["replace_baseline"]
            else 0
        )
        print(
            f"{r['name']:>8}  {r['proxy_cost']:>8.4f}"
            f"  {r['sa_baseline']:>8.4f}  {r['replace_baseline']:>8.4f}"
            f"  {vs_sa:>+7.1f}%  {vs_rep:>+9.1f}%  {r['overlaps']:>8}"
        )

    avg_proxy = sum(r["proxy_cost"] for r in results) / len(results)
    avg_sa = sum(r["sa_baseline"] for r in results) / len(results)
    avg_rep = sum(r["replace_baseline"] for r in results) / len(results)
    total_overlaps = sum(r["overlaps"] for r in results)
    total_runtime = sum(r["runtime"] for r in results)

    print("-" * 80)
    print(
        f"{'AVG':>8}  {avg_proxy:>8.4f}  {avg_sa:>8.4f}  {avg_rep:>8.4f}"
        f"  {(avg_sa - avg_proxy) / avg_sa * 100:>+7.1f}%"
        f"  {(avg_rep - avg_proxy) / avg_rep * 100:>+9.1f}%  {total_overlaps:>8}"
    )
    print()
    print(f"Total runtime: {total_runtime:.2f}s")
    if total_overlaps > 0:
        print(f"⚠  DISQUALIFIED: {total_overlaps} total overlaps across benchmarks")
    print()


# ── CLI entry point ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate a macro-placement submission on IBM ICCAD04 benchmarks.",
    )
    parser.add_argument(
        "placer",
        help="Path to a placer .py file (e.g. submissions/examples/greedy_row_placer.py).",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        default=None,
        help="Run on a specific benchmark (e.g. ibm01). Default: ibm01.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run on all 17 IBM benchmarks.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Visualize each placement after evaluation (saves to vis/<benchmark>.png).",
    )
    args = parser.parse_args()

    # ── resolve paths ────────────────────────────────────────────────────
    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")
    if not testcase_root.exists():
        print(f"Error: Testcases not found at {testcase_root}")
        print("Run: git submodule update --init external/MacroPlacement")
        sys.exit(1)

    # ── load placer ──────────────────────────────────────────────────────
    placer_path = Path(args.placer)
    placer = _load_placer(placer_path)
    placer_name = type(placer).__name__

    benchmarks_to_run = BENCHMARKS if args.all else [args.benchmark or "ibm01"]

    # ── run ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"evaluate · {placer_name}  ({placer_path})")
    print("=" * 80)
    print()

    results = []
    for name in benchmarks_to_run:
        print(f"  {name}...", end=" ", flush=True)
        result = evaluate_benchmark(placer, name, str(testcase_root))
        results.append(result)

        status = (
            "VALID"
            if result["overlaps"] == 0
            else f"INVALID ({result['overlaps']} overlaps)"
        )
        print(
            f"proxy={result['proxy_cost']:.4f}  "
            f"(wl={result['wirelength']:.3f} den={result['density']:.3f} cong={result['congestion']:.3f})  "
            f"{status}  [{result['runtime']:.2f}s]"
        )

        if args.vis:
            vis_dir = Path("vis")
            vis_dir.mkdir(exist_ok=True)
            save_path = str(vis_dir / f"{name}.png")
            visualize_placement(result["placement"], result["benchmark"], save_path=save_path)

    if len(results) > 1:
        _print_summary_table(results)


if __name__ == "__main__":
    main()
