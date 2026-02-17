"""
macro_place – Macro Placement Challenge toolkit.

Install with:
    uv sync

Then import anywhere:
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement
    from macro_place.benchmark import Benchmark
"""

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
from macro_place.utils import validate_placement, visualize_placement

__all__ = [
    "Benchmark",
    "load_benchmark",
    "load_benchmark_from_dir",
    "compute_proxy_cost",
    "compute_overlap_metrics",
    "validate_placement",
    "visualize_placement",
]
