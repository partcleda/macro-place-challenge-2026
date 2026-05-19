"""Smoke tests to verify the competition infrastructure works end-to-end."""

import torch
import pytest
from pathlib import Path

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


TESTCASE_ROOT = Path("external/MacroPlacement/Testcases/ICCAD04")


@pytest.fixture
def ibm01():
    """Load ibm01 benchmark from source."""
    path = TESTCASE_ROOT / "ibm01"
    if not path.exists():
        pytest.skip("TILOS submodule not initialized")
    return load_benchmark_from_dir(str(path))


def test_load_benchmark_pt():
    """Benchmark .pt files can be loaded."""
    pt = Path("benchmarks/processed/public/ibm01.pt")
    if not pt.exists():
        pytest.skip("Benchmark .pt files not present")
    b = Benchmark.load(str(pt))
    assert b.num_macros > 0
    assert b.macro_positions.shape == (b.num_macros, 2)
    assert b.macro_sizes.shape == (b.num_macros, 2)


def test_load_benchmark_from_dir(ibm01):
    """Benchmark can be loaded from ICCAD04 directory."""
    benchmark, plc = ibm01
    assert benchmark.num_macros > 0
    assert benchmark.canvas_width > 0
    assert benchmark.canvas_height > 0


def test_compute_proxy_cost(ibm01):
    """Proxy cost can be computed on the default placement."""
    benchmark, plc = ibm01
    costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    assert "proxy_cost" in costs
    assert "wirelength_cost" in costs
    assert "density_cost" in costs
    assert "congestion_cost" in costs
    assert costs["proxy_cost"] > 0


def test_validate_placement(ibm01):
    """Validation function runs without errors on default placement."""
    benchmark, plc = ibm01
    is_valid, violations = validate_placement(benchmark.macro_positions, benchmark)
    # Default placement may have overlaps — we just check the function works
    assert isinstance(is_valid, bool)
    assert isinstance(violations, list)


def test_net_pin_nodes(ibm01):
    """Loader exposes pin-level net connectivity consistent with net_nodes."""
    import torch

    benchmark, _ = ibm01
    assert len(benchmark.net_pin_nodes) == benchmark.num_nets

    for net_id, (net_pins, net_owners) in enumerate(
        zip(benchmark.net_pin_nodes, benchmark.net_nodes)
    ):
        # Shape: [pins_in_net, 2] — columns are (owner_idx, pin_slot)
        assert net_pins.ndim == 2 and net_pins.shape[1] == 2, (
            f"net {net_id}: net_pins shape {net_pins.shape}"
        )

        # Dedup+sort of owner column must match existing net_nodes exactly
        owners_sorted = torch.unique(net_pins[:, 0]).sort().values
        assert torch.equal(owners_sorted, net_owners), (
            f"net {net_id}: owners {owners_sorted.tolist()} != net_nodes {net_owners.tolist()}"
        )

        # Pin slots must index into macro_pin_offsets[owner] for hard macros
        for owner, slot in net_pins.tolist():
            if owner < benchmark.num_hard_macros:
                num_pins_on_macro = benchmark.macro_pin_offsets[owner].shape[0]
                assert slot < num_pins_on_macro, (
                    f"net {net_id}: owner {owner} slot {slot} >= "
                    f"macro_pin_offsets[{owner}].shape[0] {num_pins_on_macro}"
                )
            else:
                assert slot == 0, (
                    f"net {net_id}: non-hard-macro owner {owner} must use slot 0, got {slot}"
                )


def test_benchmark_save_load_roundtrip(ibm01, tmp_path):
    """Benchmark.save/load preserves net_pin_nodes."""
    import torch

    benchmark, _ = ibm01
    out = tmp_path / "roundtrip.pt"
    benchmark.save(str(out))
    loaded = Benchmark.load(str(out))

    assert len(loaded.net_pin_nodes) == len(benchmark.net_pin_nodes)
    for a, b in zip(loaded.net_pin_nodes, benchmark.net_pin_nodes):
        assert torch.equal(a, b)


def test_greedy_row_placer(ibm01):
    """Greedy row placer produces a valid, zero-overlap placement."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "greedy_row_placer",
        "submissions/examples/greedy_row_placer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    benchmark, plc = ibm01
    placer_cls = next(
        cls for name, cls in vars(mod).items()
        if isinstance(cls, type) and hasattr(cls, "place")
    )
    placer = placer_cls()
    placement = placer.place(benchmark)

    assert placement.shape == (benchmark.num_macros, 2)
    costs = compute_proxy_cost(placement, benchmark, plc)
    assert costs["overlap_count"] == 0, f"Greedy placer has {costs['overlap_count']} overlaps"
