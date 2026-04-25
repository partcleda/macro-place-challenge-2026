from pathlib import Path
from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import visualize_placement
from graph import build_weight_graph
from init import compute_initial_placement
from macro_place.objective import compute_proxy_cost
from sa import sa_refine

BENCHMARKS = [
    "ibm01"
    # "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    # "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

total_proxy = 0.0

for name in BENCHMARKS:
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    benchmark, plc = load_benchmark_from_dir(str(root))

    # build position lookup
    pos_map = {m.get_name(): m for m in plc.modules_w_pins}

    # build graph
    adj = build_weight_graph(benchmark, plc)
    # compute initial placement
    placement = compute_initial_placement(benchmark, adj, pos_map)

    # compute proxy costs
    costs = compute_proxy_cost(placement, benchmark, plc)
        # SA refinement
    print(f"Before SA: {costs['proxy_cost']:.4f}, Overlaps: {costs['overlap_count']}")
    placement = sa_refine(placement, benchmark, plc, num_iterations=100)
    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"After SA: {costs['proxy_cost']:.4f}, Overlaps: {costs['overlap_count']}")
    total_proxy += costs['proxy_cost']
    print(f"{'Benchmark':<12} {'Proxy':>8} {'WL':>8} {'Den':>8} {'Cong':>8} {'Overlaps':>10}")
    print("-" * 60)
    print(f"{name:<12} {costs['proxy_cost']:>8.4f} {costs['wirelength_cost']:>8.4f} {costs['density_cost']:>8.4f} {costs['congestion_cost']:>8.4f} {costs['overlap_count']:>10}")
    
    img_name = name + ".png"
    img_path = Path("submissions/src/images") / name
    # visualize
    visualize_placement(placement, benchmark, save_path=str(img_path))
    
print(f'Average Proxy: {total_proxy / len(BENCHMARKS)}')