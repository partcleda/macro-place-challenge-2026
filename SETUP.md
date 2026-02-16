# Setup & API Reference

## Installation

```bash
# Clone the repository
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── loader.py       # Load benchmarks from ICCAD04 format
│   ├── benchmark.py    # Benchmark dataclass (PyTorch tensors)
│   ├── objective.py    # Proxy cost computation
│   └── utils.py        # Validation and visualization
├── submissions/
│   └── examples/       # Example placers (greedy_row_placer.py, simple_random_placer.py)
├── external/
│   └── MacroPlacement/ # TILOS evaluator and ICCAD04 testcases
├── benchmarks/
│   └── processed/      # Pre-processed .pt benchmark files
└── SETUP.md            # This file
```

## API Reference

### Loading a Benchmark

```python
import sys
sys.path.insert(0, 'src')
from loader import load_benchmark_from_dir

benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')
```

Returns:
- `benchmark`: A `Benchmark` dataclass with PyTorch tensors
- `plc`: A `PlacementCost` object (needed for cost computation)

### Benchmark Object

The `Benchmark` dataclass contains:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Benchmark name (e.g., "ibm01") |
| `canvas_width` | `float` | Canvas width in microns |
| `canvas_height` | `float` | Canvas height in microns |
| `num_macros` | `int` | Number of hard macros |
| `macro_positions` | `Tensor [N, 2]` | (x, y) center positions |
| `macro_sizes` | `Tensor [N, 2]` | (width, height) of each macro |
| `macro_fixed` | `Tensor [N]` | Boolean mask of fixed macros |
| `macro_names` | `List[str]` | Macro names for debugging |
| `num_nets` | `int` | Number of nets |
| `grid_rows`, `grid_cols` | `int` | Grid dimensions for density/congestion |

Helper methods:
- `benchmark.get_movable_mask()` — returns `~macro_fixed`
- `benchmark.save(path)` / `Benchmark.load(path)` — serialize to/from `.pt` files

### Computing Proxy Cost

```python
from objective import compute_proxy_cost

costs = compute_proxy_cost(placement, benchmark, plc)
```

**Input**: `placement` — a `[num_macros, 2]` tensor of (x, y) center positions.

**Output**: A dictionary with:

| Key | Description |
|-----|-------------|
| `proxy_cost` | Weighted sum: 1.0 × WL + 0.5 × density + 0.5 × congestion |
| `wirelength_cost` | Normalized HPWL across all nets |
| `density_cost` | Top 10% grid cell density |
| `congestion_cost` | Top 5% routing congestion with smoothing |
| `overlap_count` | Number of overlapping macro pairs |
| `total_overlap_area` | Total overlap area in μm² |
| `overlap_ratio` | Fraction of macros involved in overlaps |

### Validating a Placement

```python
from utils import validate_placement

is_valid, violations = validate_placement(placement, benchmark)
```

Checks:
- Correct tensor shape
- No NaN/Inf values
- All macros within canvas bounds
- Fixed macros at original positions
- Zero macro-to-macro overlaps

### Visualizing a Placement

```python
from utils import visualize_placement

visualize_placement(placement, benchmark, save_path='output.png')
```

## Writing a Placer

Your placer takes a `Benchmark` and returns a `[num_macros, 2]` tensor of positions:

```python
import torch
from benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask()

        # Your algorithm here — modify positions for movable macros
        # ...

        return placement
```

Key constraints:
- Positions are **center coordinates** (not corners)
- Fixed macros must stay at their original positions
- All macros must be fully within canvas bounds
- **Zero overlaps** required (automatic disqualification otherwise)

See `submissions/examples/greedy_row_placer.py` for a complete working example.

## Net Connectivity

Net connectivity is stored inside the `PlacementCost` object (`plc`), not in the `Benchmark` tensors. The proxy cost computation uses it automatically.

If you need direct access to net data for your algorithm (e.g., for a GNN), you can access it through the PlacementCost API:

```python
# Number of nets
print(plc.net_cnt)

# Access individual modules and their connections
for i, module in enumerate(plc.modules_w_pins):
    print(module.get_name(), module.get_pos())
```

See the [TILOS MacroPlacement source](https://github.com/TILOS-AI-Institute/MacroPlacement/blob/main/CodeElements/Plc_client/plc_client_os.py) for the full PlacementCost API.

## Running Benchmarks

### IBM Benchmarks (Tier 1 — Proxy Cost)

The 17 IBM ICCAD04 benchmarks are in `external/MacroPlacement/Testcases/ICCAD04/`. Run a single benchmark or the full suite using the demo placer:

```bash
# Single benchmark
python submissions/examples/greedy_row_placer.py --benchmark ibm01

# All 17 benchmarks with comparison table
python submissions/examples/greedy_row_placer.py --all
```

To evaluate your own placer on all benchmarks, follow the same pattern — loop over the benchmark directories:

```python
BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

for name in BENCHMARKS:
    benchmark, plc = load_benchmark_from_dir(f'external/MacroPlacement/Testcases/ICCAD04/{name}')
    placement = my_placer.place(benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"{name}: proxy={costs['proxy_cost']:.4f}  overlaps={costs['overlap_count']}")
```

### NG45 Designs (Tier 2 — OpenROAD Flow)

The top 7 submissions by proxy score will be evaluated through the full OpenROAD PnR flow on NanGate45 designs. These designs are located in the TILOS repository:

```
external/MacroPlacement/Flows/NanGate45/
├── ariane133/    # RISC-V core, 133 macros
├── ariane136/    # RISC-V core, 136 macros
└── mempool_tile/ # Memory pool, 20 macros
```

Pre-processed `.pt` versions are available in `benchmarks/processed/public/` for quick loading:

```python
from benchmark import Benchmark

benchmark = Benchmark.load('benchmarks/processed/public/ariane133_ng45_random.pt')
```

The OpenROAD flow evaluation measures WNS (worst negative slack), TNS (total negative slack), and Area. Participants do not need to run OpenROAD themselves — the judges will run it on top submissions.

#### Running ORFS Locally (Optional)

If you want to test your placement through the full PnR flow locally, we provide `scripts/evaluate_with_orfs.py` which automates the entire process.

**Prerequisites**: Install [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) adjacent to this repository:

```bash
cd ..
git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
cd macro-place-challenge-2026
```

**Run the ORFS evaluation**:

```bash
# Evaluate a single NG45 design (uses default placement)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker

# Evaluate with your own placement (saved as a [num_macros, 2] tensor)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker \
    --placement my_placement.pt

# Evaluate all NG45 designs
python scripts/evaluate_with_orfs.py --all --no-docker

# Point to a custom ORFS installation
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 \
    --orfs-root /path/to/OpenROAD-flow-scripts --no-docker
```

The script will:
1. Load the benchmark and compute proxy cost
2. Generate a macro placement TCL script (handling the name mapping between protobuf and ODB formats)
3. Copy the design config into ORFS with necessary patches
4. Run the full ORFS flow (synthesis → floorplan → placement → CTS → routing)
5. Parse and report WNS, TNS, Area, and other metrics

A full ORFS run takes approximately 3-8 hours per design depending on the benchmark and machine.
