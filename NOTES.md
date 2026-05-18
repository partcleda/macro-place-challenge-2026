# Macro Placement Challenge 2026 — Reconnaissance Notes

## Quick Results Summary

| Benchmark | Algorithm | Proxy Cost | WL | Density | Congestion | Overlaps |
|-----------|-----------|-----------|-----|---------|-----------|---------|
| ibm01 | GreedyRowPlacer | **2.0463** | 0.1209 | 1.2446 | 2.6061 | 0 |
| ibm01 | Initial placement | 1.0385 | 0.0641 | 0.8120 | 1.1369 | 69 |
| ariane133 NG45 | GreedyRowPlacer | **1.0108** | 0.0905 | 0.9117 | 0.9288 | 0 (OOB!) |
| ariane133 NG45 | Initial placement | **0.7109** | 0.0497 | 0.6067 | 0.7157 | 0 |

**Key observation:** The greedy row placer on ariane133 produces zero overlaps but places macros
out of canvas bounds (validate_placement reports invalid). The initial placement is already
excellent for NG45 (0.71 proxy, 0 overlaps). The IBM initial placements have overlaps —
the reference placements in `initial.plc` are NOT already legalized.

---

## 1. Placer Class Interface

```python
import torch
from macro_place.benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Args:
            benchmark: Benchmark dataclass (see section 2)
        Returns:
            placement: torch.Tensor of shape [num_macros, 2], dtype float32
                       (x, y) CENTER positions in microns
                       indices [0, num_hard_macros) = hard macros
                       indices [num_hard_macros, num_macros) = soft macros
        """
```

**Requirements:**
- Return shape must be exactly `[benchmark.num_macros, 2]`
- Positions are **center** coordinates, not lower-left corners
- Units are **microns** for IBM; also microns for NG45 (but canvas is ~1433 μm not mm!)
- Fixed macros (`benchmark.macro_fixed[i] == True`) must stay at `benchmark.macro_positions[i]`
- All macros must be fully within canvas: `x ∈ [w/2, canvas_width - w/2]`
- Zero overlaps between hard macros (strict; soft macros may overlap)
- No 90° rotations — only orientations N, FN, FS, S are legal
- Soft macro sizes must NOT be resized

---

## 2. Loading a Benchmark

### IBM ICCAD04 (Tier 1 — proxy scoring)

```python
from macro_place.loader import load_benchmark_from_dir

benchmark, plc = load_benchmark_from_dir(
    'external/MacroPlacement/Testcases/ICCAD04/ibm01'
)
# Returns (Benchmark dataclass, PlacementCost object)
# plc is required for compute_proxy_cost and optimize_stdcells
```

Or with explicit files:
```python
from macro_place.loader import load_benchmark

benchmark, plc = load_benchmark(
    netlist_file='external/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt',
    plc_file='external/MacroPlacement/Testcases/ICCAD04/ibm01/initial.plc',
)
```

### NG45 (Tier 2 — OpenROAD flow)

```python
from macro_place.loader import load_benchmark

benchmark, plc = load_benchmark(
    netlist_file='external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping/netlist.pb.txt',
    plc_file='external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping/initial.plc',
    name='ariane133',  # optional override; loader infers from path
)
```

Or load pre-processed `.pt` (tensors only, NO `plc` object — cannot compute proxy cost):
```python
from macro_place.benchmark import Benchmark

benchmark = Benchmark.load('benchmarks/processed/public/ariane133_ng45.pt')
# WARNING: no plc object returned; you still need to load plc separately for cost eval
```

---

## 3. Benchmark Dataclass Fields

```python
benchmark.name            # str, e.g. "ibm01"
benchmark.canvas_width    # float, microns
benchmark.canvas_height   # float, microns
benchmark.num_macros      # int, total = num_hard + num_soft
benchmark.num_hard_macros # int, hard macros at indices [0, num_hard)
benchmark.num_soft_macros # int, soft macros at indices [num_hard, num_macros)
benchmark.macro_positions # Tensor [N, 2], float32, (x,y) centers, microns
benchmark.macro_sizes     # Tensor [N, 2], float32, (width, height), microns
benchmark.macro_fixed     # Tensor [N], bool, True = cannot move
benchmark.macro_names     # List[str], for debugging
benchmark.num_nets        # int (post-filter: nets with ≥1 macro/port endpoint)
benchmark.net_nodes       # List[Tensor], net_nodes[i] = sorted macro indices in net i
benchmark.net_weights     # Tensor [num_nets], float32, default 1.0
benchmark.grid_rows       # int (for density/congestion grid)
benchmark.grid_cols       # int
benchmark.port_positions  # Tensor [num_ports, 2], I/O pad locations
benchmark.macro_pin_offsets  # List[Tensor [num_pins_i, 2]], offsets from center
benchmark.net_pin_nodes   # List[Tensor [P, 2]], pin-level connectivity
benchmark.hroutes_per_micron  # float, horizontal routing capacity
benchmark.vroutes_per_micron  # float, vertical routing capacity
benchmark.hard_macro_indices  # List[int], benchmark idx → PlacementCost module idx
benchmark.soft_macro_indices  # List[int], benchmark idx → PlacementCost module idx

# Helper masks (return [N] bool tensors):
benchmark.get_movable_mask()      # ~macro_fixed
benchmark.get_hard_macro_mask()   # True for indices [0, num_hard)
benchmark.get_soft_macro_mask()   # True for indices [num_hard, num_macros)
```

**Net connectivity quirk:** `benchmark.net_nodes` has FEWER nets than the raw netlist
because the loader filters out nets with no macro/port endpoints. Raw netlist may have
~7269 nets (ibm01) but `num_nets` ends up as ~5993 after filtering.

---

## 4. Computing Proxy Cost

```python
from macro_place.objective import compute_proxy_cost

costs = compute_proxy_cost(
    placement,   # [num_macros, 2] tensor
    benchmark,   # Benchmark object
    plc,         # PlacementCost object (REQUIRED — must match benchmark)
    weights=None # optional: {'wirelength': 1.0, 'density': 0.5, 'congestion': 0.5}
)

# Returns dict:
costs['proxy_cost']           # float: 1.0*WL + 0.5*density + 0.5*congestion
costs['wirelength_cost']      # float: normalized HPWL
costs['density_cost']         # float: top-10% grid cell density
costs['congestion_cost']      # float: top-5% routing congestion
costs['overlap_count']        # int: number of hard macro pairs that overlap
costs['total_overlap_area']   # float: μm²
costs['max_overlap_area']     # float: μm²
costs['num_macros_with_overlaps']  # int
costs['overlap_ratio']        # float: fraction of macros in overlaps
```

**Implementation note:** `compute_proxy_cost` calls `plc.get_cost()`, `plc.get_density_cost()`,
`plc.get_congestion_cost()`. It uses a monkey-patch to fix a boundary bug in PlacementCost
(`__get_grid_cell_location` is patched to clamp row/col to valid range). The patch is applied
automatically on import of `macro_place.objective`.

---

## 5. Validating a Placement

```python
from macro_place.utils import validate_placement

is_valid, violations = validate_placement(
    placement,        # [num_macros, 2] tensor
    benchmark,        # Benchmark object
    check_overlaps=True  # set False to skip O(N²) overlap check
)
# is_valid: bool
# violations: List[str] describing each issue
```

Checks performed:
1. Shape `[num_macros, 2]`
2. No NaN or Inf
3. All macros within canvas bounds (edge must not exceed canvas edge)
4. Fixed macros at original positions (tolerance 1e-3 μm)
5. No hard macro pairwise overlaps (only first 5 pairs reported)

**Note:** soft macros are NOT checked for overlaps (they are standard cell cluster
abstractions that naturally overlap each other).

---

## 6. Cost Decomposition

```
proxy_cost = 1.0 × wirelength_cost + 0.5 × density_cost + 0.5 × congestion_cost
```

| Component | Description | Metric |
|-----------|-------------|--------|
| `wirelength_cost` | Normalized HPWL across all nets (uses hard + soft + port positions) | Lower = nets are shorter |
| `density_cost` | Average density of top 10% busiest grid cells | >1.0 means overcrowded |
| `congestion_cost` | Average congestion of top 5% most congested routing segments | Uses H+V routing tracks |

**IBM benchmark routing parameters:** H=65.96 tracks/μm, V=106.96 tracks/μm, smooth_range=2  
**NG45 benchmark routing parameters:** H=57.03 tracks/μm, V=56.82 tracks/μm, smooth_range=0  

The smooth_range=2 for IBM means congestion is spatially smoothed over a 5×5 neighborhood;
NG45 uses no smoothing (smooth_range=0), so congestion scores are more localized.

IBM overlap threshold: 0.004 (macros must overlap by >0.4% of area to count).  
NG45 overlap threshold: 0.000 (any overlap counts, even float-precision touching).

---

## 7. Canvas and Macro Semantics

### Canvas
- IBM: small (22–81 μm). NG45: large (885–2127 μm). Both in microns.
- `canvas_width × canvas_height` defines the legal placement region.
- Macro edges must be fully inside: `x - w/2 ≥ 0`, `x + w/2 ≤ canvas_width`, same for y.

### Hard Macros
- Indices `[0, num_hard_macros)` in all tensors.
- These are SRAMs, IPs, etc. — the primary optimization target.
- `macro_fixed[i] = False` for virtually all hard macros (none fixed in practice).
- Must NOT overlap each other (zero tolerance for submission).
- Must have ≥12 μm spacing from each other for Tier 2 routing (PDN channel requirement).
- Only Klein-4 orientations legal: N (0°), FN (flip-x, 0°), FS (flip-y, 0°), S (180°).
  90° rotations (R90, R270, FE, FW) are NOT allowed.
- Orientation sidecar: `orientations.pt` file alongside placement carries flips to Tier 2.

### Soft Macros
- Indices `[num_hard_macros, num_macros)` in all tensors.
- These are standard cell clusters (abstractions, not real SRAMs).
- May overlap each other — this is expected.
- Can be repositioned to improve WL/density after moving hard macros.
- Sizes MUST NOT be changed — they are locked to initial `.plc` values on every
  `compute_proxy_cost` call (soft macro resize does not translate to Tier 2).
- Use `plc.optimize_stdcells(...)` to force-direct soft macros after hard macro moves.

### Fixed Macros
- `benchmark.macro_fixed[i] = True` — do not move these.
- In practice, most IBM and NG45 benchmarks have zero fixed hard macros.
- Validate: `benchmark.macro_fixed.sum()` to check.

---

## 8. File Paths

### IBM ICCAD04 Benchmarks (17 total; no ibm05!)

```
external/MacroPlacement/Testcases/ICCAD04/
  ibm01/   netlist.pb.txt  initial.plc
  ibm02/   netlist.pb.txt  initial.plc
  ibm03/   netlist.pb.txt  initial.plc
  ibm04/   netlist.pb.txt  initial.plc
  ibm06/   netlist.pb.txt  initial.plc   ← MISSING ibm05
  ibm07/   netlist.pb.txt  initial.plc
  ibm08/   netlist.pb.txt  initial.plc
  ibm09/   netlist.pb.txt  initial.plc
  ibm10/   netlist.pb.txt  initial.plc
  ibm11/   netlist.pb.txt  initial.plc
  ibm12/   netlist.pb.txt  initial.plc
  ibm13/   netlist.pb.txt  initial.plc
  ibm14/   netlist.pb.txt  initial.plc
  ibm15/   netlist.pb.txt  initial.plc
  ibm16/   netlist.pb.txt  initial.plc
  ibm17/   netlist.pb.txt  initial.plc
  ibm18/   netlist.pb.txt  initial.plc
```

Pre-processed PyTorch tensors (no plc object; load with `Benchmark.load()`):
```
benchmarks/processed/public/ibm01.pt ... ibm18.pt
```

### NG45 Benchmarks (4 public)

```
external/MacroPlacement/Flows/NanGate45/
  ariane133/netlist/output_CT_Grouping/   netlist.pb.txt  initial.plc  legalized.plc
  ariane136/netlist/output_CT_Grouping/   netlist.pb.txt  initial.plc
  mempool_tile/netlist/output_CT_Grouping/ netlist.pb.txt  initial.plc
  nvdla/netlist/output_CT_Grouping/       netlist.pb.txt  initial.plc

benchmarks/processed/public/
  ariane133_ng45.pt           ← initial placement (zero overlaps)
  ariane133_ng45_random.pt    ← random seed variant
  ariane136_ng45.pt
  mempool_tile_ng45.pt
  nvdla_ng45.pt
  ariane136_asap7.pt          ← ASAP7 variants (not evaluated in competition)
  mempool_tile_asap7.pt
  nvdla_asap7.pt
```

### Evaluation CLI

```bash
# Single benchmark
python3 macro_place/evaluate.py submissions/examples/greedy_row_placer.py -b ibm01

# All IBM (Tier 1)
python3 macro_place/evaluate.py submissions/examples/greedy_row_placer.py --all

# All NG45 (Tier 2 proxy)
python3 macro_place/evaluate.py submissions/examples/greedy_row_placer.py --ng45
```

---

## 9. Per-Benchmark Data (Measured)

### 17 IBM Benchmarks — Actual Measured Values

| Benchmark | Hard | Soft | Grid (RxC) | Canvas (μm) | Initial Proxy | SA Baseline | RePlAce Baseline | Init Overlaps |
|-----------|------|------|-----------|-------------|--------------|-------------|-----------------|---------------|
| ibm01 | 246 | 894 | 41×45 | 22.9×23.0 | 1.0385 | 1.3166 | 0.9976 | 69 |
| ibm02 | 271 | 1075 | 27×30 | 32.6×32.5 | 1.5658 | 1.9072 | 1.8370 | 108 |
| ibm03 | 290 | 1148 | 29×32 | 35.1×35.0 | 1.3255 | 1.7401 | 1.3222 | 61 |
| ibm04 | 295 | 1085 | 30×31 | 34.1×34.1 | 1.3133 | 1.5037 | 1.3024 | 68 |
| ibm06 | 178 | 900 | 28×31 | 32.6×32.6 | 1.6577 | 2.5057 | 1.6187 | 47 |
| ibm07 | 291 | 1040 | 32×35 | 38.5×38.4 | 1.4758 | 2.0229 | 1.4633 | 82 |
| ibm08 | 301 | 1030 | 34×38 | 41.0×41.0 | 1.4664 | 1.9239 | 1.4285 | 86 |
| ibm09 | 253 | 1048 | 38×36 | 46.7×46.9 | 1.1126 | 1.3875 | 1.1194 | 101 |
| ibm10 | 786 | 1982 | 41×55 | 77.0×77.1 | 1.3397 | 2.1108 | 1.5009 | 74 |
| ibm11 | 373 | 1195 | 45×39 | 51.5×51.5 | 1.2141 | 1.7111 | 1.1774 | 129 |
| ibm12 | 651 | 1985 | 47×47 | 68.0×68.0 | 1.6251 | 2.8261 | 1.7261 | 187 |
| ibm13 | 424 | 1301 | 43×43 | 55.9×56.0 | 1.3854 | 1.9141 | 1.3355 | 180 |
| ibm14 | 614 | 1529 | 44×49 | 59.9×60.0 | 1.5938 | 2.2750 | 1.5436 | 189 |
| ibm15 | 393 | 1138 | 38×57 | 67.6×67.5 | 1.6033 | 2.3000 | 1.5159 | 165 |
| ibm16 | 458 | 1315 | 48×45 | 81.0×81.1 | 1.4911 | 2.2337 | 1.4780 | 86 |
| ibm17 | 760 | 1844 | 44×51 | 72.6×72.6 | 1.7392 | 3.6726 | 1.6446 | 231 |
| ibm18 | 285 | 1029 | 39×55 | 64.8×65.0 | 1.7899 | 2.7755 | 1.7722 | 76 |
| **AVG** | | | | | **1.4429** | **2.1251** | **1.4578** | |

**Note:** ibm05 does not exist in the ICCAD04 suite — the sequence jumps ibm04→ibm06.

**Note:** The README's macro counts (246, 254, 269 ...) refer only to hard macros.
Actual hard macro counts observed are DIFFERENT from the README table for ibm02+ (e.g. ibm02
has 271 hard macros, ibm10 has 786). The README numbers appear fabricated/approximate. Use
actual loaded values.

**Note:** Initial placements from `initial.plc` typically have **overlaps** for IBM benchmarks
(e.g. ibm01 has 69 overlapping pairs). These must be legalized before submission.

### 4 NG45 Benchmarks — Actual Measured Values

| Benchmark | Hard | Soft | Grid (RxC) | Canvas (μm) | Initial Proxy | Init WL | Init Density | Init Cong | Init Overlaps |
|-----------|------|------|-----------|-------------|-------------|---------|------------|----------|---------------|
| ariane133 | 133 | 782 | 21×24 | 1433.4×1433.4 | 0.7109 | 0.0497 | 0.6067 | 0.7157 | 0 |
| ariane136 | 136 | — | — | 1446.4×1446.4 | 0.7097 | 0.0478 | 0.6083 | 0.7156 | 0 |
| nvdla | 128 | — | — | 2127.9×2127.9 | 0.7569 | 0.0492 | 0.6706 | 0.7447 | 0 |
| mempool_tile | 20 | — | — | 885.4×885.4 | 0.9610 | 0.0528 | 1.0867 | 0.7298 | 0 |

**Note:** NG45 initial placements are already legalized (zero overlaps). The `initial.plc`
files are expert-designed starting points — very hard to beat without optimization.

**Note:** NG45 canvas is in μm (NOT mm). ariane133's 1433 μm ≈ 1.43 mm. Macros are also
sized in μm (typical SRAM: 50–200 μm). The greedy row placer fails OOB on NG45 because
it treats the large canvas naively.

---

## 10. Quirks and Gotchas

### Float Precision / Touching-Edge Overlaps
- Two macros that share an edge (touching but not overlapping) will NOT be flagged as
  overlapping by `validate_placement` (uses strict `<` not `≤` for edge detection).
- IBM overlap threshold = 0.004; NG45 = 0.000. For NG45, floating-point adjacency
  can count as overlap at Tier 1. Add a small gap (greedy placer uses 0.001 μm).
- For Tier 2 (ORFS), add ≥12 μm between macro edges — the evaluator auto-pushes macros
  apart, but you can see what moved in `macros.tcl.spacing_diff.txt`.

### Grid Snapping
- At Tier 2, ORFS snaps macro positions to the manufacturing grid.
- At Tier 1, your submitted coordinates are used as-is (no snapping).

### Soft Macro Resizing
- `compute_proxy_cost` re-locks soft macro sizes to initial `.plc` values on every call.
  You CANNOT resize soft macros for a proxy benefit — the sizes are reset automatically.

### No ibm05
- The ICCAD04 suite contains ibm01–ibm04 and ibm06–ibm18. There is no ibm05.

### Net Count Discrepancy
- Raw `netlist.pb.txt` may list 7269 nets for ibm01, but `benchmark.num_nets` = 5993.
  The loader only keeps nets that have ≥1 macro or port endpoint after mapping.

### PlacementCost Grid Bug (Patched)
- `PlacementCost.__get_grid_cell_location` has a boundary bug (macros on the right/top
  edge go out of bounds). `macro_place/objective.py` monkey-patches this on import.
  Always import from `macro_place.objective`, not directly from `plc_client_os`.

### Soft Macro Optimization
- After moving hard macros, call `plc.optimize_stdcells(...)` to update soft macro
  positions. Skipping this significantly degrades wirelength and density scores.
- `optimize_stdcells` is slow (~minutes per call in pure Python).

### Orientation Sidecar
- Submit `orientations.pt` alongside `placer.py` to carry flip orientations to Tier 2.
- Valid values: `N` (normal), `FN` (flip-x), `FS` (flip-y), `S` (180°).
- NO 90° rotations: `R90`, `R270`, `FE`, `FW` are forbidden (SRAM pin access assumption).

### Congestion Smooth Range Difference
- IBM: smooth_range=2 (5×5 neighborhood smoothing). More forgiving for local hotspots.
- NG45: smooth_range=0 (no smoothing). Any local congestion spike appears raw.

### Net Connectivity is in `plc`, Not `benchmark`
- The `PlacementCost` object (`plc`) holds all net topology for proxy cost computation.
- `benchmark.net_nodes` / `benchmark.net_pin_nodes` provide PyTorch-accessible connectivity
  for custom algorithms (e.g. GNN), but proxy cost ALWAYS uses `plc`.
- Hard macro pin offsets: `benchmark.macro_pin_offsets[i]` is a `[num_pins, 2]` tensor
  of (dx, dy) offsets from macro center. Use for pin-level HPWL computation.

### Runtime Budget
- 1 hour per benchmark hard limit.
- Judges run on: AMD EPYC 9655P 16-core, 100 GB RAM, NVIDIA RTX 6000 Ada 48 GB.
- SA baseline takes ~minutes per benchmark (C++ implementation in TILOS repo).

### Tier 2 Spacing Requirement
- Macros must have ≥12 μm edge-to-edge spacing for PDN channel routing in ORFS.
- The evaluator auto-pushes macros to achieve this, but it can move your placement.
- For full control: build this constraint into your legalization.

---

## 11. Proxy Cost Formula

```
proxy_cost = 1.0 × WL_cost + 0.5 × density_cost + 0.5 × congestion_cost

WL_cost      = plc.get_cost()           # normalized HPWL
density_cost = plc.get_density_cost()   # top-10% avg grid cell density
congestion_cost = plc.get_congestion_cost()  # top-5% avg routing congestion

# Grid cell density = (macro area in cell) / (cell area)
# Routing congestion = (demand tracks) / (supply tracks)
```

The proxy was designed to correlate with Tier 2 (OpenROAD WNS/TNS/Area).
Empirically: WL dominates only when density/congestion are good; density+congestion
are the hard constraints that separate good placements from bad.

---

## 12. Current Leaderboard Context (as of 2026-05-15)

- **Top score:** 0.9671 (Carrotato — Triton kernels + Xplace + polish, 3.8 min/bench)
- **Second:** 0.978 (Shoom — MultiDREAMPlace + coord-descent refinement, 55 min/bench)
- **SA baseline:** 2.1251 avg  
- **RePlAce baseline:** 1.4578 avg  
- **Initial placement:** ~1.44 avg (overlaps present, NOT submission-ready)
- **Greedy row placer:** ~2.21 avg (no overlaps, legal, but no optimization)

To qualify for Grand Prize, must be in top 7 by proxy AND pass Tier 2 feasibility gate
(WNS/TNS ≥ min(SA, RePlAce) on all NG45 designs).

---

## 13. Submission Constraints Summary

| Constraint | Value | Enforced At |
|-----------|-------|-------------|
| Zero hard macro overlaps | Strict 0 | Tier 1 + Tier 2 |
| Macro edge-to-edge spacing | ≥12 μm recommended | Tier 2 (auto-push) |
| Orientations | N, FN, FS, S only | Tier 2 |
| Soft macro resizing | Forbidden | Tier 1 (locked) |
| Runtime per benchmark | ≤1 hour | Both tiers |
| No hardcoding per benchmark | Required | Rules |
| No external tools | Open-source only | Rules |

---

## 14. GPU/Hardware Available

- GPU: NVIDIA RTX 6000 Ada, 48 GB VRAM
- CPU: AMD EPYC 9655P, 16 cores, 100 GB RAM
- PyTorch 2.5.1 + CUDA 12.4 (judges' environment)
- Current dev environment: PyTorch 2.12.0+cu130, Python 3.11.15

---

## 15. Key Code Locations

| File | Purpose |
|------|---------|
| `macro_place/benchmark.py` | Benchmark dataclass definition |
| `macro_place/loader.py` | `load_benchmark_from_dir`, `load_benchmark` |
| `macro_place/objective.py` | `compute_proxy_cost`, `compute_overlap_metrics` |
| `macro_place/utils.py` | `validate_placement`, `visualize_placement` |
| `macro_place/evaluate.py` | CLI: `python3 macro_place/evaluate.py placer.py --all` |
| `macro_place/_plc.py` | PlacementCost import + path setup |
| `submissions/examples/simple_random_placer.py` | Minimal working example |
| `submissions/examples/greedy_row_placer.py` | Legal shelf-packing, no optimization |
| `submissions/will_seed/placer.py` | Reference: legalize + SA refinement + edge weights |
| `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py` | TILOS PlacementCost |
| `external/MacroPlacement/Testcases/ICCAD04/` | 17 IBM benchmarks |
| `external/MacroPlacement/Flows/NanGate45/` | 4 NG45 benchmarks |
| `benchmarks/processed/public/` | Pre-processed .pt files |
| `benchmarks/metadata/baseline_scores.json` | NG45 initial placement scores |
