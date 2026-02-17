# Example Submissions

This directory contains example placement algorithms demonstrating the PyTorch infrastructure.

## Simple Random Placer

**File:** `simple_random_placer.py`

A minimal working example that demonstrates the complete workflow:
1. Load a benchmark using our loader
2. Implement a simple placement algorithm
3. Validate the placement
4. Compute proxy cost

### Running the Example

```bash
# Make sure you're in the repository root
cd /path/to/macro-place-challenge-2026

# Install the package (one-time setup)
uv sync

# Run the example
python submissions/examples/simple_random_placer.py
```

### Expected Output

```
============================================================
Simple Random Placer - Example Workflow
============================================================

[1/4] Loading benchmark from external/MacroPlacement/Testcases/ICCAD04/ibm01...
  ✓ Loaded ibm01
    - Macros: 246 (246 movable, 0 fixed)
    - Nets: 7269
    - Canvas: 22.9 × 23.0 μm
    - Grid: 41 × 45

[2/4] Running placer...
  ✓ Generated placement for 246 macros

[3/4] Validating placement...
  ✓ Placement is valid!

[4/4] Computing proxy cost...
  ✓ Costs computed:
    - Wirelength:  0.128768
    - Density:     1.276113
    - Congestion:  2.248285
    - Proxy Cost:  1.890967 ⭐
```

### Algorithm Overview

The `SimpleRandomPlacer` class:
- Places each movable macro at a random position within canvas bounds
- Ensures macro centers keep the macro edges within the canvas
- Respects fixed macros (keeps them at their original positions)
- Uses a random seed for reproducibility

**Note:** Random placement is just a baseline. It typically performs worse than the initial placement. This example shows the infrastructure - you should implement smarter algorithms (simulated annealing, reinforcement learning, GNN-based, etc.)!

## Creating Your Own Placer

### Minimal Interface

```python
from macro_place.benchmark import Benchmark
import torch

class MyPlacer:
    def __init__(self, **kwargs):
        # Initialize your algorithm
        pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Generate placement.

        Args:
            benchmark: Benchmark object with:
                - benchmark.num_macros: Number of macros
                - benchmark.macro_sizes: [num_macros, 2] (width, height)
                - benchmark.macro_fixed: [num_macros] bool (True if fixed)
                - benchmark.macro_positions: [num_macros, 2] initial positions
                - benchmark.canvas_width: float
                - benchmark.canvas_height: float

        Returns:
            placement: [num_macros, 2] tensor of (x, y) centers
        """
        placement = torch.zeros(benchmark.num_macros, 2)

        # Your algorithm here
        # ...

        # Remember to respect fixed macros!
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return placement
```

### Complete Workflow

```python
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

# 1. Load benchmark
benchmark, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")

# 2. Run your placer
placer = MyPlacer()
placement = placer.place(benchmark)

# 3. Validate (optional but recommended)
is_valid, violations = validate_placement(placement, benchmark)
if not is_valid:
    print(f"Invalid placement: {violations}")

# 4. Compute cost
costs = compute_proxy_cost(placement, benchmark, plc)
print(f"Proxy cost: {costs['proxy_cost']:.6f}")
```

## Standard Cells (Soft Macros)

**Q: Do I need to place standard cells?**

**A: No!** The benchmarks only require placing **hard macros** (large memory blocks). Standard cells are pre-grouped into "soft macros" and already positioned in the benchmark. The cost computation automatically accounts for them.

From the ibm01 benchmark:
- **Hard macros**: 246 (you place these)
- **Soft macros**: 894 (pre-placed, used for cost computation)
- **Nets**: 7269 (connect both hard and soft macros)

This is standard practice in macro placement - you optimize the placement of large blocks, and standard cells are handled separately.

## Tips for Competition

1. **Start with the initial placement** - It's usually quite good! Try local improvements.
2. **Check validity early** - Use `validate_placement()` during development
3. **Focus on the proxy cost** - That's what you're optimizing for
4. **Test on multiple benchmarks** - Don't overfit to ibm01
5. **Use PyTorch features** - Batching, GPU, autograd can help with learning-based methods

## Cost Components

The proxy cost is a weighted sum:
```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

- **Wirelength**: Half-perimeter wirelength (HPWL) normalized
- **Density**: Top 10% densest grid cells (implicitly penalizes overlaps when grid density > 1.0)
- **Congestion**: Top 5% most congested routing segments

Lower is better! 🎯

### Important: Overlaps

**Q: What happens if macros overlap?**

**A:**
- The **density cost** implicitly penalizes overlaps (overlapping macros create grid cells with density > 1.0)
- The **objective can still be computed** for overlapping placements (they just score poorly)
- The **SA baseline** enforces zero overlap as a **hard constraint** via `IsFeasible()`
- Our `validate_placement()` checks for overlaps by default

**Recommendation:** Ensure your placement has zero overlaps to match the SA baseline behavior. While the objective can handle overlaps, valid competition submissions should have no overlaps.
