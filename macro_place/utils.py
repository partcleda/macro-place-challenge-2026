"""
Utility functions for placement validation and visualization.
"""

import torch
from typing import Tuple, List, Optional

from macro_place.benchmark import Benchmark


def validate_placement(
    placement: torch.Tensor, benchmark: Benchmark, check_overlaps: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate placement legality.

    Checks:
    - All macros within canvas bounds
    - No NaN/Inf values
    - Correct shape
    - Fixed macros at original positions
    - No macro overlaps (optional, can be slow for large designs)

    Args:
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object
        check_overlaps: If True, check for macro-to-macro overlaps (default: True)

    Returns:
        (is_valid, violations)
    """
    violations = []

    # Check shape
    if placement.shape != (benchmark.num_macros, 2):
        violations.append(
            f"Shape mismatch: expected {(benchmark.num_macros, 2)}, got {placement.shape}"
        )
        return False, violations

    # Check for NaN/Inf
    if torch.isnan(placement).any():
        violations.append("Placement contains NaN values")
    if torch.isinf(placement).any():
        violations.append("Placement contains Inf values")

    # Check bounds
    x_coords = placement[:, 0]
    y_coords = placement[:, 1]
    widths = benchmark.macro_sizes[:, 0]
    heights = benchmark.macro_sizes[:, 1]

    x_min = x_coords - widths / 2
    x_max = x_coords + widths / 2
    y_min = y_coords - heights / 2
    y_max = y_coords + heights / 2

    if (x_min < 0).any() or (x_max > benchmark.canvas_width).any():
        violations.append("Macros outside horizontal canvas bounds")
    if (y_min < 0).any() or (y_max > benchmark.canvas_height).any():
        violations.append("Macros outside vertical canvas bounds")

    # Check fixed macros
    fixed_mask = benchmark.macro_fixed
    if fixed_mask.any():
        original_pos = benchmark.macro_positions[fixed_mask]
        new_pos = placement[fixed_mask]
        if not torch.allclose(original_pos, new_pos, atol=1e-3):
            violations.append("Fixed macros have been moved")

    # Check overlaps among hard macros only (soft macros naturally overlap)
    if check_overlaps:
        overlap_count = 0
        num_hard = getattr(benchmark, 'num_hard_macros', benchmark.num_macros)
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                # Get bounding boxes
                lx_i, ux_i = x_min[i].item(), x_max[i].item()
                ly_i, uy_i = y_min[i].item(), y_max[i].item()
                lx_j, ux_j = x_min[j].item(), x_max[j].item()
                ly_j, uy_j = y_min[j].item(), y_max[j].item()

                # Check if boxes overlap (NOT just touching)
                # No overlap if: one box is completely to the left, right, above, or below the other
                if not (lx_i >= ux_j or ux_i <= lx_j or ly_i >= uy_j or uy_i <= ly_j):
                    overlap_count += 1
                    if overlap_count <= 5:  # Only report first 5 to avoid spam
                        violations.append(f"Macros {i} and {j} overlap")

        if overlap_count > 5:
            violations.append(f"... and {overlap_count - 5} more overlaps")

    return len(violations) == 0, violations


def visualize_placement(
    placement: torch.Tensor, benchmark: Benchmark, save_path: Optional[str] = None
):
    """
    Visualize placement (requires matplotlib).

    Args:
        placement: [num_macros, 2] positions
        benchmark: Benchmark data
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw canvas
    ax.add_patch(
        Rectangle(
            (0, 0),
            benchmark.canvas_width,
            benchmark.canvas_height,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )

    # Draw macros
    for i in range(benchmark.num_macros):
        x, y = placement[i].tolist()
        w, h = benchmark.macro_sizes[i].tolist()

        x_min = x - w / 2
        y_min = y - h / 2

        color = "red" if benchmark.macro_fixed[i] else "blue"
        alpha = 0.3 if benchmark.macro_fixed[i] else 0.5

        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                w,
                h,
                fill=True,
                facecolor=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )
        )

    ax.set_xlim(0, benchmark.canvas_width)
    ax.set_ylim(0, benchmark.canvas_height)
    ax.set_aspect("equal")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.set_title(f"Placement: {benchmark.name}")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", alpha=0.5, edgecolor="black", label="Movable macros"),
        Patch(facecolor="red", alpha=0.3, edgecolor="black", label="Fixed macros"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
