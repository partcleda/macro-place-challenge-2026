"""Generate ORFS design configurations from our benchmarks."""

from pathlib import Path
from dataclasses import dataclass
import shutil


@dataclass
class ORFSDesign:
    """ORFS design configuration."""
    name: str
    tech: str  # "nangate45" or "asap7"
    verilog_files: list
    macro_placement_tcl: Path
    clock_period: float
    core_utilization: float
    top_module: str = None  # Optional: top-level module name (if different from name)


def create_orfs_design(
    design: ORFSDesign,
    orfs_root: Path,
    source_dir: Path = None
) -> Path:
    """
    Create ORFS design directory.

    Args:
        design: Design configuration
        orfs_root: Path to OpenROAD-flow-scripts/
        source_dir: Path to our benchmark source (optional)

    Returns:
        Path to created design directory
    """
    design_dir = orfs_root / "flow" / "designs" / design.tech / design.name
    design_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Verilog files
    for vfile in design.verilog_files:
        vfile_path = Path(vfile)
        if not vfile_path.exists():
            raise FileNotFoundError(f"Verilog file not found: {vfile}")
        shutil.copy(vfile, design_dir / vfile_path.name)

    # 2. Generate config.mk
    # DESIGN_HOME resolves to the designs/ root directory
    # So we need the full path: $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/file
    # DESIGN_NAME must match the top-level Verilog module name
    verilog_names = ' '.join(f"$(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/{Path(v).name}" for v in design.verilog_files)

    # Use top_module if provided, otherwise use name
    top_module = design.top_module if design.top_module else design.name

    config_mk = f"""export DESIGN_NICKNAME = {design.name}
export DESIGN_NAME = {top_module}
export PLATFORM = {design.tech}

export VERILOG_FILES = {verilog_names}
export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

# Macro placement
export MACRO_PLACEMENT_TCL = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/macros.tcl

# Floorplan
export CORE_UTILIZATION = {design.core_utilization}
export CORE_ASPECT_RATIO = 1
export CORE_MARGIN = 2

# Placement
export PLACE_DENSITY = 0.65

# Routing
export MIN_ROUTING_LAYER = metal2
export MAX_ROUTING_LAYER = metal10
"""
    (design_dir / "config.mk").write_text(config_mk)

    # 3. Generate constraint.sdc
    sdc = f"""set clk_name  core_clock
set clk_port_name clk_i
set clk_period {design.clock_period}
set clk_io_pct 0.2

create_clock -name $clk_name -period $clk_period [get_ports $clk_port_name]

set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name [all_inputs]
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]
"""
    (design_dir / "constraint.sdc").write_text(sdc)

    # 4. Copy macro placement TCL
    if design.macro_placement_tcl and design.macro_placement_tcl.exists():
        shutil.copy(design.macro_placement_tcl, design_dir / "macros.tcl")
    else:
        # Create empty macros.tcl if none provided
        (design_dir / "macros.tcl").write_text("# No macro placement\n")

    print(f"✓ Created ORFS design: {design_dir}")
    return design_dir
