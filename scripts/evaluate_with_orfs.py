#!/usr/bin/env python3
"""
Evaluate macro placements using OpenROAD-flow-scripts.

This script:
1. Loads a benchmark
2. Generates macro placement TCL
3. Creates ORFS design configuration
4. Runs ORFS flow (make)
5. Parses results

Usage:
    python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45
    python scripts/evaluate_with_orfs.py --all  # All modern benchmarks
    python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --skip-synthesis  # Skip Yosys
"""

import sys
import json
import argparse
import shutil
import subprocess
import resource
import re
import torch
from pathlib import Path

# Memory limit for ORFS subprocesses (64 GB)
MEMORY_LIMIT_BYTES = 100 * 1024 * 1024 * 1024  # 100GB for rtl_macro_placer

def _set_memory_limit():
    """Pre-exec hook: cap virtual memory for the child process tree."""
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root (for macro_place.*)
sys.path.insert(0, str(Path(__file__).parent.parent / "macro_place"))  # for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import Benchmark
from loader import load_benchmark_from_dir
from objective import compute_proxy_cost
try:
    from orfs_integration.design_generator import create_orfs_design, ORFSDesign
except ImportError:
    create_orfs_design = None  # Only needed for fallback config generation
from generate_macro_placement_tcl import write_orfs_macro_placement


def get_top_module_name(benchmark_name: str, verilog_file: Path) -> str:
    """
    Get top-level module name for a benchmark.

    For these netlists, the top module name is usually the base design name.
    """
    # Known mappings
    module_map = {
        'ariane133_ng45': 'ariane',
        'ariane136_ng45': 'ariane',
        'ariane136_asap7': 'ariane',
        'nvdla_ng45': 'NV_nvdla',
        'nvdla_asap7': 'NV_nvdla',
        'mempool_tile_ng45': 'mempool_tile',
        'mempool_tile_asap7': 'mempool_tile',
        'bp_quad_ng45': 'black_parrot',
    }

    if benchmark_name in module_map:
        return module_map[benchmark_name]

    # Fallback: use filename without extension
    return verilog_file.stem


def run_orfs_flow(design_dir: Path, orfs_root: Path, use_docker: bool = True, skip_synthesis: bool = False) -> dict:
    """
    Run ORFS flow using make (with optional Docker).

    Args:
        design_dir: Path to design directory in ORFS
        orfs_root: Path to OpenROAD-flow-scripts root
        use_docker: Use docker_shell wrapper (recommended)
        skip_synthesis: Skip Yosys synthesis (use pre-synthesized netlist)

    Returns:
        Dict with metrics
    """
    flow_dir = orfs_root / "flow"

    # Design name relative to flow/designs/{tech}/
    tech = design_dir.parent.name
    design_name = design_dir.name

    print(f"Running ORFS flow for {tech}/{design_name}...")

    # Build command with docker_shell wrapper if requested
    if use_docker:
        cmd = [
            "util/docker_shell",
            "make",
            f"DESIGN_CONFIG=./designs/{tech}/{design_name}/config.mk",
            "finish"  # Run through detailed routing
        ]
    else:
        cmd = [
            "make",
            f"DESIGN_CONFIG=./designs/{tech}/{design_name}/config.mk",
            "OPENROAD_ARGS=-threads 16",
            "finish"
        ]
        # Help ORFS find system-installed tools when not using Nix or Docker
        import shutil as _shutil
        for tool_var, tool_name in [("YOSYS_EXE", "yosys"), ("OPENROAD_EXE", "openroad")]:
            tool_path = _shutil.which(tool_name)
            if tool_path:
                cmd.append(f"{tool_var}={tool_path}")

    # Stream output to log files instead of buffering in memory
    log_dir = design_dir / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "orfs_stdout.log"
    stderr_log = log_dir / "orfs_stderr.log"

    print(f"  Logs: {stdout_log}")
    print(f"         {stderr_log}")

    with open(stdout_log, 'w') as fout, open(stderr_log, 'w') as ferr:
        try:
            result = subprocess.run(
                cmd,
                cwd=flow_dir,
                stdout=fout,
                stderr=ferr,
                timeout=43200,  # 12 hour timeout
                preexec_fn=_set_memory_limit,
            )
        except subprocess.TimeoutExpired:
            print("ERROR: ORFS timed out after 6 hours")
            return {'error': 'ORFS flow timed out'}
        except MemoryError:
            print("ERROR: ORFS hit memory limit")
            return {'error': 'ORFS flow hit memory limit'}

    # Check if final artifacts exist even if exit code was non-zero
    # (e.g. gui::show_worst_path fails headless but PnR completed)
    results_dir = flow_dir / "results" / tech / design_name / "base"
    final_artifacts = list(results_dir.glob("6_final.*")) if results_dir.exists() else []

    if result.returncode != 0 and not final_artifacts:
        print(f"ERROR: ORFS failed with return code {result.returncode}")
        # Print tail of logs
        for label, logf in [("STDOUT", stdout_log), ("STDERR", stderr_log)]:
            tail = logf.read_text()[-2000:]
            if tail.strip():
                print(f"{label} (last 2000 chars):\n{tail}")
        return {'error': f'ORFS flow failed with code {result.returncode}'}

    if result.returncode != 0:
        print(f"WARNING: ORFS exited with code {result.returncode} but final artifacts exist — parsing metrics anyway")

    # Parse results from ORFS logs and reports
    metrics = parse_orfs_results(flow_dir, tech, design_name)

    return metrics


def parse_orfs_results(flow_dir: Path, tech: str, design_name: str) -> dict:
    """
    Parse ORFS output using genMetrics.py.

    Uses ORFS's official metrics extraction tool to generate a JSON with all metrics.
    """
    import tempfile

    metrics = {}

    # ORFS uses DESIGN_NICKNAME (not dir name) for log/result paths
    nickname = design_name
    config_path = flow_dir / "designs" / tech / design_name / "config.mk"
    if config_path.exists():
        m = re.search(r'DESIGN_NICKNAME\s*=\s*(\S+)', config_path.read_text())
        if m:
            nickname = m.group(1)

    # Use ORFS genMetrics.py to extract all metrics
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        metrics_file = Path(tmp.name)

    try:
        # Run genMetrics.py (use relative paths since cwd=flow_dir)
        cmd = [
            'python3',
            'util/genMetrics.py',
            '--design', nickname,
            '--platform', tech,
            '--logs', f'logs/{tech}/{nickname}/base',
            '--reports', f'reports/{tech}/{nickname}/base',
            '--results', f'results/{tech}/{nickname}/base',
            '--output', str(metrics_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=flow_dir)

        if result.returncode == 0 and metrics_file.exists():
            with open(metrics_file) as f:
                all_metrics = json.load(f)

            # Extract key final metrics
            # Derive fmax from clock period and slack
            clock_period = 0
            clock_details = all_metrics.get('constraints__clocks__details', [])
            if clock_details:
                # Format: ['core_clock: 4.0000']
                m = re.search(r':\s*([\d.]+)', clock_details[0])
                if m:
                    clock_period = float(m.group(1))
            wns = all_metrics.get('finish__timing__setup__ws', 0)
            # fmax = 1 / (period - slack) in MHz; positive slack = timing met
            period_min = clock_period - wns if clock_period > 0 else 0
            fmax = 1000.0 / period_min if period_min > 0 else 0

            metrics = {
                'tns': all_metrics.get('finish__timing__setup__tns', 0),
                'wns': wns,
                'hold_tns': all_metrics.get('finish__timing__hold__tns', 0),
                'hold_wns': all_metrics.get('finish__timing__hold__ws', 0),
                'wire_length': all_metrics.get('detailedroute__route__wirelength', 0),
                'area': all_metrics.get('finish__design__core__area', 0),
                'power': all_metrics.get('finish__power__total', 0),
                'fmax': round(fmax, 2),
                'clock_period': clock_period,
            }
        else:
            print(f"Warning: genMetrics.py failed: {result.stderr}")

    finally:
        # Clean up temp file
        if metrics_file.exists():
            metrics_file.unlink()

    return metrics


def evaluate_benchmark(
    benchmark_name: str,
    orfs_root: Path,
    output_dir: Path,
    use_docker: bool = True,
    skip_synthesis: bool = False,
    placement_path: Path = None
) -> dict:
    """Evaluate a single benchmark."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {benchmark_name}")
    print(f"{'='*80}")

    # Load benchmark
    pt_file = Path(f"benchmarks/processed/public/{benchmark_name}.pt")
    if not pt_file.exists():
        print(f"ERROR: {pt_file} not found")
        return {'error': 'benchmark not found', 'benchmark': benchmark_name}

    benchmark = Benchmark.load(str(pt_file))
    print(f"✓ Loaded benchmark: {benchmark.num_macros} macros")

    # Resolve source paths
    tech = "nangate45" if "ng45" in benchmark_name else "asap7"
    source_name = benchmark_name.replace("_ng45", "").replace("_asap7", "")

    # Map benchmark names to protobuf source directories
    source_dir_overrides = {
        'bp_quad': Path("external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW/test/bp_ng45"),
    }

    if source_name in source_dir_overrides:
        source_dir = source_dir_overrides[source_name]
    elif tech == "nangate45":
        source_dir = Path(f"external/MacroPlacement/Flows/NanGate45/{source_name}/netlist/output_CT_Grouping")
    else:
        source_dir = Path(f"external/MacroPlacement/Flows/ASAP7/{source_name}/netlist/output_CT_Grouping")

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return {'error': 'source directory not found', 'benchmark': benchmark_name}

    _, plc = load_benchmark_from_dir(str(source_dir))

    # Load placement: use provided tensor or fall back to benchmark default
    if placement_path is not None:
        placement = torch.load(placement_path, weights_only=True)
        print(f"✓ Loaded placement from {placement_path} (shape: {list(placement.shape)})")
    else:
        placement = benchmark.macro_positions

    # 1. Compute proxy cost
    print("\n[1/4] Computing proxy cost...")
    proxy_metrics = compute_proxy_cost(placement, benchmark, plc)
    print(f"  ✓ Proxy cost: {proxy_metrics['proxy_cost']:.6f}")

    # 2. Generate macro placement TCL (will be regenerated with core_area clamping below)
    print("\n[2/4] Generating macro placement TCL...")
    tcl_file = output_dir / f"{benchmark_name}_macros.tcl"

    # 3. Check for existing ORFS configuration
    print("\n[3/4] Looking for existing ORFS configuration...")

    # Path to their OpenROAD scripts directory
    if tech == "nangate45":
        orfs_config_dir = Path(f"external/MacroPlacement/Flows/NanGate45/{source_name}/scripts/OpenROAD/{source_name}")
    else:
        orfs_config_dir = Path(f"external/MacroPlacement/Flows/ASAP7/{source_name}/scripts/OpenROAD/{source_name}")

    # Extract .tar.gz configs (ariane136, mempool_tile ship as tarballs)
    if not orfs_config_dir.exists():
        import tarfile
        tar_parent = orfs_config_dir.parent
        for tar_path in tar_parent.glob("*.tar.gz") if tar_parent.exists() else []:
            print(f"  Extracting {tar_path.name}...")
            with tarfile.open(tar_path) as tar:
                # Find config.mk inside the tar (may be nested)
                config_members = [m for m in tar.getmembers() if m.name.endswith("config.mk")]
                if config_members:
                    orfs_config_dir.mkdir(parents=True, exist_ok=True)
                    # Extract all files, stripping the leading path to get just filenames
                    for member in tar.getmembers():
                        if member.isfile():
                            member_name = Path(member.name).name
                            member.name = member_name
                            tar.extract(member, orfs_config_dir)
                    print(f"  ✓ Extracted {len(tar.getmembers())} files to {orfs_config_dir}")
                    break

    # Generate ORFS config for nvdla if it doesn't exist (no upstream collateral)
    if not orfs_config_dir.exists() and source_name == "nvdla":
        orfs_config_dir.mkdir(parents=True, exist_ok=True)
        enable_dir = Path("external/MacroPlacement/Enablements/NanGate45")
        netlist_dir = Path(f"external/MacroPlacement/Flows/NanGate45/nvdla/netlist")

        # Copy Genus netlist and fakeram files
        shutil.copy(netlist_dir / "NV_NVDLA_partition_c.v", orfs_config_dir)
        shutil.copy(enable_dir / "lef" / "fakeram45_256x64.lef", orfs_config_dir)
        shutil.copy(enable_dir / "lib" / "fakeram45_256x64.lib", orfs_config_dir)

        # Write config.mk
        (orfs_config_dir / "config.mk").write_text("""\
export DESIGN_NICKNAME = nvdla
export DESIGN_NAME = NV_nvdla
export PLATFORM    = nangate45

export VERILOG_FILES = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/NV_NVDLA_partition_c.v

export SDC_FILE      = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc
export ABC_CLOCK_PERIOD_IN_PS = 2000

export ADDITIONAL_LEFS = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/fakeram45_256x64.lef
export ADDITIONAL_LIBS = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/fakeram45_256x64.lib

export DIE_AREA    = 0.0 0.0 3200.00 3200.00
export CORE_AREA   = 10.07 9.94 3189.93 3190.06
export PLACE_PINS_ARGS = -exclude left:0-400 -exclude left:2800-3200 \\
                         -exclude right:0-400 -exclude right:2800-3200 \\
                         -exclude top:0-400 -exclude top:2800-3200 \\
                         -exclude bottom:0-400 -exclude bottom:2800-3200

export PLACE_DENSITY_LB_ADDON ?= 0.10
""")
        # Write constraint.sdc (4ns clock matching other NG45 benchmarks)
        (orfs_config_dir / "constraint.sdc").write_text("""\
create_clock [get_ports nvdla_core_clk]  -name core_clock  -period 4
set_input_delay -clock core_clock 0 [all_inputs]
set_output_delay -clock core_clock 0 [all_outputs]
""")
        print(f"  ✓ Generated ORFS config for nvdla (128 macros, fakeram45_256x64)")

    # Fallback: check ORFS built-in designs (maps source_name to ORFS design name)
    orfs_builtin_map = {
        'bp_quad': 'black_parrot',
    }
    if not orfs_config_dir.exists() and source_name in orfs_builtin_map:
        orfs_design_name_builtin = orfs_builtin_map[source_name]
        builtin_dir = orfs_root / "flow" / "designs" / tech / orfs_design_name_builtin
        if builtin_dir.exists():
            orfs_config_dir = builtin_dir
            # Use the ORFS design name for consistency
            source_name = orfs_design_name_builtin

    if orfs_config_dir.exists():
        print(f"  ✓ Found existing ORFS config: {orfs_config_dir}")

        # Use their original design name to keep paths consistent
        design_dir = orfs_root / "flow" / "designs" / tech / source_name
        if design_dir.resolve() != orfs_config_dir.resolve():
            # Copy from external config into ORFS
            if design_dir.exists():
                shutil.rmtree(design_dir)
            shutil.copytree(orfs_config_dir, design_dir)
        # else: config is already an ORFS built-in design, use in-place

        # For ASAP7, copy SRAM libraries from MacroPlacement/Enablements
        if tech == "asap7":
            asap7_enablements = Path("external/MacroPlacement/Enablements/ASAP7")
            if asap7_enablements.exists():
                # Copy SRAM LEF files
                sram_lefs = list((asap7_enablements / "lef").glob("sram_*.lef"))
                for lef in sram_lefs:
                    shutil.copy(lef, design_dir / lef.name)

                # Copy SRAM LIB files
                sram_libs = list((asap7_enablements / "lib").glob("sram_*.lib"))
                for lib in sram_libs:
                    shutil.copy(lib, design_dir / lib.name)

                print(f"  ✓ Copied {len(sram_lefs)} SRAM LEF and {len(sram_libs)} LIB files from Enablements")

        # If skip_synthesis is enabled, modify config.mk to use pre-synthesized netlist
        if skip_synthesis:
            config_mk = design_dir / "config.mk"
            with open(config_mk, 'a') as f:
                f.write("\n# Skip synthesis - use pre-synthesized netlist\n")
                f.write("export SYNTH_NETLIST_FILES = $(VERILOG_FILES)\n")
            print(f"  ✓ Added SYNTH_NETLIST_FILES to skip synthesis")

        # Fix benchmark-specific config issues
        config_mk = design_dir / "config.mk"
        if config_mk.exists():
            config_content = config_mk.read_text()

            # Use pre-mapped Genus gate netlist when available (bypasses Yosys).
            # This fixes ariane133 where PRESERVE_CELLS causes Yosys to drop
            # 89 of 133 SRAMs (see issue #50).
            _using_genus_netlist = False
            genus_netlist_dir = Path(
                f"external/MacroPlacement/Flows/NanGate45/{source_name}/netlist"
            )
            if genus_netlist_dir.exists():
                for candidate in sorted(genus_netlist_dir.glob("*.v")):
                    with open(candidate) as fv:
                        n_sram = sum(1 for line in fv if "fakeram45_" in line)
                    if n_sram > 0:
                        # Copy Genus netlist to design dir so we can patch it
                        patched_netlist = design_dir / "genus_netlist.v"
                        shutil.copy(candidate, patched_netlist)

                        # Fix Genus netlist syntax: join split "module\n  name" declarations
                        # OpenROAD's Verilog reader can't handle module name on next line
                        genus_raw = patched_netlist.read_text()
                        genus_raw = re.sub(r'^module\s*\n\s+', 'module ', genus_raw, flags=re.MULTILINE)
                        patched_netlist.write_text(genus_raw)

                        # Patch: add missing gate-level module definitions (issue #65).
                        # Genus netlist references lzc_MODE1_WIDTH64, lzc_WIDTH3, lzc_WIDTH4
                        # but their definitions were stripped. Use pre-synthesized gate-level patches.
                        lzc_patch_file = Path(__file__).parent / "ariane133_lzc_patches.v"
                        if lzc_patch_file.exists():
                            genus_text = patched_netlist.read_text()
                            patch_text = lzc_patch_file.read_text()
                            # Only append modules that are referenced but not defined
                            needed = [m for m in ['lzc_MODE1_WIDTH64', 'lzc_WIDTH3', 'lzc_WIDTH4']
                                      if m in genus_text and f"module {m}" not in genus_text]
                            if needed:
                                with open(patched_netlist, 'a') as pf:
                                    pf.write(f"\n// --- Gate-level patches for {', '.join(needed)} ---\n")
                                    pf.write(patch_text)
                                print(f"  ✓ Patched {len(needed)} missing modules: {', '.join(needed)}")

                        config_content += (
                            f"\n# Override: use patched Genus gate netlist ({n_sram} SRAMs)\n"
                            f"export SYNTH_NETLIST_FILES = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/genus_netlist.v\n"
                        )
                        # Pre-place the patched Genus netlist where ORFS expects 1_2_yosys.v.
                        # This bypasses Yosys entirely — the Makefile sees the output
                        # already exists and skips the canonicalize + synthesis steps.
                        yosys_out = orfs_root / "flow" / "results" / tech / source_name / "base"
                        yosys_out.mkdir(parents=True, exist_ok=True)
                        shutil.copy(patched_netlist, yosys_out / "1_2_yosys.v")
                        _using_genus_netlist = True
                        print(f"  ✓ Using Genus gate netlist: {candidate.name} ({n_sram} SRAMs)")
                        break

            if source_name == "mempool_tile":
                # 1. Disable hierarchical flow
                config_content = re.sub(
                    r'export FLOW_VARIANT = hier',
                    '# export FLOW_VARIANT = hier  # Disabled for flat flow',
                    config_content
                )
                config_content = re.sub(
                    r'export SYNTH_HIERARCHICAL = 1',
                    '# export SYNTH_HIERARCHICAL = 1  # Disabled for flat flow',
                    config_content
                )
                config_content = re.sub(
                    r'export RTLMP_FLOW = True',
                    '# export RTLMP_FLOW = True  # Disabled for flat flow',
                    config_content
                )
                # 2. Remove FLOORPLAN_DEF (conflicts with DIE_AREA/CORE_AREA)
                config_content = re.sub(
                    r'^(export FLOORPLAN_DEF\s*=.*)$',
                    r'# \1  # Disabled: conflicts with DIE_AREA/CORE_AREA',
                    config_content,
                    flags=re.MULTILINE
                )
                # 3. Increase die size to 2000x2000 for 1272 IO pins
                config_content = re.sub(
                    r'export DIE_AREA\s*=\s*0\.0 0\.0 1000 1000',
                    'export DIE_AREA    = 0.0 0.0 2000 2000  # Increased for 1272 IO pins',
                    config_content
                )
                config_content = re.sub(
                    r'export CORE_AREA\s*=\s*10\.07 9\.94 990 990',
                    'export CORE_AREA   = 10.07 9.94 1990 1990  # Increased with DIE_AREA',
                    config_content
                )
                # 4. Open all 4 die sides for pin placement with small corner exclusions
                config_content = re.sub(
                    r'export PLACE_PINS_ARGS\s*=.*',
                    'export PLACE_PINS_ARGS = -exclude left:0-200 -exclude left:1800-2000 '
                    '-exclude right:0-200 -exclude right:1800-2000 '
                    '-exclude top:0-200 -exclude top:1800-2000 '
                    '-exclude bottom:0-200 -exclude bottom:1800-2000',
                    config_content
                )
                # 5. Reduce placement density addon (die is 4x larger)
                config_content = re.sub(
                    r'export PLACE_DENSITY_LB_ADDON\s*=\s*0\.20',
                    'export PLACE_DENSITY_LB_ADDON = 0.05  # Reduced: 4x larger die area',
                    config_content
                )
                print(f"  ✓ Fixed mempool_tile config (disabled hierarchical flow, increased die to 2000x2000, opened all pin sides)")

            if source_name in ("ariane133", "ariane136"):
                # Reduce macro halo so macros fit (default 22.4x15.12 is too large for 133+ macros)
                if 'MACRO_PLACE_HALO' not in config_content:
                    config_content += '\nexport MACRO_PLACE_HALO = 5.0 5.0\n'
                else:
                    config_content = re.sub(
                        r'export MACRO_PLACE_HALO\s*=.*',
                        'export MACRO_PLACE_HALO = 5.0 5.0',
                        config_content
                    )
                print(f"  ✓ Reduced {source_name} MACRO_PLACE_HALO to 5.0 5.0")

            if source_name == "black_parrot":
                # Disable hierarchical synthesis — we use our own macro placement
                config_content = re.sub(
                    r'export SYNTH_HIERARCHICAL = 1',
                    '# export SYNTH_HIERARCHICAL = 1  # Disabled: using our macro placement',
                    config_content
                )
                print(f"  ✓ Disabled hierarchical synthesis for black_parrot")

            # Fix ASAP7 SRAM library paths to use local copies
            if tech == "asap7":
                # Replace PLATFORM_DIR references with local paths
                config_content = re.sub(
                    r'\$\(PLATFORM_DIR\)/lef/(sram_[^)]+\.lef)',
                    r'./designs/asap7/' + source_name + r'/\1',
                    config_content
                )
                config_content = re.sub(
                    r'\$\(PLATFORM_DIR\)/lib/(sram_[^)]+\.lib)',
                    r'./designs/asap7/' + source_name + r'/\1',
                    config_content
                )
                print(f"  ✓ Fixed ASAP7 config to use local SRAM libraries")

            # Add MACRO_PLACEMENT_TCL for ALL designs so ORFS uses our placement
            if 'MACRO_PLACEMENT_TCL' not in config_content:
                config_content += '\nexport MACRO_PLACEMENT_TCL = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/macros.tcl\n'

            # Fix Genus netlist issue: constant-1 nets typed as POWER can't be routed.
            # Reclassify them as SIGNAL before global routing.
            if _using_genus_netlist:
                fix_tcl = design_dir / "fix_power_nets.tcl"
                fix_tcl.write_text(
                    "# Reclassify constant nets mistyped as POWER/GROUND\n"
                    "set block [ord::get_db_block]\n"
                    "foreach net [$block getNets] {\n"
                    "  set type [$net getSigType]\n"
                    "  set name [$net getName]\n"
                    "  if { ($type eq \"POWER\" || $type eq \"GROUND\") && $name ni {VDD VSS} } {\n"
                    "    $net setSigType SIGNAL\n"
                    "    puts \"Reclassified net $name from $type to SIGNAL\"\n"
                    "  }\n"
                    "}\n"
                )
                config_content += f'\nexport PRE_GLOBAL_ROUTE_TCL = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/fix_power_nets.tcl\n'
                print(f"  ✓ Added PRE_GLOBAL_ROUTE_TCL to fix Genus power net typing")

            # Workaround: repair_timing -sequence is not supported in older OpenROAD builds.
            # Set REMOVE_ABC_BUFFERS=1 so floorplan.tcl takes the remove_buffers path
            # instead of calling repair_timing_helper with -sequence.
            if 'REMOVE_ABC_BUFFERS' not in config_content:
                config_content += '\nexport REMOVE_ABC_BUFFERS = 1\n'

            config_mk.write_text(config_content)

        # Patch ORFS macro_place_util.tcl to skip rtl_macro_placer when
        # MACRO_PLACEMENT_TCL is set (our pre-computed placement).
        # rtl_macro_placer crashes on already-placed macros in some OpenROAD versions.
        mp_util = orfs_root / "flow" / "scripts" / "macro_place_util.tcl"
        mp_util_text = mp_util.read_text()
        if 'SKIP_RTLMP' not in mp_util_text:
            mp_util_text = mp_util_text.replace(
                'log_cmd rtl_macro_placer {*}$all_args',
                'if { [env_var_exists_and_non_empty SKIP_RTLMP] } {\n'
                '    puts "Skipping rtl_macro_placer (SKIP_RTLMP set)"\n'
                '  } else {\n'
                '    log_cmd rtl_macro_placer {*}$all_args\n'
                '  }'
            )
            mp_util.write_text(mp_util_text)
            print(f"  ✓ Patched macro_place_util.tcl to support SKIP_RTLMP")

        # Parse CORE_AREA from config.mk
        core_area = None
        config_mk = design_dir / "config.mk"
        config_text = config_mk.read_text()
        m = re.search(r'CORE_AREA\s*=\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', config_text)
        if m:
            core_area = tuple(float(x) for x in m.groups())
            print(f"  ✓ Parsed CORE_AREA: {core_area}")

        # Set SKIP_RTLMP in config (only when we're providing our own placement)
        if placement_path is not None and 'SKIP_RTLMP' not in config_text:
            config_text += '\nexport SKIP_RTLMP = 1\n'
            config_mk.write_text(config_text)
            print(f"  ✓ Set SKIP_RTLMP=1 in config")
        elif placement_path is None:
            # No custom placement — let ORFS's rtl_macro_placer handle it (baseline mode)
            # Remove MACRO_PLACEMENT_TCL so ORFS does its own placement
            config_text = re.sub(r'\nexport MACRO_PLACEMENT_TCL\s*=.*\n', '\n', config_text)
            # Set RTLMP fence to core area so rtl_macro_placer knows the bounds
            if core_area and 'RTLMP_FENCE' not in config_text:
                config_text += (
                    f'\n# rtl_macro_placer fence bounds (= CORE_AREA)\n'
                    f'export RTLMP_FENCE_LX = {core_area[0]}\n'
                    f'export RTLMP_FENCE_LY = {core_area[1]}\n'
                    f'export RTLMP_FENCE_UX = {core_area[2]}\n'
                    f'export RTLMP_FENCE_UY = {core_area[3]}\n'
                )
            config_mk.write_text(config_text)
            print(f"  ✓ Baseline mode: letting ORFS rtl_macro_placer handle placement")

        # Regenerate TCL with core_area clamping
        write_orfs_macro_placement(placement, benchmark, plc, str(tcl_file), core_area=core_area, use_genus_names=_using_genus_netlist)
        shutil.copy(tcl_file, design_dir / "macros.tcl")
        # Also overwrite any existing macro placement TCL referenced in config
        tcl_ref = re.search(r'MACRO_PLACEMENT_TCL\s*=.*?/([^/\s]+\.tcl)', config_text)
        if tcl_ref and tcl_ref.group(1) != "macros.tcl":
            shutil.copy(tcl_file, design_dir / tcl_ref.group(1))
            print(f"  ✓ Also overwrote {tcl_ref.group(1)} with our placement")

        print(f"  ✓ Copied config to: {design_dir}")
        print(f"  ✓ Using original design name: {source_name}")
        print(f"  ✓ Using our macro placement: {tcl_file.name}")
    else:
        print(f"  ⚠️  No existing config found at {orfs_config_dir}")
        print(f"  Generating basic config (may not work)")

        # Fallback to generated config
        verilog_files = list(source_dir.glob("*.v"))
        if not verilog_files:
            parent_netlist = source_dir.parent
            verilog_files = list(parent_netlist.glob("*.v"))

        if not verilog_files:
            return {'error': 'no verilog files', 'benchmark': benchmark_name}

        # Generate TCL without core_area clamping (fallback path)
        write_orfs_macro_placement(placement, benchmark, plc, str(tcl_file))

        top_module = get_top_module_name(benchmark_name, verilog_files[0])
        design = ORFSDesign(
            name=benchmark_name,
            tech=tech,
            verilog_files=verilog_files,
            macro_placement_tcl=tcl_file,
            clock_period=4.0,  # Match their 4ns
            core_utilization=0.65,
            top_module=top_module
        )
        design_dir = create_orfs_design(design, orfs_root, source_dir)

    # 4. Run ORFS flow
    print("\n[4/4] Running OpenROAD-flow-scripts...")
    print("  (This may take 20-40 minutes per benchmark)")

    # Use source_name for the ORFS design if we copied their config
    if orfs_config_dir.exists():
        # Update config to point to correct design
        orfs_design_name = source_name
    else:
        orfs_design_name = benchmark_name

    # Clean stale ORFS results/logs so changed config (e.g. DIE_AREA) takes effect
    # Check both the design directory name and the DESIGN_NICKNAME
    nickname = orfs_design_name
    config_path = design_dir / "config.mk"
    if config_path.exists():
        m = re.search(r'DESIGN_NICKNAME\s*=\s*(\S+)', config_path.read_text())
        if m:
            nickname = m.group(1)
    stale_names = {orfs_design_name, nickname} if orfs_config_dir.exists() else {benchmark_name}
    for subdir in ["results", "logs", "objects"]:
        for sname in stale_names:
            stale = orfs_root / "flow" / subdir / tech / sname
            if stale.exists():
                shutil.rmtree(stale)
                print(f"  ✓ Cleaned stale {subdir}/{tech}/{stale.name}")

    orfs_metrics = run_orfs_flow(design_dir, orfs_root, use_docker, skip_synthesis)

    # 5. Combine results
    results = {
        'benchmark': benchmark_name,
        'num_macros': int(benchmark.num_macros),
        'proxy_cost': float(proxy_metrics['proxy_cost']),
        'wirelength': float(proxy_metrics['wirelength_cost']),
        'density': float(proxy_metrics['density_cost']),
        'congestion': float(proxy_metrics['congestion_cost']),
        'orfs': orfs_metrics
    }

    print(f"\n✓ Evaluation complete for {benchmark_name}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmarks with ORFS')
    parser.add_argument('--benchmark', type=str, help='Single benchmark')
    parser.add_argument('--all', action='store_true', help='All modern benchmarks')
    parser.add_argument('--orfs-root', type=Path,
                       default=Path("../OpenROAD-flow-scripts"),
                       help='Path to OpenROAD-flow-scripts')
    parser.add_argument('--output', type=Path,
                       default=Path("output/orfs_evaluation"),
                       help='Output directory')
    parser.add_argument('--no-docker', action='store_true',
                       help='Run without Docker (use native ORFS installation)')
    parser.add_argument('--skip-synthesis', action='store_true',
                       help='Skip Yosys synthesis (use pre-synthesized netlist)')
    parser.add_argument('--placement', type=Path,
                       help='Path to placement tensor (.pt file) with shape [num_macros, 2]')

    args = parser.parse_args()

    # Verify ORFS exists
    if not args.orfs_root.exists():
        print(f"ERROR: OpenROAD-flow-scripts not found at {args.orfs_root}")
        print("\nTo set up ORFS:")
        print("  cd ..")
        print("  git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts")
        return 1

    # Discover benchmarks
    if args.all:
        benchmarks = [
            'ariane133_ng45', 'ariane136_ng45', 'bp_quad_ng45', 'nvdla_ng45', 'mempool_tile_ng45',
            'ariane136_asap7', 'nvdla_asap7', 'mempool_tile_asap7'
        ]
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        print("ERROR: Specify --benchmark or --all")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # Evaluate all
    all_results = []
    for name in benchmarks:
        result = evaluate_benchmark(
            name,
            args.orfs_root,
            args.output,
            use_docker=not args.no_docker,
            skip_synthesis=args.skip_synthesis,
            placement_path=args.placement
        )
        all_results.append(result)

        # Save incremental results
        summary_file = args.output / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"Results: {args.output / 'evaluation_summary.json'}")
    print(f"{'='*80}")

    # Print table
    print(f"\n{'Benchmark':<25} {'Proxy Cost':<15} {'WNS (ns)':<12} {'TNS (ns)':<12} {'Fmax (MHz)':<12} {'Wire (um)':<12} {'Area (um²)':<15}")
    print("-" * 115)

    for result in all_results:
        orfs = result.get('orfs', {})
        wns = orfs.get('wns', 'N/A')
        tns = orfs.get('tns', 'N/A')
        fmax = orfs.get('fmax', 'N/A')
        wire_length = orfs.get('wire_length', 'N/A')
        area = orfs.get('area', 'N/A')

        wns_str = f"{wns}" if isinstance(wns, str) else f"{wns:.2f}"
        tns_str = f"{tns}" if isinstance(tns, str) else f"{tns:.2f}"
        fmax_str = f"{fmax / 1e6:.1f}" if isinstance(fmax, (int, float)) else "N/A"
        wire_str = f"{wire_length / 1e6:.2f}" if isinstance(wire_length, (int, float)) else "N/A"
        area_str = f"{area / 1e6:.3f}" if isinstance(area, (int, float)) else "N/A"

        print(f"{result['benchmark']:<25} "
              f"{result['proxy_cost']:<15.6f} "
              f"{wns_str:<12} "
              f"{tns_str:<12} "
              f"{fmax_str:<12} "
              f"{wire_str:<12} "
              f"{area_str:<15}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
