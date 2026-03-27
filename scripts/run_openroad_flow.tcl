# Simple OpenROAD P&R Flow
# Takes a DEF with macro placement and runs full physical design flow
#
# Usage: openroad scripts/run_openroad_flow.tcl
#
# This script:
# 1. Reads LEF files (technology + macros)
# 2. Reads netlist (Verilog)
# 3. Reads DEF with macro placement
# 4. Places standard cells
# 5. Adds clock tree
# 6. Routes the design
# 7. Reports QoR metrics

set design "ariane"
set platform "NanGate45"

# Paths
set design_dir "external/MacroPlacement/Flows/${platform}/ariane133"
set enablements_dir "external/MacroPlacement/Enablements/${platform}"
set output_dir "output/openroad_results"

# Input files
set verilog_file "${design_dir}/netlist/ariane.v"
set sdc_file "${design_dir}/constraints/ariane.sdc"
set def_file "output/output_CT_Grouping_random.def"  ;# Our generated DEF

# LEF files
set tech_lef "${enablements_dir}/lef/NangateOpenCellLibrary.tech.lef"
set macro_lef "${enablements_dir}/lef/NangateOpenCellLibrary.macro.mod.lef"
set sram_lefs [glob ${enablements_dir}/lef/fakeram45_*.lef]

# Library files (for timing)
set lib_files [glob ${enablements_dir}/lib/*.lib]

# Create output directory
file mkdir $output_dir

puts "========================================================================"
puts "OpenROAD Physical Design Flow"
puts "========================================================================"
puts "Design: $design"
puts "Platform: $platform"
puts "Input DEF: $def_file"
puts "Output: $output_dir"
puts "========================================================================"
puts ""

# Check input files exist
if {![file exists $verilog_file]} {
    puts "ERROR: Verilog netlist not found: $verilog_file"
    exit 1
}

if {![file exists $def_file]} {
    puts "ERROR: DEF file not found: $def_file"
    puts "Run: python scripts/demo_placement_to_def.py first"
    exit 1
}

# Step 1: Read LEF files
puts "\[1/8\] Reading LEF files..."
read_lef $tech_lef
read_lef $macro_lef
foreach lef $sram_lefs {
    read_lef $lef
}
puts "  ✓ LEF files loaded"

# Step 2: Read DEF with macro placement (must be done before verilog)
puts "\[2/8\] Reading DEF with macro placement..."
read_def $def_file
puts "  ✓ DEF loaded (die area and macros placed)"

# Step 3: Read Liberty files (for timing)
puts "\[3/8\] Reading Liberty files..."
foreach lib $lib_files {
    read_liberty $lib
}
puts "  ✓ Liberty files loaded"

# Step 4: Read Verilog netlist to populate standard cells
puts "\[4/8\] Reading Verilog netlist..."
read_verilog $verilog_file
link_design $design
puts "  ✓ Netlist loaded and linked"

# Step 5: Read timing constraints
puts "\[5/8\] Reading timing constraints..."
read_sdc $sdc_file
puts "  ✓ SDC loaded"

# Step 6: Global placement (standard cells only, macros already placed)
puts "\[6/8\] Running global placement..."
global_placement -skip_initial_place
puts "  ✓ Global placement complete"

# Step 7: Detailed placement
puts "\[7/8\] Running detailed placement..."
detailed_placement
puts "  ✓ Detailed placement complete"

# Step 8: Write output DEF
puts "\[8/8\] Writing results..."
set output_def "${output_dir}/${design}_placed.def"
write_def $output_def
puts "  ✓ Output DEF: $output_def"

# Report design stats
puts ""
puts "========================================================================"
puts "Design Statistics"
puts "========================================================================"

set num_instances [llength [get_cells *]]
set num_nets [llength [get_nets *]]
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]

puts "Instances: $num_instances"
puts "Nets: $num_nets"

# Report utilization
set die_area [[$block getDieArea] area]
set core_area [[lindex [$block getCoreArea] 0] area]
puts "Die area: [expr {$die_area / 1000000.0}] um^2"
puts "Core area: [expr {$core_area / 1000000.0}] um^2"

# Report timing (if possible)
puts ""
puts "========================================================================"
puts "Timing Analysis"
puts "========================================================================"

# Create clock
set clock_period 1.3
create_clock -period $clock_period [get_ports clk_i]

# Report timing
report_checks -path_delay max -format full_clock_expanded

puts ""
puts "========================================================================"
puts "Flow Complete!"
puts "========================================================================"
puts "Output DEF: $output_def"
puts ""
puts "Next steps:"
puts "  1. View result: openroad -gui $output_def"
puts "  2. Run routing: (add routing steps above)"
puts "  3. Extract metrics: (add reporting steps)"
