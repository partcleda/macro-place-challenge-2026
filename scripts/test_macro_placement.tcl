# Test OpenROAD Macro Placement
# Simple script to verify macro placement TCL works
#
# Usage: openroad scripts/test_macro_placement.tcl

set design "ariane"
set platform "NanGate45"

# Paths
set design_dir "external/MacroPlacement/Flows/${platform}/ariane133"
set enablements_dir "external/MacroPlacement/Enablements/${platform}"

# Input files
set verilog_file "${design_dir}/netlist/ariane.v"
set sdc_file "${design_dir}/constraints/ariane.sdc"

# LEF files
set tech_lef "${enablements_dir}/lef/NangateOpenCellLibrary.tech.lef"
set macro_lef "${enablements_dir}/lef/NangateOpenCellLibrary.macro.mod.lef"
set sram_lefs [glob ${enablements_dir}/lef/fakeram45_*.lef]

# Library files (for timing)
set lib_files [glob ${enablements_dir}/lib/*.lib]

puts "========================================================================"
puts "OpenROAD Macro Placement Test"
puts "========================================================================"
puts "Design: $design"
puts "Platform: $platform"
puts "========================================================================"
puts ""

# Check input files exist
if {![file exists $verilog_file]} {
    puts "ERROR: Verilog netlist not found: $verilog_file"
    exit 1
}

# Step 1: Read LEF files
puts "\[1/5\] Reading LEF files..."
read_lef $tech_lef
read_lef $macro_lef
foreach lef $sram_lefs {
    read_lef $lef
}
puts "  ✓ LEF files loaded"

# Step 2: Read Liberty files (for timing)
puts "\[2/5\] Reading Liberty files..."
foreach lib $lib_files {
    read_liberty $lib
}
puts "  ✓ Liberty files loaded"

# Step 3: Read Verilog netlist
puts "\[3/5\] Reading Verilog netlist..."
read_verilog $verilog_file
link_design $design
puts "  ✓ Netlist loaded and linked"

# Step 4: Initialize floorplan
puts "\[4/5\] Initializing floorplan..."
# Get die area from benchmark data (ariane133: 2868.28 x 2868.28 um)
initialize_floorplan \
    -die_area {0 0 5736560 5736560} \
    -core_area {0 0 5736560 5736560} \
    -site FreePDK45_38x28_10R_NP_162NW_34O

puts "  ✓ Floorplan initialized"

# Step 5: Place macros using our generated TCL
puts "\[5/5\] Placing macros from TCL script..."

# First verify an instance exists
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]
set test_inst [$block findInst {i_cache_subsystem/i_icache/sram_block\[0\].data_sram/macro_mem\[0\].i_ram}]
if {$test_inst != "NULL"} {
    puts "  ✓ Test instance found, proceeding with placement..."
} else {
    puts "  ✗ ERROR: Test instance not found!"
    exit 1
}

source output/place_macros.tcl
puts "  ✓ Macros placed"

# Report design stats
puts ""
puts "========================================================================"
puts "Design Statistics"
puts "========================================================================"

set num_instances [llength [get_cells *]]
set num_nets [llength [get_nets *]]

puts "Instances: $num_instances"
puts "Nets: $num_nets"

# Get database
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]

# Report utilization
set die_area [[$block getDieArea] area]
puts "Die area: [expr {$die_area / 1000000.0}] um^2"

# Count placed macros
set placed_count 0
foreach inst [$block getInsts] {
    if {[$inst isPlaced]} {
        incr placed_count
    }
}
puts "Placed instances: $placed_count"

puts ""
puts "========================================================================"
puts "Test Complete!"
puts "========================================================================"
puts "✓ Macro placement TCL executed successfully"
puts "✓ $placed_count instances placed"
puts ""
puts "Next steps:"
puts "  1. Run global placement for standard cells"
puts "  2. Run detailed placement"
puts "  3. Run routing"
