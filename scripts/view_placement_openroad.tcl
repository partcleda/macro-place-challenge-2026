# OpenROAD Visualization Script
# Usage: openroad -gui scripts/view_placement_openroad.tcl
#
# This script loads the generated DEF file and visualizes the placement

# Set paths
set lef_tech "external/MacroPlacement/CodeElements/Clustering/test/lefs/NangateOpenCellLibrary.tech.lef"
set lef_macro "external/MacroPlacement/CodeElements/Clustering/test/lefs/NangateOpenCellLibrary.macro.mod.lef"
set def_file "output/output_CT_Grouping_random.def"

# Check if files exist
if {![file exists $lef_tech]} {
    puts "ERROR: Technology LEF not found: $lef_tech"
    exit 1
}

if {![file exists $lef_macro]} {
    puts "ERROR: Macro LEF not found: $lef_macro"
    exit 1
}

if {![file exists $def_file]} {
    puts "ERROR: DEF file not found: $def_file"
    puts "Run: python scripts/demo_placement_to_def.py first"
    exit 1
}

puts "================================"
puts "OpenROAD Placement Visualization"
puts "================================"
puts ""

# Load technology and macro LEFs
puts "Loading LEF files..."
read_lef $lef_tech
read_lef $lef_macro
puts "  ✓ LEF files loaded"

# Load DEF
puts "Loading DEF file: $def_file"
read_def $def_file
puts "  ✓ DEF loaded"

# Get design stats
set num_instances [llength [get_cells *]]
puts ""
puts "Design Statistics:"
puts "  - Instances: $num_instances"

# Fit design in window
if {[info exists gui::gui_mode]} {
    puts ""
    puts "Starting GUI..."
    gui::fit

    # Optional: Color by type
    # gui::selection_add [get_cells i_cache*]

    puts ""
    puts "GUI Controls:"
    puts "  - Scroll: Zoom in/out"
    puts "  - Click: Select cells"
    puts "  - Right-click: Context menu"
    puts "  - F: Fit design in window"
    puts ""
    puts "Design loaded successfully!"
}
