"""
Write DEF (Design Exchange Format) files from PlacementCost objects.

This allows exporting placement results to industry-standard DEF format
for use with EDA tools like Innovus, ICC2, OpenROAD, etc.
"""

from typing import Optional

from macro_place._plc import PlacementCost


def write_def(plc: PlacementCost, def_file: str, design_name: Optional[str] = None):
    """
    Write placement to DEF (Design Exchange Format) file.

    Args:
        plc: PlacementCost object with placement data
        def_file: Output DEF file path
        design_name: Optional design name (defaults to extracted from plc)

    Returns:
        None (writes file to disk)
    """
    if design_name is None:
        # Try to extract design name from plc
        design_name = getattr(plc, 'design_name', 'design')

    # DEF database units (2000 = 1 micron)
    db_unit = 2000

    with open(def_file, 'w') as fp:
        # Header
        fp.write("VERSION 5.8 ;\n")
        fp.write("DIVIDERCHAR \"/\" ;\n")
        fp.write("BUSBITCHARS \"[]\" ;\n\n")
        fp.write(f"DESIGN {design_name} ;\n")
        fp.write(f"UNITS DISTANCE MICRONS {db_unit} ;\n\n")

        # Die area
        canvas_width, canvas_height = plc.get_canvas_width_height()
        llx = 0
        lly = 0
        urx = int(canvas_width * db_unit)
        ury = int(canvas_height * db_unit)
        fp.write(f"DIEAREA ( {llx} {lly} ) ( {urx} {ury} ) ;\n\n")

        # Write ROWS (for standard cell placement sites)
        _write_rows(fp, plc, db_unit)

        # Write COMPONENTS (macros and standard cells)
        _write_components(fp, plc, db_unit)

        # Write PINS (I/O ports)
        _write_pins(fp, plc, db_unit)

        # Write NETS (connectivity)
        _write_nets(fp, plc)

        # Footer
        fp.write("END DESIGN\n")

    print(f"DEF file written to: {def_file}")


def _write_rows(fp, plc: PlacementCost, db_unit: int):
    """Write ROW definitions for standard cell sites."""
    canvas_width, canvas_height = plc.get_canvas_width_height()

    # Estimate site size from grid or standard cell heights
    # Use grid as approximation
    site_height = canvas_height / plc.grid_row
    site_width = site_height  # Assume square sites for simplicity

    height_db = int(site_height * db_unit)
    width_db = int(site_width * db_unit)

    num_sites_x = plc.grid_col
    num_rows_y = plc.grid_row

    site_name = "CoreSite"

    for i in range(num_rows_y):
        row_y = int(i * height_db)
        orient = 'N' if i % 2 == 0 else 'FS'  # Alternate row orientation
        fp.write(f"ROW ROW_{i} {site_name} 0 {row_y} {orient} "
                 f"DO {num_sites_x} BY 1 STEP {width_db} 0 ;\n")

    fp.write("\n")


def _write_components(fp, plc: PlacementCost, db_unit: int):
    """Write COMPONENTS section (macros and standard cells)."""
    # Collect all placeable components (hard macros, soft macros)
    component_indices = plc.hard_macro_indices + plc.soft_macro_indices

    fp.write(f"COMPONENTS {len(component_indices)} ;\n")

    for idx in component_indices:
        node = plc.modules_w_pins[idx]

        name = node.get_name()
        x, y = node.get_pos()
        width = node.get_width()
        height = node.get_height()

        # Convert to DEF coordinates (lower-left corner, not center)
        x_ll = int((x - width / 2) * db_unit)
        y_ll = int((y - height / 2) * db_unit)

        # Get node type for ref_name
        node_type = node.get_type()
        if node_type == 'MACRO':
            ref_name = name  # Use actual macro name as ref
        else:
            ref_name = "STDCELL"  # Generic for soft macros

        # Orientation
        orient = node.get_orientation() if node.get_orientation() else "N"

        # Placement status
        is_fixed = node.get_fix_flag()
        status = "FIXED" if is_fixed else "PLACED"

        fp.write(f"  - {name} {ref_name} + {status} ( {x_ll} {y_ll} ) {orient} ;\n")

    fp.write("END COMPONENTS\n\n")


def _write_pins(fp, plc: PlacementCost, db_unit: int):
    """Write PINS section (I/O ports)."""
    pin_indices = plc.port_indices

    fp.write(f"PINS {len(pin_indices)} ;\n")

    for idx in pin_indices:
        node = plc.modules_w_pins[idx]

        name = node.get_name()
        x, y = node.get_pos()

        x_db = int(x * db_unit)
        y_db = int(y * db_unit)

        # Determine direction (heuristic: if connected to inputs, it's INPUT)
        # For simplicity, just use INPUT for now
        direction = "INPUT"

        # Determine side based on position
        canvas_width, canvas_height = plc.get_canvas_width_height()
        side = _get_pin_side(x, y, canvas_width, canvas_height)

        # Pin shape (small rectangle)
        pin_size = int(0.05 * db_unit)  # 0.05 micron

        fp.write(f"  - {name} + NET {name} + DIRECTION {direction} + USE SIGNAL\n")
        fp.write(f"      + LAYER metal1 ( -{pin_size} -{pin_size} ) ( {pin_size} {pin_size} )\n")
        fp.write(f"      + FIXED ( {x_db} {y_db} ) {side} ;\n")

    fp.write("END PINS\n\n")


def _get_pin_side(x: float, y: float, width: float, height: float) -> str:
    """Determine which side of the die a pin is on."""
    threshold = 0.01  # 1% threshold

    if abs(x) < threshold * width:
        return 'W'  # Left
    elif abs(x - width) < threshold * width:
        return 'E'  # Right
    elif abs(y) < threshold * height:
        return 'S'  # Bottom
    elif abs(y - height) < threshold * height:
        return 'N'  # Top
    else:
        return 'N'  # Default


def _write_nets(fp, plc: PlacementCost):
    """Write NETS section (connectivity).

    plc.nets is a dict: {driver_pin_name: [sink_pin_names]}.
    Pin names are either bare port names (e.g. "p3") or "instance/pin"
    (e.g. "a7419/IP1").
    """
    nets = plc.nets

    # Build set of port names for fast lookup
    port_names = set()
    for idx in plc.port_indices:
        port_names.add(plc.modules_w_pins[idx].get_name())

    fp.write(f"NETS {len(nets)} ;\n")

    for net_idx, (driver, sinks) in enumerate(nets.items()):
        net_name = f"net_{net_idx}"
        pin_connections = []

        for pin_name in [driver] + sinks:
            if pin_name in port_names:
                # I/O port
                pin_connections.append(f"( PIN {pin_name} )")
            elif "/" in pin_name:
                # Instance pin: "instance_name/pin_name"
                inst, pin = pin_name.split("/", 1)
                pin_connections.append(f"( {inst} {pin} )")

        if pin_connections:
            pins_str = " ".join(pin_connections)
            fp.write(f"  - {net_name} {pins_str} + USE SIGNAL ;\n")

    fp.write("END NETS\n\n")


# Example usage
if __name__ == "__main__":
    from macro_place.loader import load_benchmark_from_dir

    # Load a benchmark
    benchmark, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")

    # Write DEF
    write_def(plc, "ibm01_output.def", design_name="ibm01")
    print("DEF file written successfully!")
