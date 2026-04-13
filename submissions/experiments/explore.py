# explore.py
# 
# Python script getting general information
# Includes: 
# - Macro counts (total, hard, soft)
# - Soft macro position range
# - Info about first present hard macro
# - Most connected macros (all soft)
# - Most connected hard macros
# - Macro average connection
# - Example macro connection list with positions
# - Test to see if hard-hard connections are bidirectional

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir

benchmark, plc  = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")

# Macro counts
print(f"Hard Macros: {benchmark.num_hard_macros}")
print(f"Soft Macros: {benchmark.num_soft_macros}")
print(f"Total Macros: {benchmark.num_macros}")

# Soft Macro Positions
soft_mask = benchmark.get_soft_macro_mask()
soft_positions = benchmark.macro_positions[soft_mask]
print(f"Soft macro x range: {soft_positions[:,0].min():.1f} to {soft_positions[:,0].max():.1f}")
print(f"Soft macro y range: {soft_positions[:,1].min():.1f} to {soft_positions[:,1].max():.1f}")

print("--- types in the design ---")
types = {}
for m in plc.modules_w_pins:
    t = m.get_type()
    types[t] = types.get(t, 0) + 1
print(types)

# find first MACRO and first SOFT_MACRO and inspect them
print("\n--- first HARD macro ---")
for m in plc.modules_w_pins:
    if m.get_type() == "MACRO":
        print(f"Name: {m.get_name()}")
        print(f"Connection: {m.get_connection()}")
        break

degrees = []
for m in plc.modules_w_pins:
    if m.get_type() == "MACRO":
        conn = m.get_connection()
        degrees.append((m.get_name(), len(conn), sum(conn.values())))

degrees.sort(key=lambda x: -x[1])
print("Top 10 most connected macros (name, num_connections, total_weight):")
for name, deg, weight in degrees[:10]:
    print(f"  {name}: {deg} connections, weight {weight:.1f}")

print(f"\nTotal macros with connections: {len([d for d in degrees if d[1] > 0])}")
print(f"Macros with zero connections: {len([d for d in degrees if d[1] == 0])}")
print(f"Average connections: {sum(d[1] for d in degrees)/len(degrees):.1f}")

# separate hard vs soft by name prefix as a proxy
# then verify against benchmark mask
hard_mask = benchmark.get_hard_macro_mask()
hard_indices = torch.where(hard_mask)[0].tolist()
hard_names = set(benchmark.macro_names[i] for i in hard_indices)

hard_degrees = []
soft_degrees = []

for m in plc.modules_w_pins:
    if m.get_type() != "MACRO":
        continue
    conn = m.get_connection()
    deg = len(conn)
    if m.get_name() in hard_names:
        hard_degrees.append((m.get_name(), deg, sum(conn.values()) if conn else 0))
    else:
        soft_degrees.append((m.get_name(), deg, sum(conn.values()) if conn else 0))

hard_degrees.sort(key=lambda x: -x[1])
print(f"Hard macros: {len(hard_degrees)}, Soft macros: {len(soft_degrees)}")

print("\nTop 10 most connected HARD macros:")
for name, deg, weight in hard_degrees[:10]:
    print(f"  {name}: {deg} connections, weight {weight:.1f}")

print(f"\nHard macro avg connections: {sum(d[1] for d in hard_degrees)/len(hard_degrees):.1f}")
print(f"Soft macro avg connections: {sum(d[1] for d in soft_degrees)/len(soft_degrees):.1f}")

for m in plc.modules_w_pins:
    if m.get_name() == "a12324":
        print(f"Connections: {m.get_connection()}")
        # look up positions of each connection
        pos_map = {m.get_name(): m.get_pos() for m in plc.modules_w_pins}
        for neighbor, weight in m.get_connection().items():
            pos = pos_map.get(neighbor, "unknown")
            print(f"  -> {neighbor} (weight {weight}) at {pos}")
        
    
pos_map = {m.get_name(): m for m in plc.modules_w_pins}

for m in plc.modules_w_pins:
    if m.get_name() not in hard_names:
        continue
    conn = m.get_connection()
    for neighbor_name, weight in conn.items():
        if neighbor_name in hard_names:
            # found a hard-hard connection, check the reverse
            neighbor = pos_map[neighbor_name]
            reverse_conn = neighbor.get_connection()
            reverse_weight = reverse_conn.get(m.get_name(), "MISSING")
            print(f"{m.get_name()} -> {neighbor_name}: {weight}")
            print(f"{neighbor_name} -> {m.get_name()}: {reverse_weight}")
            print()
            
one_way = 0
both_ways = 0

for m in plc.modules_w_pins:
    if m.get_name() not in hard_names:
        continue
    for neighbor_name, weight in m.get_connection().items():
        if neighbor_name in hard_names:
            neighbor = pos_map[neighbor_name]
            if m.get_name() in neighbor.get_connection():
                both_ways += 1
            else:
                one_way += 1

print(f"One-directional: {one_way}")
print(f"Both directions: {both_ways}")