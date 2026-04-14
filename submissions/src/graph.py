import torch
import numpy as np
from pathlib import Path
from macro_place.loader import load_benchmark, load_benchmark_from_dir


def load_plc(name):
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        benchmark,plc = load_benchmark_from_dir(str(root))
        return benchmark, plc
    return None

def build_weight_graph(benchmark, plc):
    hard_mask = benchmark.get_hard_macro_mask()
    hard_indices = torch.where(hard_mask)[0].tolist()
    hard_names = set(benchmark.macro_names[i] for i in hard_indices)
    
    adj = {}
    for name in hard_names:
        adj[name] = {}
    
    for m in plc.modules_w_pins:
        if m.get_type() == "MACRO":
            if m.get_name() in hard_names:
                name = m.get_name()
                conn = m.get_connection()
                for neighbor_name, weight in conn.items():
                    adj[name][neighbor_name] = weight
                    if neighbor_name in hard_names:
                        adj[neighbor_name][name] = weight
                    
    return adj
