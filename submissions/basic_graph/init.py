import torch
import numpy as np
from macro_place.benchmark import Benchmark

def compute_initial_placement(benchmark, adj, pos_map, seed=42):
    # start from existing positions (keeps soft macros in place)
    placement = benchmark.macro_positions.clone()
    rng = np.random.default_rng(seed)
    
    
    hard_mask = benchmark.get_hard_macro_mask()
    hard_indices = torch.where(hard_mask)[0].tolist()
    
    targets = {}
    for i in hard_indices:
        name = benchmark.macro_names[i]
        neighbors = adj[name]
        
        if len(neighbors) == 0:
            # truly isolated
            half_w = benchmark.macro_sizes[i, 0].item() / 2
            half_h = benchmark.macro_sizes[i, 1].item() / 2
            targets[name] = (
                rng.uniform(0, benchmark.canvas_width),
                rng.uniform(0, benchmark.canvas_height)
            )
        else:
            # compute weighted average position (based on pos_map)
            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            for neighbor_name, weight in neighbors.items():
                neighbor = pos_map[neighbor_name]
                weighted_x += weight * neighbor.x
                weighted_y += weight * neighbor.y
                total_weight += weight
                
            half_w = benchmark.macro_sizes[i, 0].item() / 2
            half_h = benchmark.macro_sizes[i, 1].item() / 2
            upper_x = min(weighted_x / total_weight, benchmark.canvas_width - half_w)
            upper_y = min(weighted_y / total_weight, benchmark.canvas_height - half_h)
            target_x = max(half_w, upper_x)
            target_y = max(half_h, upper_y)
            targets[name] = (target_x, target_y)
    
    # second pass
    for i in hard_indices:
        name = benchmark.macro_names[i]
        placement[i, 0] = targets[name][0]
        placement[i, 1] = targets[name][1]
    return placement