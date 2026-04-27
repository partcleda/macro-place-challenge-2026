import torch
import numpy as np
from macro_place.objective import compute_proxy_cost

seed = 42

def sa_refine(placement, benchmark, plc, 
              initial_temp=1.0, 
              cooling_rate=0.995, 
              num_iterations=100):
    rng = np.random.default_rng(seed)
    
    # clones so we don't modify original until we want to
    current = placement.clone()
    best = placement.clone()
    
    # for updating cost as we go
    current_cost = compute_proxy_cost(current, benchmark, plc)['proxy_cost']
    best_cost = current_cost
    
    temp = initial_temp
    
    # find hard macro indices
    hard_mask = benchmark.get_hard_macro_mask()
    movable_mask = benchmark.get_movable_mask()
    movable_hard = hard_mask & movable_mask
    indices = torch.where(movable_hard)[0].tolist()
    
    for iteration in range(num_iterations):
        # pick random hard macro
        idx = indices[rng.integers(0, len(indices))]
        
        # storing old values in case result is less favorable
        old_x = current[idx, 0].item()
        old_y = current[idx, 1].item()
        
        # so that macro isn't outside boundary
        half_w = benchmark.macro_sizes[idx, 0]. item() / 2
        half_h = benchmark.macro_sizes[idx, 1]. item() / 2
        
        # random macro position
        new_x = rng.uniform(half_w, benchmark.canvas_width - half_w)
        new_y = rng.uniform(half_h, benchmark.canvas_width - half_h)
        
        # apply change
        current[idx, 0] = new_x
        current[idx, 1] = new_y
        
        # evaluate new cost
        new_cost = compute_proxy_cost(current, benchmark, plc)['proxy_cost']
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_cost = new_cost
            if new_cost < best_cost:
                best = current.clone()
                best_cost = new_cost
            else:
                current[idx, 0] = old_x
                current[idx, 1] = old_y
                
        temp *= cooling_rate
        if iteration % 20 == 0:
            print(f"Iter {iteration:4d} | Cost: {current_cost:.4f} | Temp: {temp:.4f}")
        
    return best

        
    
    
    
    