import os
import json
from datetime import datetime

def output_selection(args, trial, cycle, Q_index):
    base_dir = getattr(args, 'save_dir', './outputs')

    dataset_name = getattr(args, 'dataset', 'Dataset')
    method_name = getattr(args, 'method', 'Method')
    uncertainty_name = getattr(args, 'uncertainty', 'None')
    diversity_name = getattr(args, 'diversity', 'None')
    hybrid_strategy = getattr(args, 'hybrid_strategy', 'None')

    timestamp = datetime.now().strftime("%Y%m%d")

    if method_name == "Uncertainty":
        dir_name = f"{dataset_name}_{method_name}_{uncertainty_name}_{timestamp}"
    elif method_name == "Diversity":
        dir_name = f"{dataset_name}_{method_name}_{diversity_name}_{timestamp}"
    elif method_name == "Hybrid":
        dir_name = f"{dataset_name}_{method_name}_{hybrid_strategy}_{uncertainty_name}_{diversity_name}_{timestamp}"
    elif method_name == "Random":
        dir_name = f"{dataset_name}_{method_name}_{timestamp}"
    else:
         raise NotImplementedError(f"'{method_name}' not found.")

    save_dir = os.path.join(base_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"trial{trial}_cycle{cycle}_{len(Q_index)}.json"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(Q_index, f, indent=4) 
        
    print(f"[Log] Saved {len(Q_index)} selected indices to {filepath}")
