# utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id, seed=None):
    """Set random seed for each worker process of DataLoader"""
    if seed is not None:
        # Set deterministic seed for each worker
        worker_seed = seed + worker_id
    else:
        # Original logic
        worker_seed = torch.initial_seed() % 2**32
    
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)