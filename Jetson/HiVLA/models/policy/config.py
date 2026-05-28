# models/policy/config.py

import torch

# ==============================================================================
# 1. Costmap and Grid Settings
# ==============================================================================
# Grid Resolution (number of cells in one dimension)
GRID_RES = 128
# Physical size of the costmap (in meters)
COSTMAP_SIZE_M = 6.4
# Resolution (meters per pixel/cell)
RESOLUTION = COSTMAP_SIZE_M / GRID_RES

# Filtering Thresholds (for LiDAR point clouds)
# Minimum Z-axis threshold for points to be considered valid obstacles
Z_MIN_THRESHOLD = 0.10
# Maximum Z-axis threshold for points to be considered valid obstacles
Z_MAX_THRESHOLD = 1.00

# Costmap Update Parameters
# Size of the kernel used for inflation (e.g., robot footprint)
INFLATION_KERNEL_SIZE = 3
# Factor by which old costmap cells decay (0.0 to 1.0)
DECAY_FACTOR = 0.9

# ==============================================================================
# 2. System and Hardware Settings
# ==============================================================================
# Primary device for PyTorch/GPU computations
DEVICE = 'cuda'
# Data type used for GPU tensors (FP16 is often preferred on Jetson Orin for performance)
DTYPE = torch.float16