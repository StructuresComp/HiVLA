# models/policy/costmap.py

import torch
import torch.nn.functional as F
from .config import DEVICE, DTYPE, INFLATION_KERNEL_SIZE, DECAY_FACTOR, GRID_RES, Z_MIN_THRESHOLD, Z_MAX_THRESHOLD

# ==============================================================================
# 1. Batch GPU Local Costmap Core
# ==============================================================================

class BatchGPULocalCostmapCore:
    """
    Manages a local occupancy grid costmap entirely on the GPU.
    It handles LiDAR point cloud integration, map decay, inflation, 
    and ego-motion compensation via differentiable warping (F.grid_sample).
    """
    def __init__(self, device='cuda', num_envs=1, grid_res=128, map_size_m=6.4, resolution=0.05):
        self.device = device
        self.num_envs = num_envs
        self.grid_res = grid_res
        self.map_range_m = map_size_m / 2.0
        self.resolution = resolution
        self.inv_resolution = 1.0 / resolution 
        
        # Initialize the core costmap tensor (Batch, Channel, Height, Width)
        self.costmap_tensor = torch.zeros(
            (num_envs, 1, grid_res, grid_res), 
            device=device, 
            dtype=DTYPE
        )
        self.inflation_kernel_size = INFLATION_KERNEL_SIZE

        # Pre-allocate Affine Matrix Buffer for F.affine_grid
        # The matrix template is:
        # [[1 0 0]  (Rotation/Scale + Translation X)
        #  [0 1 0]] (Rotation/Scale + Translation Y)
        self.affine_matrix_buffer = torch.zeros(
            (num_envs, 2, 3), 
            device=device, 
            dtype=DTYPE
        )
        self.affine_matrix_buffer[:, 0, 0] = 1.0 # R00
        self.affine_matrix_buffer[:, 1, 1] = 1.0 # R11

    def update_costmap(self, points_batch, dx_batch, dy_batch, dtheta_batch):
        """
        Executes the main costmap update loop (Shift, Decay, Update, Inflate).
        """
        # 1. Compensate for Ego-Motion (Map Warping)
        self.costmap_tensor = self._compensate_ego_motion(
            self.costmap_tensor, dx_batch, dy_batch, dtheta_batch
        )
        
        # 2. Decay Old Observations
        self.costmap_tensor *= DECAY_FACTOR

        # 3. Integrate New Sensor Data
        if points_batch is not None:
            # Ensure datatype matches system configuration (FP16/FP32)
            if points_batch.dtype != DTYPE:
                points_batch = points_batch.to(DTYPE)
                
            new_obs = self._batch_lidar_to_occupancy(points_batch)
            # Use maximum to treat new data as obstacle updates
            self.costmap_tensor = torch.maximum(self.costmap_tensor, new_obs)

        # 4. Inflate Obstacles (using max pooling for morphological dilation)
        inflated_costmap = F.max_pool2d(
            self.costmap_tensor, 
            kernel_size=self.inflation_kernel_size, 
            stride=1, 
            padding=self.inflation_kernel_size // 2
        )
        
        return inflated_costmap

    def _compensate_ego_motion(self, current_map, dx, dy, dtheta):
        """
        Uses an Affine Transformation to shift and rotate the map, compensating 
        for robot movement since the last update.
        """
        cos_t = torch.cos(dtheta).to(DTYPE)
        sin_t = torch.sin(dtheta).to(DTYPE)
        
        # Normalize translation to the [-1, 1] range required by affine_grid, 
        # where 1.0 corresponds to the map edge (map_range_m).
        trans_x = (dx / self.map_range_m).to(DTYPE)
        trans_y = (dy / self.map_range_m).to(DTYPE)

        # Build the transformation matrix (mat)
        # mat = [[R00 R01 T02]
        #        [R10 R11 T12]]
        mat = self.affine_matrix_buffer
        mat[:, 0, 0] = cos_t     # R00 (Rotation X)
        mat[:, 0, 1] = -sin_t    # R01 (Rotation Y)
        mat[:, 0, 2] = trans_x   # T02 (Translation X)
        mat[:, 1, 0] = sin_t     # R10
        mat[:, 1, 1] = cos_t     # R11
        mat[:, 1, 2] = trans_y   # T12 (Translation Y)

        # Calculate the inverse sampling grid based on the transformation matrix
        grid_coords = F.affine_grid(mat, current_map.size(), align_corners=False)
        
        # Warp the map using grid_sample to perform the shift/rotation
        shifted_map = F.grid_sample(
            current_map, 
            grid_coords, 
            mode='nearest', 
            padding_mode='zeros', 
            align_corners=False
        )
        
        return shifted_map

    def _batch_lidar_to_occupancy(self, points):
        """
        Converts a batch of (X, Y, Z) LiDAR points into a 2D occupancy grid 
        using GPU tensor indexing (stamping).
        """
        # Utilizing FP16 to maximize Jetson Orin's Tensor Core efficiency
        B, N, _ = points.shape
        
        # 1. Convert METERS (m) to PIXEL INDICES (px)
        # Formula: (X_m + MapRange_m) * InvResolution = U_px
        u = ((points[:, :, 0] + self.map_range_m) * self.inv_resolution).long() 
        v = ((points[:, :, 1] + self.map_range_m) * self.inv_resolution).long() 
        
        # 2. Validity masking (Bounds & Z-Height)
        valid_mask = (u >= 0) & (u < self.grid_res) & \
                     (v >= 0) & (v < self.grid_res) & \
                     (points[:, :, 2] >= Z_MIN_THRESHOLD) & (points[:, :, 2] <= Z_MAX_THRESHOLD)

        # 3. Create zero occupancy map template
        occupancy = torch.zeros_like(self.costmap_tensor)
        
        # Prepare batch indices for stamping
        batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
        
        # 4. Extract indices corresponding to valid points
        b_idx = batch_indices[valid_mask]
        u_idx = u[valid_mask]
        v_idx = v[valid_mask]
        
        # 5. GPU Parallel Stamping
        # Set all valid (b, c, v, u) locations to 1.0 simultaneously
        occupancy.index_put_((b_idx, torch.zeros_like(b_idx), v_idx, u_idx), 
                             torch.tensor(1.0, device=self.device, dtype=DTYPE))
                             
        return occupancy