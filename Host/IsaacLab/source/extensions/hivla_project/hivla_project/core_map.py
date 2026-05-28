import torch
import torch.nn.functional as F

GRID_RES = 128
COSTMAP_SIZE_M = 6.4
RESOLUTION = COSTMAP_SIZE_M / GRID_RES
Z_MIN_THRESHOLD = 0.1
Z_MAX_THRESHOLD = 1.0
INFLATION_KERNEL_SIZE = 3
DECAY_FACTOR = 0.9

DTYPE = torch.float16


class BatchGPULocalCostmapCore:
    def __init__(self, device='cuda', num_envs=1, grid_res=128, map_size_m=6.4, resolution=0.05):
        self.device = device
        self.num_envs = num_envs
        self.grid_res = grid_res
        self.map_range_m = map_size_m / 2.0
        self.resolution = resolution
        self.inv_resolution = 1.0 / resolution

        self.costmap_tensor = torch.zeros(
            (num_envs, 1, grid_res, grid_res),
            device=device,
            dtype=DTYPE
        )
        self.inflation_kernel_size = INFLATION_KERNEL_SIZE

        self.affine_matrix_buffer = torch.zeros((num_envs, 2, 3), device=device, dtype=DTYPE)
        self.affine_matrix_buffer[:, 0, 0] = 1.0
        self.affine_matrix_buffer[:, 1, 1] = 1.0

    def update_costmap(self, points_batch, dx_batch, dy_batch, dtheta_batch):
        self.costmap_tensor = self._compensate_ego_motion(
            self.costmap_tensor, dx_batch, dy_batch, dtheta_batch
        )
        self.costmap_tensor *= DECAY_FACTOR

        if points_batch is not None:
            if points_batch.dtype != DTYPE:
                points_batch = points_batch.to(DTYPE)
            new_obs = self._batch_lidar_to_occupancy(points_batch)
            self.costmap_tensor = torch.maximum(self.costmap_tensor, new_obs)

        return F.max_pool2d(
            self.costmap_tensor,
            kernel_size=self.inflation_kernel_size,
            stride=1,
            padding=self.inflation_kernel_size // 2
        )

    def _compensate_ego_motion(self, current_map, dx, dy, dtheta):
        cos_t = torch.cos(dtheta).to(DTYPE)
        sin_t = torch.sin(dtheta).to(DTYPE)
        trans_x = (dx / self.map_range_m).to(DTYPE)
        trans_y = (dy / self.map_range_m).to(DTYPE)

        mat = self.affine_matrix_buffer
        mat[:, 0, 0] = cos_t
        mat[:, 0, 1] = -sin_t
        mat[:, 0, 2] = trans_x
        mat[:, 1, 0] = sin_t
        mat[:, 1, 1] = cos_t
        mat[:, 1, 2] = trans_y

        grid_coords = F.affine_grid(mat, current_map.size(), align_corners=False)
        return F.grid_sample(
            current_map,
            grid_coords,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )

    def _batch_lidar_to_occupancy(self, points):
        B, N, _ = points.shape

        u = ((points[:, :, 0] + self.map_range_m) * self.inv_resolution).long()
        v = ((points[:, :, 1] + self.map_range_m) * self.inv_resolution).long()

        valid_mask = (
            (u >= 0) & (u < self.grid_res) &
            (v >= 0) & (v < self.grid_res) &
            (points[:, :, 2] >= Z_MIN_THRESHOLD) &
            (points[:, :, 2] <= Z_MAX_THRESHOLD)
        )

        occupancy = torch.zeros_like(self.costmap_tensor)
        batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(B, N)

        b_idx = batch_indices[valid_mask]
        u_idx = u[valid_mask]
        v_idx = v[valid_mask]

        occupancy.index_put_(
            (b_idx, torch.zeros_like(b_idx), v_idx, u_idx),
            torch.tensor(1.0, device=self.device, dtype=DTYPE)
        )
        return occupancy
