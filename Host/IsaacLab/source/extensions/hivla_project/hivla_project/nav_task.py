from PIL import Image
import math
import numpy as np
import os
import torch
import omni.physx as _physx

from isaaclab.envs import ManagerBasedRLEnv

# Custom Import
from .core_map import BatchGPULocalCostmapCore


def quat_to_yaw(quaternions: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def compute_visual_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns the local occupancy costmap as a visual observation.
    Shape: (num_envs, 1, 128, 128)
    """
    if not hasattr(env, 'current_costmap'):
        return torch.zeros((env.num_envs, 1, 128, 128), device=env.device, dtype=torch.float16)

    obs = env.current_costmap.clone()
    # Adjust for CNN format: [Batch, Channel, Height, Width] and correct orientation
    obs = obs.permute(0, 1, 3, 2)
    obs = torch.flip(obs, dims=[-2, -1])
    return obs

def compute_state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the goal position relative to the robot's local frame. Output: (num_envs, 2)"""
    if not hasattr(env, 'goal_pos'):
         return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    robot_pos = env.scene["robot"].data.root_pos_w
    robot_quat = env.scene["robot"].data.root_quat_w
    diff = env.goal_pos - robot_pos
    robot_yaw = quat_to_yaw(robot_quat)
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    goal_x_local = diff[:, 0] * cos_yaw + diff[:, 1] * sin_yaw
    goal_y_local = -diff[:, 0] * sin_yaw + diff[:, 1] * cos_yaw
    state_vec = torch.stack([goal_x_local, goal_y_local], dim=-1) / 20.0
    state_vec = torch.clamp(state_vec, -1.0, 1.0)
    return state_vec


def _get_collision_status(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Detects collisions using force sensors on chassis and wheels.
    Ignores collisions during the warm-up period (first 10 steps).
    """
    is_early_step = env.episode_length_buf < 20
    
    contact_sensor = env.scene["contact_sensor"]
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]

    # 1. Chassis Collision Check (matches the 200N criterion reported in the paper)
    chassis_force = torch.norm(forces[:, 0, :], dim=-1)
    is_chassis_crash = chassis_force > 200.0

    # 2. Wheel Collision Check
    if forces.shape[1] > 1:
        wheel_forces = forces[:, 1:, :]
        wheel_forces_xy = torch.sqrt(wheel_forces[..., 0]**2 + wheel_forces[..., 1]**2)
        max_wheel_xy = torch.max(wheel_forces_xy, dim=-1).values
        is_wheel_crash = max_wheel_xy > 200.0
    else:
        is_wheel_crash = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    is_colliding = is_chassis_crash | is_wheel_crash
    is_colliding[is_early_step] = False
    
    return is_colliding
    
def check_collision_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _get_collision_status(env)

def check_goal_reached(env: ManagerBasedRLEnv, threshold: float = None) -> torch.Tensor:
    """Checks if the robot has reached the goal position (2D XY distance only)."""
    if threshold is None:
        threshold = env.cfg.goal_threshold
    diff = env.goal_pos[:, :2] - env.scene["robot"].data.root_pos_w[:, :2]
    current_dist = torch.norm(diff, dim=-1)
    return current_dist < threshold

def check_stall_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminates if the robot has been stuck for more than 3.0 seconds."""
    return env.stall_timer > 3.0

def compute_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    # -----------------------------------------------------------------
    # 1. Data Extraction & Preprocessing
    # -----------------------------------------------------------------
    robot = env.scene["robot"]
    current_pos = robot.data.root_pos_w
    lin_vel_x = robot.data.root_lin_vel_b[:, 0]
    ang_vel_z = robot.data.root_ang_vel_b[:, 2] 
    
    robot_quat = robot.data.root_quat_w
    robot_yaw = quat_to_yaw(robot_quat)

    # -----------------------------------------------------------------
    # 2. Goal & Heading Calculation
    # -----------------------------------------------------------------
    to_goal = env.goal_pos - current_pos
    dist_to_goal = torch.norm(to_goal[:, :2], dim=-1)  # 2D XY only (goal z=0, robot z≈0.5)
    target_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    
    # Calculate Yaw Error (Wrapped to -pi ~ pi)
    yaw_error = target_yaw - robot_yaw
    yaw_error = torch.where(yaw_error > math.pi, yaw_error - 2*math.pi, yaw_error)
    yaw_error = torch.where(yaw_error < -math.pi, yaw_error + 2*math.pi, yaw_error)
    abs_yaw_error = torch.abs(yaw_error)
    
    # -----------------------------------------------------------------
    # 3. Lidar Processing & Filtering
    # -----------------------------------------------------------------
    sensor = env.scene["lidar"]
    lidar_ranges = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    lidar_ranges = torch.nan_to_num(lidar_ranges, posinf=10.0)
    
    # Transform Lidar points to Robot Local Frame
    pts_x = lidar_ranges * env.lidar_dir_x
    pts_y = lidar_ranges * env.lidar_dir_y
    # Dynamic sensor height (robot settles to different Z than spawn height)
    sensor_height = (sensor.data.pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(1)  # (B,1)
    pts_z = lidar_ranges * env.lidar_dir_z + sensor_height

    ROBOT_HALF_LEN = 0.47 + 0.06
    ROBOT_HALF_WID = 0.35 + 0.10

    # Filtering Masks (Ignore Ground & Self-body)
    ground_mask = pts_z < 0.15  # Match costmap Z_MIN (0.1) + margin; ground hits ≈ 0.0
    ceiling_mask = pts_z > 1.5  # Ignore sky/ceiling hits from upward rays
    self_mask = (pts_x > -ROBOT_HALF_LEN) & (pts_x < ROBOT_HALF_LEN) & \
                (pts_y > -ROBOT_HALF_WID) & (pts_y < ROBOT_HALF_WID)
    combined_mask = ground_mask | self_mask | ceiling_mask

    # -----------------------------------------------------------------
    # 4. Obstacle Detection & Path Analysis
    # -----------------------------------------------------------------
    # Forward range: 1.5m (Long look-ahead to prevent collision with large obstacles)
    CORRIDOR_FORWARD = 1.5 
    # Width margin: 0.1m (Tight margin to allow traversing narrow passages)
    CORRIDOR_WIDTH_MARGIN = 0.35
    
    # Define safety corridor
    valid_forward = (pts_x > ROBOT_HALF_LEN) & (pts_x < (ROBOT_HALF_LEN + CORRIDOR_FORWARD))
    valid_width = torch.abs(pts_y) < (ROBOT_HALF_WID + CORRIDOR_WIDTH_MARGIN)
    corridor_mask = valid_forward & valid_width & (~combined_mask)
    
    # Calculate distance to the nearest obstacle within the corridor
    corridor_ranges = lidar_ranges.clone()
    corridor_ranges[~corridor_mask] = 100.0
    min_corridor_dist = torch.min(corridor_ranges, dim=-1).values

    is_path_blocked = min_corridor_dist < 0.8
    is_colliding = _get_collision_status(env)

    # -----------------------------------------------------------------
    # 5. Side Zone Detection (for navigating between obstacles)
    # -----------------------------------------------------------------
    SIDE_ZONE_FORWARD = 0.8   # How far ahead to check on sides
    SIDE_ZONE_WIDTH = 1.0     # How far to the side to check
    SIDE_DANGER_THRESHOLD = 1.3  # Distance at which side becomes "tight" (was 1.0)

    # Left side zone (positive y in robot frame)
    left_zone = (pts_x > -0.2) & (pts_x < SIDE_ZONE_FORWARD) & \
                (pts_y > ROBOT_HALF_WID) & (pts_y < ROBOT_HALF_WID + SIDE_ZONE_WIDTH)
    left_zone = left_zone & (~combined_mask)

    # Right side zone (negative y in robot frame)
    right_zone = (pts_x > -0.2) & (pts_x < SIDE_ZONE_FORWARD) & \
                 (pts_y < -ROBOT_HALF_WID) & (pts_y > -(ROBOT_HALF_WID + SIDE_ZONE_WIDTH))
    right_zone = right_zone & (~combined_mask)

    # Min distance in each side zone
    left_ranges = lidar_ranges.clone()
    left_ranges[~left_zone] = 100.0
    min_left_dist = torch.min(left_ranges, dim=-1).values

    right_ranges = lidar_ranges.clone()
    right_ranges[~right_zone] = 100.0
    min_right_dist = torch.min(right_ranges, dim=-1).values

    # Tightness factor (0 = wide open, 1 = very tight)
    left_tightness = torch.clamp(1.0 - min_left_dist / SIDE_DANGER_THRESHOLD, 0, 1)
    right_tightness = torch.clamp(1.0 - min_right_dist / SIDE_DANGER_THRESHOLD, 0, 1)
    side_tightness = torch.max(left_tightness, right_tightness)

    # =====================================================================
    # Reward Calculation (Core Logic)
    # =====================================================================
    
    # [A] Goal Progress & Success
    # Hybrid shaping: dense linear term works at any distance; exp term boosts the
    # "homing" gradient near the goal where fine positioning matters.
    prev_dist = torch.norm(env.goal_pos[:, :2] - env.last_robot_pos[:, :2], dim=-1)
    linear_progress = (prev_dist - dist_to_goal) * 10.0
    k = 0.25
    exp_homing = (torch.exp(-k * dist_to_goal) - torch.exp(-k * prev_dist)) * 50.0
    rew_goal = linear_progress + exp_homing
    rew_success = torch.where(dist_to_goal < env.cfg.goal_threshold, 20.0, 0.0)

    # [B] Projected Velocity Reward (Straight-Line Driver)
    vel_projected = lin_vel_x * torch.cos(yaw_error)
    velocity_scale = 1.0 - 0.85 * side_tightness
    rew_velocity = torch.where(
        ~is_path_blocked,
        vel_projected * 3.0 * velocity_scale,
        0.0
    )

    # [C] Heading Alignment Reward
    alignment_score = 1.0 - (abs_yaw_error / math.pi)
    is_moving_forward = lin_vel_x > 0.1
    rew_heading = torch.where(
        is_moving_forward & (~is_path_blocked),
        alignment_score * 1.0,
        0.0
    )

    # [D] Safety & Danger Penalties
    rew_collision = torch.where(is_colliding, -50.0, 0.0)

    MAX_CORRIDOR_DIST = math.sqrt((ROBOT_HALF_LEN + CORRIDOR_FORWARD)**2 + (ROBOT_HALF_WID + CORRIDOR_WIDTH_MARGIN)**2)
    normalized_dist = torch.clamp(min_corridor_dist / MAX_CORRIDOR_DIST, min=0.0, max=1.0)
    rew_danger = -3.0 * ((1.0 - normalized_dist) ** 2)

    # [E] Angular Control (Anti-Oscillation)
    angular_scale = 1.0 - side_tightness
    rew_angular = torch.where(
        is_path_blocked,
        0.0,
        torch.abs(ang_vel_z) * -1.0 * angular_scale
    )

    # [E2] Centering Reward
    side_imbalance = right_tightness - left_tightness
    rew_centering = side_imbalance * ang_vel_z * 2.5

    # [F] Stall Prevention (cumulative displacement + grace period)
    # Accumulate per-step displacement; reset stall when enough total motion
    if not hasattr(env, 'motion_accum'):
        env.motion_accum = torch.zeros(env.num_envs, device=env.device)
    step_disp_3d = torch.norm(current_pos - env.last_robot_pos, dim=-1)
    env.motion_accum = env.motion_accum + step_disp_3d
    has_escaped = env.motion_accum > 0.2  # 20cm cumulative motion resets stall
    dt = env.step_dt

    env.stall_timer = torch.where(has_escaped, torch.zeros_like(env.stall_timer), env.stall_timer + dt)
    env.motion_accum = torch.where(has_escaped, torch.zeros_like(env.motion_accum), env.motion_accum)
    stall_grace = 1.0  # seconds: allow turning/repositioning before penalty
    t_after_grace = torch.clamp(env.stall_timer - stall_grace, min=0.0)
    rew_stall = torch.clamp(-0.5 * (t_after_grace ** 2), min=-2.0)

    # [G] Displacement Reward (encourages any motion / exploration)
    step_disp = torch.norm(current_pos[:, :2] - env.last_robot_pos[:, :2], dim=-1)
    rew_move = 0.2 * torch.tanh(step_disp / 0.03)

    # [H] Yaw Error Reduction Reward (rewards turning to face goal)
    if not hasattr(env, 'last_abs_yaw_error'):
        env.last_abs_yaw_error = abs_yaw_error.clone()
    rew_yaw = 2.0 * (env.last_abs_yaw_error - abs_yaw_error)
    env.last_abs_yaw_error = abs_yaw_error.clone()

    rew_time = -0.05

    total_reward = (
        rew_goal
        + rew_success
        + rew_velocity
        + rew_heading
        + rew_angular
        + rew_centering
        + rew_collision
        + rew_danger
        + rew_stall
        + rew_move
        + rew_yaw
        + rew_time
    )

    # SKRL reads env.extras["episode"] and logs each key as "Info / {key}"
    if not hasattr(env, '_rew_accum'):
        env._rew_accum = {}
        env._rew_step_count = 0
    env._rew_step_count += 1

    components = {
        "rew_goal": rew_goal,
        "rew_success": rew_success,
        "rew_velocity": rew_velocity,
        "rew_heading": rew_heading,
        "rew_angular": rew_angular,
        "rew_centering": rew_centering,
        "rew_collision": rew_collision,
        "rew_danger": rew_danger,
        "rew_stall": rew_stall,
        "rew_move": rew_move,
        "rew_yaw": rew_yaw,
    }
    for name, val in components.items():
        if name not in env._rew_accum:
            env._rew_accum[name] = 0.0
        env._rew_accum[name] += val.mean().item()

    # flush every 100 steps (matches SKRL write_interval)
    if env._rew_step_count % 100 == 0:
        episode_info = {}
        for name in components:
            episode_info[name] = torch.tensor(env._rew_accum[name] / 100.0)
            env._rew_accum[name] = 0.0
        episode_info["dist_to_goal"] = torch.tensor(dist_to_goal.mean().item())
        episode_info["collision_rate"] = torch.tensor(is_colliding.float().mean().item())
        episode_info["blocked_rate"] = torch.tensor(is_path_blocked.float().mean().item())
        episode_info["mean_speed"] = torch.tensor(lin_vel_x.mean().item())
        env.extras["episode"] = episode_info
    return total_reward



class NavigationEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.costmap_core = BatchGPULocalCostmapCore(
            device=self.device,
            num_envs=self.num_envs,
            grid_res=128,
            map_size_m=6.4
        )
        
        self.current_costmap = torch.zeros(
            (self.num_envs, 1, 128, 128), 
            device=self.device, 
            dtype=torch.float16
        )
        
        # Buffers
        self.last_robot_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_robot_yaw = torch.zeros(self.num_envs, device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Lidar vectors (initialized lazily)
        self.lidar_dir_x = None
        self.lidar_dir_y = None
        self.lidar_dir_z = None

        self.stall_timer = torch.zeros(self.num_envs, device=self.device)

        # Success-gated curriculum tracking
        self.curriculum_level = 0.0  # [0, 1] — advances only when success_rate > threshold
        self._curriculum_successes = 0
        self._curriculum_episodes = 0

    def _precompute_lidar_directions(self, num_rays: int = None):
        """Precomputes Lidar ray direction vectors based on FOV config.

        Args:
            num_rays: If provided, infer num_horz from total rays / channels.
                      Otherwise, compute from config.
        """
        lidar_cfg = self.scene["lidar"].cfg.pattern_cfg
        num_channels = lidar_cfg.channels

        elevations = torch.linspace(
            lidar_cfg.vertical_fov_range[0],
            lidar_cfg.vertical_fov_range[1],
            num_channels,
            device=self.device
        )
        elevations = torch.deg2rad(elevations)

        # Fix #1: Infer num_horz from actual data or compute from config
        if num_rays is not None:
            num_horz = num_rays // num_channels
        else:
            h_fov = lidar_cfg.horizontal_fov_range[1] - lidar_cfg.horizontal_fov_range[0]
            num_horz = int(h_fov / lidar_cfg.horizontal_res) + 1

        azimuths = torch.linspace(
            math.radians(lidar_cfg.horizontal_fov_range[0]),
            math.radians(lidar_cfg.horizontal_fov_range[1]),
            num_horz,
            device=self.device
        )
        elev_grid, azim_grid = torch.meshgrid(elevations, azimuths, indexing="ij")

        # Spherical to Cartesian
        self.lidar_dir_x = (torch.cos(elev_grid) * torch.cos(azim_grid)).flatten().unsqueeze(0).expand(self.num_envs, -1)
        self.lidar_dir_y = (torch.cos(elev_grid) * torch.sin(azim_grid)).flatten().unsqueeze(0).expand(self.num_envs, -1)
        self.lidar_dir_z = torch.sin(elev_grid).flatten().unsqueeze(0).expand(self.num_envs, -1)

    def _update_costmap(self) -> None:
        """Updates the local costmap with ego-motion compensation and lidar data."""
        robot_pos = self.scene["robot"].data.root_pos_w
        robot_quat = self.scene["robot"].data.root_quat_w
        robot_yaw = quat_to_yaw(robot_quat)

        # 1. Odometry Calculation (Global to Local Delta)
        if self.common_step_counter > 0:
            dx_global = robot_pos[:, 0] - self.last_robot_pos[:, 0]
            dy_global = robot_pos[:, 1] - self.last_robot_pos[:, 1]
            dtheta = robot_yaw - self.last_robot_yaw
            dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
            
            cos_yaw = torch.cos(self.last_robot_yaw)
            sin_yaw = torch.sin(self.last_robot_yaw)
            dx_local = dx_global * cos_yaw + dy_global * sin_yaw
            dy_local = -dx_global * sin_yaw + dy_global * cos_yaw
        else:
            dx_local = torch.zeros(self.num_envs, device=self.device)
            dy_local = torch.zeros(self.num_envs, device=self.device)
            dtheta = torch.zeros(self.num_envs, device=self.device)

        # 2. Lidar Processing
        sensor = self.scene["lidar"]
        hits_w = sensor.data.ray_hits_w
        pos_w = sensor.data.pos_w
        diff_vec = hits_w - pos_w.unsqueeze(1)
        ranges = torch.norm(diff_vec, dim=-1)
        ranges = torch.nan_to_num(ranges, posinf=150.0)


        if self.lidar_dir_x is None or self.lidar_dir_x.shape[1] != ranges.shape[1]:
            self._precompute_lidar_directions(num_rays=ranges.shape[1])

        pts_x = ranges * self.lidar_dir_x
        pts_y = ranges * self.lidar_dir_y

        # sensor height varies as robot settles — compute dynamically
        sensor_height = (pos_w[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(1)
        pts_z = ranges * self.lidar_dir_z + sensor_height

        half_len = 0.47 + 0.08
        half_wid = 0.35 + 0.05
        self_mask = (pts_x > -half_len) & (pts_x < half_len) & (pts_y > -half_wid) & (pts_y < half_wid)
        pts_z[self_mask] = -100.0

        points_local = torch.stack([pts_x, pts_y, pts_z], dim=-1)
        self.current_costmap = self.costmap_core.update_costmap(points_local, dx_local, dy_local, dtheta)

        self.last_robot_pos[:] = robot_pos
        self.last_robot_yaw[:] = robot_yaw

    def step(self, action: torch.Tensor):
        self._update_costmap()
        return super().step(action)

    def _update_curriculum(self, reached_goal_mask: torch.Tensor):
        """Update success-gated curriculum. Call on episode termination."""
        n = reached_goal_mask.numel()
        self._curriculum_episodes += n
        self._curriculum_successes += reached_goal_mask.sum().item()

        # Evaluate every 200 episodes to reduce noise
        if self._curriculum_episodes >= 200:
            success_rate = self._curriculum_successes / self._curriculum_episodes
            if success_rate > 0.6:
                # Advance by 5% per evaluation window
                self.curriculum_level = min(self.curriculum_level + 0.05, 1.0)
            elif success_rate < 0.3:
                # Regress slightly if struggling
                self.curriculum_level = max(self.curriculum_level - 0.02, 0.0)
            self._curriculum_successes = 0
            self._curriculum_episodes = 0

    def _sample_goals(self, env_ids, total_steps):
        """Generates random goal positions with success-gated curriculum.

        `total_steps` is unused for distance scaling (curriculum is success-gated via
        `self.curriculum_level`); it remains in the signature for backwards compatibility.
        """
        progress = self.curriculum_level
        MIN_DIST = 5.0
        MAX_DIST = 20.0
        current_max_dist = MIN_DIST + (MAX_DIST - MIN_DIST) * progress
        rand_dist = (torch.rand(len(env_ids), device=self.device)
                     * (current_max_dist - MIN_DIST)) + MIN_DIST
        rand_angles = torch.rand(len(env_ids), device=self.device) * (2 * math.pi)
        dx = rand_dist * torch.cos(rand_angles)
        dy = rand_dist * torch.sin(rand_angles)
        new_goal_offset = torch.stack([dx, dy, torch.zeros_like(dx)], dim=-1)
        self.goal_pos[env_ids] = self.scene.env_origins[env_ids] + new_goal_offset

    def _reset_idx(self, env_ids):
        if not hasattr(self, "stall_timer"):
             self.stall_timer = torch.zeros(self.num_envs, device=self.device)
        if not hasattr(self, "motion_accum"):
            self.motion_accum = torch.zeros(self.num_envs, device=self.device)

        # Update success-gated curriculum before sampling new goals
        if hasattr(self, "termination_manager") and len(env_ids) > 0:
            reached_goal = self.termination_manager.get_term("reach_goal")[env_ids]
            self._update_curriculum(reached_goal)

        total_steps = self.cfg.curriculum_steps
        MAX_RETRIES = 20
        self._sample_goals(env_ids, total_steps)
        current_env_ids_to_check = env_ids
        for retry in range(MAX_RETRIES):
            candidate_goals = self.goal_pos[current_env_ids_to_check]
            is_valid = self._check_validity_physx(candidate_goals, clearance=0.8)
            if torch.all(is_valid): break
            invalid_indices = torch.nonzero(~is_valid).squeeze(-1)
            ids_to_resample = current_env_ids_to_check[invalid_indices]
            if len(ids_to_resample) == 0: break
            self._sample_goals(ids_to_resample, total_steps)
            current_env_ids_to_check = ids_to_resample
        else:
            n_still_invalid = (~is_valid).sum().item() if not torch.all(is_valid) else 0
            if n_still_invalid > 0:
                print(f"[WARN] {n_still_invalid}/{len(env_ids)} goals still invalid after {MAX_RETRIES} retries")
        super()._reset_idx(env_ids)
        robot_asset = self.scene["robot"]
        default_root_state = robot_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        rand_yaw = (torch.rand(len(env_ids), device=self.device) * 2 * math.pi) - math.pi
        half_yaw = rand_yaw * 0.5
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)
        new_quat = torch.zeros((len(env_ids), 4), device=self.device)
        new_quat[:, 0] = cy
        new_quat[:, 3] = sy
        default_root_state[:, 3:7] = new_quat
        default_root_state[:, 7:] = 0.0
        robot_asset.write_root_state_to_sim(default_root_state, env_ids)

        spawn_ok = self._check_spawn_clearance(env_ids)
        if not spawn_ok.all():
            n_bad = int((~spawn_ok).sum().item())
            print(f"[WARN] {n_bad}/{len(env_ids)} spawns blocked at platform center "
                  f"(check terrain platform exclusion vs min_clearance config).")

        self.stall_timer[env_ids] = 0.0
        self.motion_accum[env_ids] = 0.0
        if not hasattr(self, "last_abs_yaw_error"):
            self.last_abs_yaw_error = torch.zeros(self.num_envs, device=self.device)
        # initialize to actual yaw error at spawn to avoid a spurious positive reward spike
        reset_yaw = quat_to_yaw(new_quat)
        to_goal_reset = self.goal_pos[env_ids, :2] - default_root_state[:, :2]
        target_yaw_reset = torch.atan2(to_goal_reset[:, 1], to_goal_reset[:, 0])
        yaw_err_reset = target_yaw_reset - reset_yaw
        yaw_err_reset = torch.where(yaw_err_reset > math.pi, yaw_err_reset - 2*math.pi, yaw_err_reset)
        yaw_err_reset = torch.where(yaw_err_reset < -math.pi, yaw_err_reset + 2*math.pi, yaw_err_reset)
        self.last_abs_yaw_error[env_ids] = torch.abs(yaw_err_reset)
        # Reset blocked transition tracker
        if not hasattr(self, "was_path_blocked"):
            self.was_path_blocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.was_path_blocked[env_ids] = False
        self.costmap_core.costmap_tensor[env_ids] = 0.0
        self.last_robot_pos[env_ids] = default_root_state[:, :3]
        reset_quat = default_root_state[:, 3:7]
        self.last_robot_yaw[env_ids] = quat_to_yaw(reset_quat)

    def _check_validity_physx(self, positions: torch.Tensor, clearance: float = 1.0) -> torch.Tensor:
        """
        Checks if positions are valid for navigation:
          1. Not inside an obstacle (vertical raycast)
          2. Sufficient clearance around the position (horizontal raycasts)

        Args:
            positions: (N, 3) world positions to check
            clearance: minimum obstacle-free radius in meters (default 1.0m ≈ robot width)
        """
        physx_query_interface = _physx.get_physx_scene_query_interface()
        valid_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=self.device)

        # 8 horizontal directions for clearance check
        num_rays = 8
        ray_angles = [i * 2 * math.pi / num_rays for i in range(num_rays)]
        ray_height = 0.3  # robot chassis height

        for i in range(positions.shape[0]):
            pos = positions[i]
            px, py = pos[0].item(), pos[1].item()

            # 1. Vertical check: is position inside an obstacle?
            hit = physx_query_interface.raycast_closest(
                (px, py, 2.0), (0.0, 0.0, -1.0), 3.0, False
            )
            if hit["hit"] and hit["position"][2] > 0.1:
                valid_mask[i] = False
                continue

            # 2. Horizontal clearance: raycast outward in 8 directions
            for angle in ray_angles:
                dx = math.cos(angle)
                dy = math.sin(angle)
                hit = physx_query_interface.raycast_closest(
                    (px, py, ray_height), (dx, dy, 0.0), clearance, False
                )
                if hit["hit"]:
                    valid_mask[i] = False
                    break

        return valid_mask

    def _check_spawn_clearance(self, env_ids) -> torch.Tensor:
        """
        Checks if robot spawn positions have enough clearance to start navigating.
        Raycasts 8 directions from spawn at chassis height.
        Returns bool mask: True = spawn is clear.
        """
        physx_query_interface = _physx.get_physx_scene_query_interface()
        spawn_pos = self.scene.env_origins[env_ids]  # robots spawn at tile center
        valid_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)

        num_rays = 8
        ray_angles = [i * 2 * math.pi / num_rays for i in range(num_rays)]
        clearance = 0.5
        ray_height = 0.3

        for i in range(len(env_ids)):
            px = spawn_pos[i, 0].item()
            py = spawn_pos[i, 1].item()
            blocked_count = 0

            for angle in ray_angles:
                dx = math.cos(angle)
                dy = math.sin(angle)
                hit = physx_query_interface.raycast_closest(
                    (px, py, ray_height), (dx, dy, 0.0), clearance, False
                )
                if hit["hit"]:
                    blocked_count += 1

            # If more than half the directions are blocked, spawn is bad
            if blocked_count > num_rays // 2:
                valid_mask[i] = False

        return valid_mask

    def _save_debug_costmap(self, step_count, env_id=0):
        save_dir = "debug_costmaps"
        os.makedirs(save_dir, exist_ok=True)
        
        raw_map = self.current_costmap[env_id].clone().detach()
        obs = raw_map.unsqueeze(0) 
        obs = obs.permute(0, 1, 3, 2)
        obs = torch.flip(obs, dims=[-2, -1])
        
        costmap_np = obs.squeeze().cpu().numpy()
        costmap_np = np.clip(costmap_np, 0.0, 1.0)
        img_data = (255 - (costmap_np * 255)).astype(np.uint8)
        
        img = Image.fromarray(img_data, mode='L')
        filename = f"{save_dir}/env_{env_id}_step_{step_count:05d}.png"
        img.save(filename)
