"""
Policy Evaluation Script - Fixed-Trial Point-Goal Navigation

Evaluates the low-level RL policy using a fixed number of trials per distance.
For each distance (5m, 10m, 15m, 20m), runs exactly --num_trials (default 512)
with pre-determined spawn positions and goal directions (fixed seed).

All agents share a single 60m x 60m terrain tile with mixed obstacles
(80 cylinders + 60 boxes, 2m minimum clearance).

Metrics: Success Rate, Collision Rate, Stall Rate, Timeout Rate,
         Navigation Error, SPL.

Usage:
    ./isaaclab.sh -p evaluate_policy.py --checkpoint <path> --headless
    ./isaaclab.sh -p evaluate_policy.py --checkpoint <path> --num_trials 512 --headless
    ./isaaclab.sh -p evaluate_policy.py --checkpoint <path> --distances 5,10 --headless
"""

import os
import csv
import math
import argparse

import torch
import torch.nn as nn
import numpy as np

# =====================================================================
#  1. Isaac Lab App Launcher
# =====================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed-Trial Point-Goal Navigation Evaluation")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments")
parser.add_argument("--num_trials", type=int, default=512, help="Exact trials per distance")
parser.add_argument("--eval_seed", type=int, default=42, help="Seed for spawn generation")
parser.add_argument("--distances", type=str, default="5,10,15,20",
                    help="Comma-separated goal distances in meters")
parser.add_argument("--save_csv", type=str, default=None, help="Save results to CSV")
parser.add_argument("--export_jit", action="store_true", help="Export TorchScript JIT model")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.distances = [float(d) for d in args_cli.distances.split(",")]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =====================================================================
#  2. Imports (After launching app)
# =====================================================================
from skrl.models.torch import GaussianMixin, Model
from skrl.envs.wrappers.torch import wrap_env

from hivla_project.nav_env_eval import EvalNavigationEnvCfg
from hivla_project.nav_task import NavigationEnv, quat_to_yaw


# =====================================================================
#  3. Model Definition (Must match training architecture)
# =====================================================================
class NavigationPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)

        self.net_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn_out_size = 9216

        self.net_state = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.state_out_size = 64

        self.num_actions = action_space.shape[0]
        self.net_fusion = nn.Sequential(
            nn.Linear(self.cnn_out_size + self.state_out_size, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Tanh()
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        data = inputs["states"]
        if isinstance(data, dict):
            state_obs = data["state"]
            visual_obs = data["visual"]
        else:
            state_obs = data[:, 0:2]
            visual_flat = data[:, 2:]
            visual_obs = visual_flat.view(-1, 1, 128, 128)

        feat_visual = self.net_cnn(visual_obs)
        feat_state = self.net_state(state_obs)
        fused = torch.cat([feat_visual, feat_state], dim=1)
        mu = self.net_fusion(fused)
        return mu, self.log_std_parameter, {}


# =====================================================================
#  4. JIT Export Wrapper
# =====================================================================
class PolicyForJIT(nn.Module):
    def __init__(self, policy: NavigationPolicy):
        super().__init__()
        self.net_cnn = policy.net_cnn
        self.net_state = policy.net_state
        self.net_fusion = policy.net_fusion

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state_obs = obs[:, 0:2]
        visual_obs = obs[:, 2:].view(-1, 1, 128, 128)
        feat_visual = self.net_cnn(visual_obs)
        feat_state = self.net_state(state_obs)
        fused = torch.cat([feat_visual, feat_state], dim=1)
        return self.net_fusion(fused)


def export_jit(policy, device, checkpoint_path):
    jit_wrapper = PolicyForJIT(policy).to(device).eval()
    example_input = torch.randn(1, 2 + 128 * 128, device=device)
    with torch.inference_mode():
        traced = torch.jit.trace(jit_wrapper, example_input)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    jit_path = os.path.join(checkpoint_dir, "policy_jit.pt")
    traced.save(jit_path)
    print(f"[INFO] Exported JIT model to: {jit_path}")
    loaded = torch.jit.load(jit_path, map_location=device)
    test_out = loaded(example_input)
    print(f"[INFO] JIT verification: input {example_input.shape} -> output {test_out.shape}")
    return jit_path


# =====================================================================
#  5. Trial Parameter Generation (per-distance, with boundary validation)
# =====================================================================
def generate_trial_params(num_trials, eval_seed, distances, tile_size=60.0, border=3.0):
    """Pre-generate deterministic trial parameters per distance.

    Matches training reset distribution more closely by spawning at the tile
    center and varying only the initial yaw and goal direction.

    Args:
        num_trials: Number of trials per distance.
        eval_seed: Random seed for reproducibility.
        distances: List of goal distances (m).
        tile_size: Terrain tile size (m).
        border: Safety margin from tile edges (m).

    Returns:
        Dict mapping distance (float) → dict with arrays:
            spawn_positions (N,2) -- fixed at tile center,
            spawn_yaws (N,),
            goal_angles (N,).
    """
    rng = np.random.default_rng(eval_seed)
    safe_min = border
    safe_max = tile_size - border

    all_params = {}
    for dist in distances:
        positions = np.zeros((num_trials, 2), dtype=np.float32)
        yaws = np.zeros(num_trials, dtype=np.float32)
        angles = np.zeros(num_trials, dtype=np.float32)
        center = np.array([tile_size * 0.5, tile_size * 0.5], dtype=np.float32)

        for i in range(num_trials):
            positions[i] = center
            for _ in range(10000):
                a = rng.uniform(0, 2 * math.pi)
                gx = center[0] + dist * math.cos(a)
                gy = center[1] + dist * math.sin(a)
                if safe_min <= gx <= safe_max and safe_min <= gy <= safe_max:
                    angles[i] = a
                    break

            yaws[i] = rng.uniform(-math.pi, math.pi)

        all_params[dist] = {
            "spawn_positions": positions,
            "spawn_yaws": yaws,
            "goal_angles": angles,
        }

    return all_params


def validate_and_fix_trial_params(all_trial_params, distances, env_origin,
                                  tile_size=60.0, border=3.0, clearance=0.7):
    """Validate spawn/goal positions against PhysX obstacles and regenerate bad ones.

    Must be called AFTER physics warmup so PhysX scene query is available.

    Args:
        all_trial_params: Dict from generate_trial_params().
        distances: List of goal distances.
        env_origin: (3,) tensor — world origin of the terrain tile.
        tile_size: Terrain tile size (m).
        border: Safety margin from tile edges (m).
        clearance: Minimum obstacle-free radius around spawn/goal (m).
    """
    import omni.physx as _physx
    physx_query = _physx.get_physx_scene_query_interface()

    num_rays = 8
    ray_angles = [i * 2 * math.pi / num_rays for i in range(num_rays)]
    ray_height = 0.3  # robot chassis height

    ox = env_origin[0].item()
    oy = env_origin[1].item()
    safe_min = border
    safe_max = tile_size - border

    rng = np.random.default_rng(12345)  # separate seed for replacements

    def is_position_clear(tile_x, tile_y):
        """Check if a tile position is free of obstacles using PhysX raycasts."""
        # Convert tile coords to world coords
        wx = ox + tile_x - tile_size / 2.0
        wy = oy + tile_y - tile_size / 2.0

        # 1. Vertical check: is position inside an obstacle?
        hit = physx_query.raycast_closest((wx, wy, 2.0), (0.0, 0.0, -1.0), 3.0, False)
        if hit["hit"] and hit["position"][2] > 0.1:
            return False

        # 2. Horizontal clearance: 8-direction raycasts
        for angle in ray_angles:
            dx = math.cos(angle)
            dy = math.sin(angle)
            hit = physx_query.raycast_closest(
                (wx, wy, ray_height), (dx, dy, 0.0), clearance, False
            )
            if hit["hit"]:
                return False
        return True

    total_replaced = 0
    for dist in distances:
        params = all_trial_params[dist]
        n = len(params["spawn_positions"])
        replaced = 0

        for i in range(n):
            sx, sy = params["spawn_positions"][i]
            angle = params["goal_angles"][i]
            gx = sx + dist * math.cos(angle)
            gy = sy + dist * math.sin(angle)

            spawn_ok = is_position_clear(sx, sy)
            goal_ok = is_position_clear(gx, gy)

            if spawn_ok and goal_ok:
                continue

            center_x = tile_size * 0.5
            center_y = tile_size * 0.5

            if abs(sx - center_x) < 1e-4 and abs(sy - center_y) < 1e-4:
                # Keep the training-like center spawn and only resample goal direction.
                for _ in range(10000):
                    na = rng.uniform(0, 2 * math.pi)
                    ngx = sx + dist * math.cos(na)
                    ngy = sy + dist * math.sin(na)

                    if not (safe_min <= ngx <= safe_max and safe_min <= ngy <= safe_max):
                        continue
                    if not spawn_ok and not is_position_clear(sx, sy):
                        continue
                    if not is_position_clear(ngx, ngy):
                        continue

                    params["goal_angles"][i] = na
                    replaced += 1
                    break
            else:
                # Fallback for arbitrary spawn distributions.
                for _ in range(10000):
                    nx = rng.uniform(safe_min, safe_max)
                    ny = rng.uniform(safe_min, safe_max)
                    na = rng.uniform(0, 2 * math.pi)
                    ngx = nx + dist * math.cos(na)
                    ngy = ny + dist * math.sin(na)

                    if not (safe_min <= ngx <= safe_max and safe_min <= ngy <= safe_max):
                        continue
                    if not is_position_clear(nx, ny):
                        continue
                    if not is_position_clear(ngx, ngy):
                        continue

                    params["spawn_positions"][i] = [nx, ny]
                    params["goal_angles"][i] = na
                    replaced += 1
                    break

        total_replaced += replaced
        print(f"[INFO] Distance {dist:.0f}m: {replaced}/{n} trials had bad spawn/goal → replaced")

    print(f"[INFO] Total replaced: {total_replaced}")


def yaw_to_quat_batch(yaws: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles to quaternions (w, x, y, z)."""
    half = yaws * 0.5
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(yaws)
    return torch.stack([w, zeros, zeros, z], dim=-1)


# =====================================================================
#  6. Spawn & Goal Override
# =====================================================================
def setup_trial(env, env_id, trial_idx, target_dist, trial_params, device):
    """Override robot spawn position and goal for a specific trial.

    Args:
        env: The NavigationEnv instance.
        env_id: Index of the environment to configure.
        trial_idx: Index into trial_params arrays.
        target_dist: Goal distance in meters.
        trial_params: Dict for this distance from generate_trial_params().
        device: Torch device.
    """
    robot = env.scene["robot"]
    env_origin = env.scene.env_origins[env_id]  # (3,) — same for all envs (single tile)

    # Spawn position = env_origin base + absolute tile position
    # env_origin is at tile center (tile_size/2, tile_size/2), but spawn_positions
    # are absolute tile coords [0, tile_size]. So offset from origin = pos - tile_size/2.
    tile_pos = torch.tensor(trial_params["spawn_positions"][trial_idx], device=device)
    spawn_pos = env_origin.clone().to(device)
    spawn_pos[0] += tile_pos[0] - 30.0  # offset from tile center
    spawn_pos[1] += tile_pos[1] - 30.0
    spawn_pos[2] += 0.5  # robot spawn height

    # Spawn orientation
    yaw = torch.tensor(trial_params["spawn_yaws"][trial_idx], device=device)
    quat = yaw_to_quat_batch(yaw.unsqueeze(0)).squeeze(0)  # (4,)

    # Write root state
    root_state = torch.zeros(1, 13, device=device)
    root_state[0, :3] = spawn_pos
    root_state[0, 3:7] = quat
    # Velocities stay zero (indices 7:13)
    robot.write_root_state_to_sim(root_state, torch.tensor([env_id], device=device))

    # Goal position = spawn_pos + target_dist * direction
    angle = trial_params["goal_angles"][trial_idx]
    goal_x = spawn_pos[0].item() + target_dist * math.cos(angle)
    goal_y = spawn_pos[1].item() + target_dist * math.sin(angle)
    env.goal_pos[env_id, 0] = goal_x
    env.goal_pos[env_id, 1] = goal_y
    env.goal_pos[env_id, 2] = env_origin[2].item()

    # Reset per-env state
    env.stall_timer[env_id] = 0.0
    env.motion_accum[env_id] = 0.0
    env.costmap_core.costmap_tensor[env_id] = 0.0
    env.last_robot_pos[env_id] = spawn_pos
    env.last_robot_yaw[env_id] = yaw

    if hasattr(env, "was_path_blocked"):
        env.was_path_blocked[env_id] = False

    if hasattr(env, "last_abs_yaw_error"):
        target_yaw = math.atan2(goal_y - spawn_pos[1].item(),
                                goal_x - spawn_pos[0].item())
        yaw_err = target_yaw - yaw.item()
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        env.last_abs_yaw_error[env_id] = abs(yaw_err)

    env_id_tensor = torch.tensor([env_id], device=device)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_id:env_id + 1].clone(),
        torch.zeros_like(robot.data.default_joint_vel[env_id:env_id + 1]),
        env_ids=env_id_tensor,
    )


# =====================================================================
#  7. Fixed-Trial Evaluation
# =====================================================================
def run_fixed_trial_evaluation(env, wrapped_env, policy, device,
                               distances, all_trial_params, num_trials):
    """Run fixed-trial evaluation across multiple distances.

    For each distance, runs exactly num_trials episodes using a batch approach.
    All agents share the same single terrain tile.

    Returns:
        Dict mapping distance → metrics dict.
    """
    num_envs = env.num_envs
    robot = env.scene["robot"]
    eval_goal_threshold = env.cfg.goal_threshold + 0.1
    num_batches = math.ceil(num_trials / num_envs)

    print(f"\n[INFO] === Fixed-Trial Point-Goal Navigation Evaluation ===")
    print(f"[INFO] Distances: {distances}m")
    print(f"[INFO] Trials per distance: {num_trials}")
    print(f"[INFO] Parallel envs: {num_envs}, Batches per distance: {num_batches}")
    terrain_seed = getattr(env.cfg.scene.terrain.terrain_generator, "seed", None)
    print(f"[INFO] Goal threshold: {env.cfg.goal_threshold}m (eval: {eval_goal_threshold}m)")
    print(f"[INFO] Trial seed: {args_cli.eval_seed}")
    print(f"[INFO] Terrain seed: {terrain_seed}")
    print(f"[INFO] Single tile: all envs share the same 60x60m terrain")

    scenario_results = {}

    for target_dist in distances:
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {target_dist:.0f}m  ({num_trials} trials)")
        print(f"{'='*70}")

        # Get per-distance trial params
        dist_params = all_trial_params[target_dist]

        # Per-trial results
        trial_outcomes = [None] * num_trials
        trial_nav_errors = [0.0] * num_trials
        trial_ep_steps = [0] * num_trials
        trial_path_lengths = [0.0] * num_trials
        trial_goal_dists = [0.0] * num_trials

        completed_count = 0

        for batch_idx in range(num_batches):
            batch_start = batch_idx * num_envs
            batch_end = min(batch_start + num_envs, num_trials)
            batch_size = batch_end - batch_start
            active_env_ids = list(range(batch_size))

            # Reset all envs
            wrapped_env.reset()

            # Assign trials to envs
            env_to_trial = {}
            with torch.inference_mode():
                for local_i in range(batch_size):
                    trial_idx = batch_start + local_i
                    env_id = local_i
                    setup_trial(env, env_id, trial_idx, target_dist, dist_params, device)
                    env_to_trial[env_id] = trial_idx

            # Zero-action step to refresh observations after setup_trial override.
            # Without this, the first policy action uses stale obs from reset().
            with torch.inference_mode():
                zero_actions = torch.zeros(num_envs, 2, device=device)
                obs, _, _, _, _ = wrapped_env.step(zero_actions)

            # Per-env episode tracking (start AFTER the refresh step)
            ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
            ep_last_pos = robot.data.root_pos_w[:, :3].detach().clone()
            ep_path_length = torch.zeros(num_envs, device=device)

            # Record initial goal distances
            initial_goal_dist = torch.norm(
                env.goal_pos[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
            ).detach()

            # Track which envs still have active trials
            env_done = [False] * num_envs
            for i in range(batch_size, num_envs):
                env_done[i] = True  # No trial assigned to these envs

            # Run until all active trials in this batch are complete
            step_count = 0
            max_steps = 10000  # Safety limit per batch

            while not all(env_done[:batch_size]) and step_count < max_steps and simulation_app.is_running():
                with torch.inference_mode():
                    # Capture pre-step state
                    pre_robot_pos = robot.data.root_pos_w[:, :3].detach().clone()
                    pre_goal_pos = env.goal_pos.clone()
                    pre_stall_timer = env.stall_timer.clone()

                    # Path length tracking
                    step_dist = torch.norm(pre_robot_pos - ep_last_pos, dim=-1)
                    ep_path_length += step_dist
                    ep_last_pos[:] = pre_robot_pos

                    # Policy inference + step
                    actions, _, _ = policy.compute({"states": obs}, role="policy")

                    # Zero out actions for envs with no active trial
                    for i in range(num_envs):
                        if env_done[i]:
                            actions[i] = 0.0

                    obs, rewards, terminated, truncated, infos = wrapped_env.step(actions)
                    ep_steps += 1
                    step_count += 1

                    # Detect completions. ManagerBasedRLEnv resets done envs inside
                    # step(), so use termination_manager term buffers from this step
                    # rather than post-step robot/goal state.
                    terminated_flat = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
                    truncated_flat = truncated.squeeze(-1) if truncated.dim() > 1 else truncated
                    done = terminated_flat | truncated_flat
                    goal_term = env.termination_manager.get_term("reach_goal")
                    collision_term = env.termination_manager.get_term("collision")
                    stall_term = env.termination_manager.get_term("stall")

                    for i in range(batch_size):
                        if env_done[i] or not done[i].item():
                            continue

                        trial_idx = env_to_trial[i]

                        # Use the true termination cause from the current step.
                        robot_pos_i = pre_robot_pos[i, :2]
                        goal_pos_i = pre_goal_pos[i, :2]
                        nav_error = torch.norm(goal_pos_i - robot_pos_i).item()
                        is_truncated = truncated_flat[i].item()
                        goal_reached = goal_term[i].item()
                        hit_collision = collision_term[i].item()
                        was_stalled = stall_term[i].item()

                        if goal_reached:
                            outcome = "success"
                            nav_error = 0.0
                        elif is_truncated:
                            outcome = "timeout"
                        elif was_stalled:
                            outcome = "stall"
                        elif hit_collision:
                            outcome = "collision"
                        else:
                            outcome = "collision"

                        # Record result
                        trial_outcomes[trial_idx] = outcome
                        trial_nav_errors[trial_idx] = nav_error
                        trial_ep_steps[trial_idx] = ep_steps[i].item()
                        trial_path_lengths[trial_idx] = ep_path_length[i].item()
                        trial_goal_dists[trial_idx] = initial_goal_dist[i].item()

                        env_done[i] = True
                        completed_count += 1

            # Progress report per batch
            n_success_so_far = sum(1 for o in trial_outcomes if o == "success")
            print(f"  [{target_dist:.0f}m] Batch {batch_idx+1}/{num_batches} done | "
                  f"Completed: {completed_count}/{num_trials} | "
                  f"SR so far: {n_success_so_far/max(completed_count,1):.1%}")

        # Compute metrics for this distance
        n_ep = sum(1 for o in trial_outcomes if o is not None)
        if n_ep == 0:
            scenario_results[target_dist] = {
                'n_episodes': 0, 'sr': 0, 'cr': 0, 'str': 0, 'tr': 0,
                'ne': 0, 'spl': 0, 'avg_steps': 0,
            }
            print(f"  [WARNING] No trials completed for {target_dist:.0f}m!")
            continue

        n_s = sum(1 for o in trial_outcomes if o == "success")
        n_c = sum(1 for o in trial_outcomes if o == "collision")
        n_st = sum(1 for o in trial_outcomes if o == "stall")
        n_t = sum(1 for o in trial_outcomes if o == "timeout")

        nav_errors = np.array(trial_nav_errors[:n_ep])
        goal_dists = np.array(trial_goal_dists[:n_ep])
        path_lengths = np.array(trial_path_lengths[:n_ep])

        # SPL computation
        spl_values = []
        for j in range(n_ep):
            is_succ = 1.0 if trial_outcomes[j] == "success" else 0.0
            opt_l = goal_dists[j]
            act_l = path_lengths[j]
            spl_j = is_succ * (opt_l / max(opt_l, act_l)) if max(opt_l, act_l) > 0 else 0.0
            spl_values.append(spl_j)

        scenario_results[target_dist] = {
            'n_episodes': n_ep,
            'sr': n_s / n_ep,
            'cr': n_c / n_ep,
            'str': n_st / n_ep,
            'tr': n_t / n_ep,
            'ne': np.mean(nav_errors),
            'ne_std': np.std(nav_errors),
            'spl': np.mean(spl_values),
            'avg_steps': np.mean(trial_ep_steps[:n_ep]),
        }

        print(f"\n  [{target_dist:.0f}m] FINAL: {n_ep} trials | "
              f"SR={n_s/n_ep:.1%} CR={n_c/n_ep:.1%} StR={n_st/n_ep:.1%} TR={n_t/n_ep:.1%}")

    return scenario_results


# =====================================================================
#  8. Results Display & CSV Export
# =====================================================================
def print_results_table(scenario_results, distances, checkpoint_path):
    """Print formatted results table."""
    print("\n" + "=" * 82)
    print(f"  FIXED-TRIAL POINT-GOAL NAVIGATION EVALUATION")
    print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"  Trials per distance: {args_cli.num_trials} | Seed: {args_cli.eval_seed}")
    print("=" * 82)

    header = (f"  {'Distance':<10} {'Trials':>8} {'SR':>8} {'CR':>8} "
              f"{'StR':>8} {'TR':>8} {'NE':>8} {'SPL':>8}")
    print(header)
    print(f"  {'-' * 76}")

    total_ep = total_s = total_c = total_st = total_t = 0
    all_ne = []
    all_spl = []

    for dist in distances:
        r = scenario_results.get(dist, {})
        n = r.get('n_episodes', 0)
        total_ep += n
        total_s += int(r.get('sr', 0) * n)
        total_c += int(r.get('cr', 0) * n)
        total_st += int(r.get('str', 0) * n)
        total_t += int(r.get('tr', 0) * n)
        all_ne.append(r.get('ne', 0))
        all_spl.append(r.get('spl', 0))

        print(f"  {dist:<10.0f} {n:>8} {r.get('sr',0):>7.1%} {r.get('cr',0):>7.1%} "
              f"{r.get('str',0):>7.1%} {r.get('tr',0):>7.1%} "
              f"{r.get('ne',0):>7.2f}m {r.get('spl',0):>7.3f}")

    print(f"  {'-' * 76}")
    if total_ep > 0:
        print(f"  {'OVERALL':<10} {total_ep:>8} {total_s/total_ep:>7.1%} "
              f"{total_c/total_ep:>7.1%} {total_st/total_ep:>7.1%} "
              f"{total_t/total_ep:>7.1%} {np.mean(all_ne):>7.2f}m "
              f"{np.mean(all_spl):>7.3f}")

    print(f"\n  Avg Episode Steps per Distance:")
    for dist in distances:
        r = scenario_results.get(dist, {})
        end = "  |  " if dist != distances[-1] else "\n"
        print(f"    {dist:.0f}m: {r.get('avg_steps', 0):.0f}", end=end)

    print("=" * 82 + "\n")


def save_results_csv(scenario_results, distances, csv_path):
    """Save results to CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Distance_m', 'Trials', 'SR', 'CR', 'StR', 'TR',
            'NE', 'NE_std', 'SPL', 'Avg_Steps'
        ])
        for dist in distances:
            r = scenario_results.get(dist, {})
            writer.writerow([
                dist,
                r.get('n_episodes', 0),
                f"{r.get('sr', 0):.4f}",
                f"{r.get('cr', 0):.4f}",
                f"{r.get('str', 0):.4f}",
                f"{r.get('tr', 0):.4f}",
                f"{r.get('ne', 0):.4f}",
                f"{r.get('ne_std', 0):.4f}",
                f"{r.get('spl', 0):.4f}",
                f"{r.get('avg_steps', 0):.1f}",
            ])
    print(f"[INFO] Results saved to {csv_path}")


# =====================================================================
#  9. Main
# =====================================================================
def main():
    device = "cuda:0"

    env_cfg = EvalNavigationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    print(f"[INFO] Setting up evaluation environment ({args_cli.num_envs} envs, single tile)...")
    env = NavigationEnv(cfg=env_cfg)
    wrapped_env = wrap_env(env, wrapper="isaaclab")

    # 2. Model setup
    policy = NavigationPolicy(wrapped_env.observation_space, wrapped_env.action_space, device)
    policy.eval()
    policy.to(device)

    # 3. Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        raw_state_dict = checkpoint["policy"]
    else:
        raw_state_dict = checkpoint

    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        clean_state_dict[key] = v

    policy.load_state_dict(clean_state_dict, strict=True)
    print("[INFO] Successfully loaded policy weights.")

    # 4. Export JIT if requested
    if args_cli.export_jit:
        export_jit(policy, device, args_cli.checkpoint)

    # 5. Generate trial parameters (per-distance, boundary-validated)
    all_trial_params = generate_trial_params(
        args_cli.num_trials, args_cli.eval_seed,
        args_cli.distances, tile_size=60.0, border=3.0,
    )
    for dist in args_cli.distances:
        params = all_trial_params[dist]
        print(f"[INFO] Distance {dist:.0f}m: {len(params['spawn_positions'])} trials generated")

    # 6. Warmup (let physics settle)
    print("[INFO] Running warmup (50 steps)...")
    obs, _ = wrapped_env.reset()
    zero_actions = torch.zeros(args_cli.num_envs, 2, device=device)
    for _ in range(50):
        wrapped_env.step(zero_actions)

    # Validate spawn/goal positions against PhysX obstacles
    env_origin = env.scene.env_origins[0]
    if env.num_envs > 1:
        max_drift = (env.scene.env_origins - env_origin).abs().max().item()
        assert max_drift < 1e-3, (
            f"validate_and_fix_trial_params assumes a single shared tile but env_origins "
            f"differ by {max_drift:.3f}m. Spawn validation is only valid for env 0."
        )
    validate_and_fix_trial_params(
        all_trial_params, args_cli.distances, env_origin,
        tile_size=60.0, border=3.0, clearance=0.7,
    )

    # 7. Run fixed-trial evaluation
    scenario_results = run_fixed_trial_evaluation(
        env, wrapped_env, policy, device,
        args_cli.distances, all_trial_params, args_cli.num_trials
    )

    # 8. Print results
    print_results_table(scenario_results, args_cli.distances, args_cli.checkpoint)

    # 9. Save CSV
    if args_cli.save_csv:
        save_results_csv(scenario_results, args_cli.distances, args_cli.save_csv)

    # Cleanup
    print("[INFO] Closing environment...")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] {e}")
    finally:
        simulation_app.close()
