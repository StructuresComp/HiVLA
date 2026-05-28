"""
Classical Planner Evaluation — Fixed-Trial Point-Goal Navigation

Evaluates three classical navigation baselines (APF, RRT*, DWA) using the
same fixed-trial methodology as evaluate_policy.py for fair comparison.

All planners use the same costmap (128x128, 0.05m/cell) and robot state
available to the RL policy, but replace the neural network with handcrafted
planning algorithms.

Metrics: Success Rate, Collision Rate, Stall Rate, Timeout Rate,
         Navigation Error, SPL.

Usage:
    ./isaaclab.sh -p evaluate_classical.py --planner apf --headless
    ./isaaclab.sh -p evaluate_classical.py --planner rrt_star --headless
    ./isaaclab.sh -p evaluate_classical.py --planner dwa --headless
    ./isaaclab.sh -p evaluate_classical.py --planner all --headless
"""

import csv
import math
import argparse

import torch
import numpy as np

# =====================================================================
#  1. Isaac Lab App Launcher
# =====================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Classical Planner Evaluation")
parser.add_argument("--planner", type=str, default="all",
                    choices=["apf", "dwa", "mppi", "teb", "all"],
                    help="Which planner to evaluate")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments")
parser.add_argument("--num_trials", type=int, default=512, help="Trials per distance")
parser.add_argument("--eval_seed", type=int, default=42, help="Seed for spawn generation")
parser.add_argument("--distances", type=str, default="5,10,15,20",
                    help="Comma-separated goal distances in meters")
parser.add_argument("--save_csv", type=str, default=None, help="Save results to CSV")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.distances = [float(d) for d in args_cli.distances.split(",")]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =====================================================================
#  2. Imports (After launching app)
# =====================================================================
from skrl.envs.wrappers.torch import wrap_env

from hivla_project.nav_env_eval import EvalNavigationEnvCfg
from hivla_project.nav_task import NavigationEnv, quat_to_yaw


# =====================================================================
#  3. Classical Planners
# =====================================================================

# Shared constants
GRID_RES = 128
MAP_RANGE_M = 3.2       # costmap covers ±3.2m
RESOLUTION = MAP_RANGE_M * 2 / GRID_RES  # 0.05m/cell
CENTER = GRID_RES // 2  # robot is at pixel (64, 64)
MAX_V = 0.5             # m/s max linear velocity
MAX_W = 1.5             # rad/s max angular velocity


def costmap_to_numpy(env, env_id):
    """Extract inflated costmap as numpy array (128, 128) in [0, 1].
    Uses current_costmap (post-inflation) for consistency with RL policy.
    Raw costmap layout: row=y-axis, col=x-axis."""
    return env.current_costmap[env_id, 0].float().cpu().numpy()


def batch_costmaps_to_numpy(env, n):
    """Batch transfer: single GPU→CPU call for all envs at once."""
    return env.current_costmap[:n, 0].float().cpu().numpy()


def batch_robot_states(env, n):
    """Batch transfer: get all robot states and goals in one GPU→CPU call."""
    robot = env.scene["robot"]
    positions = robot.data.root_pos_w[:n, :2].cpu().numpy()
    quats = robot.data.root_quat_w[:n].cpu()
    yaws = quat_to_yaw(quats).numpy()
    goals = env.goal_pos[:n, :2].cpu().numpy()
    return positions, yaws, goals


def get_robot_state(env, env_id):
    """Get robot position, yaw, and goal in world frame."""
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[env_id, :2].cpu()
    quat = robot.data.root_quat_w[env_id].unsqueeze(0)
    yaw = quat_to_yaw(quat).item()
    goal = env.goal_pos[env_id, :2].cpu()
    return pos, yaw, goal


def world_to_local(pos, yaw, target):
    """Convert world target to robot-local frame (forward=+x, left=+y).
    Works with both torch tensors and numpy arrays/scalars."""
    dx = float(target[0] - pos[0])
    dy = float(target[1] - pos[1])
    cos_y = math.cos(float(yaw))
    sin_y = math.sin(float(yaw))
    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y
    return local_x, local_y


def local_to_grid(local_x, local_y):
    """Convert local-frame meters to costmap grid coordinates.
    In the raw costmap: row = y-axis, col = x-axis.
    u = (x + map_range) / resolution  →  col
    v = (y + map_range) / resolution  →  row
    """
    col = int((local_x + MAP_RANGE_M) / RESOLUTION)  # x → col
    row = int((local_y + MAP_RANGE_M) / RESOLUTION)  # y → row
    return row, col


def grid_to_local(row, col):
    """Convert grid coordinates back to local-frame meters.
    col → x,  row → y."""
    local_x = col * RESOLUTION - MAP_RANGE_M
    local_y = row * RESOLUTION - MAP_RANGE_M
    return local_x, local_y


def is_free(costmap, row, col, threshold=0.3):
    """Check if a grid cell is obstacle-free."""
    if 0 <= row < GRID_RES and 0 <= col < GRID_RES:
        return costmap[row, col] < threshold
    return False


# =====================================================================
#  3a. APF — Artificial Potential Field
# =====================================================================
class APFPlanner:
    """Artificial Potential Field planner: attractive force toward goal + repulsive from obstacles."""
    def __init__(self, k_att=1.0, k_rep=0.8, rep_radius_m=1.5):
        self.k_att = k_att
        self.k_rep = k_rep
        self.rep_radius_cells = int(rep_radius_m / RESOLUTION)

    def compute_action(self, costmap, goal_local_x, goal_local_y):
        """
        Args:
            costmap: (128, 128) numpy occupancy grid
            goal_local_x, goal_local_y: goal in robot-local frame (meters)
        Returns:
            (v_cmd, w_cmd) in [-1, 1]
        """
        # --- Attractive force toward goal ---
        goal_dist = math.sqrt(goal_local_x**2 + goal_local_y**2)
        if goal_dist < 0.01:
            return 0.0, 0.0

        att_x = self.k_att * goal_local_x / goal_dist
        att_y = self.k_att * goal_local_y / goal_dist

        # --- Repulsive force from obstacles ---
        rep_x, rep_y = 0.0, 0.0
        r = self.rep_radius_cells
        robot_row, robot_col = CENTER, CENTER

        # Scan cells around robot within repulsion radius
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                row = robot_row + dr
                col = robot_col + dc
                if not (0 <= row < GRID_RES and 0 <= col < GRID_RES):
                    continue
                occ = costmap[row, col]
                if occ < 0.3:
                    continue

                # Direction from obstacle to robot in local frame
                obs_lx, obs_ly = grid_to_local(row, col)
                dist_to_obs = math.sqrt(obs_lx**2 + obs_ly**2)
                if dist_to_obs < 0.05:
                    dist_to_obs = 0.05

                max_dist = r * RESOLUTION
                # Repulsive magnitude: stronger when closer
                magnitude = self.k_rep * occ * (1.0 / dist_to_obs - 1.0 / max_dist)
                if magnitude < 0:
                    continue
                magnitude = magnitude / dist_to_obs

                # Push away from obstacle (toward robot = positive direction)
                rep_x += magnitude * (-obs_lx / dist_to_obs)
                rep_y += magnitude * (-obs_ly / dist_to_obs)

        # --- Combine forces ---
        fx = att_x + rep_x
        fy = att_y + rep_y

        # Convert force to velocity commands
        # Desired heading angle in robot frame
        desired_angle = math.atan2(fy, fx)

        # Angular velocity proportional to heading error
        w_cmd = np.clip(desired_angle / (math.pi / 2), -1.0, 1.0)

        # Linear velocity: fast when aligned, slow when turning
        alignment = max(0.0, math.cos(desired_angle))
        v_cmd = alignment * 0.8  # scale down from max

        # Slow down near obstacles
        center_occ = costmap[CENTER-5:CENTER+5, CENTER-5:CENTER+5].max()
        if center_occ > 0.5:
            v_cmd *= 0.3

        return float(np.clip(v_cmd, -1.0, 1.0)), float(np.clip(w_cmd, -1.0, 1.0))


# =====================================================================
#  3b. RRT* — Sampling-based Path Planning
# =====================================================================
class RRTStarPlanner:
    """RRT* on the local costmap. Plans robot-to-goal path on 128x128 grid, replans every N steps."""
    def __init__(self, max_iters=1500, step_size=5, goal_sample_rate=0.2,
                 replan_interval=15, neighbor_radius=10):
        self.max_iters = max_iters
        self.step_size = step_size  # grid cells
        self.goal_sample_rate = goal_sample_rate
        self.replan_interval = replan_interval
        self.neighbor_radius = neighbor_radius
        self.paths = {}        # env_id -> list of (row, col) waypoints
        self.step_counts = {}  # env_id -> steps since last plan

    def reset(self, env_id):
        self.paths.pop(env_id, None)
        self.step_counts.pop(env_id, None)

    def _plan(self, costmap, start_rc, goal_rc):
        """Run RRT* on costmap grid. Returns path as list of (row, col) or None."""
        sr, sc = start_rc
        gr, gc = goal_rc

        # Clamp goal to grid
        gr = max(0, min(GRID_RES - 1, gr))
        gc = max(0, min(GRID_RES - 1, gc))

        if not is_free(costmap, gr, gc, threshold=0.4):
            # Goal in obstacle — find nearest free cell
            found = False
            for radius in range(1, 20):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if is_free(costmap, gr + dr, gc + dc, threshold=0.4):
                            gr, gc = gr + dr, gc + dc
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        # RRT* tree: node_id -> (row, col, parent_id, cost)
        nodes = [(sr, sc, -1, 0.0)]
        rng = np.random.default_rng()

        best_goal_node = None
        best_goal_cost = float('inf')

        for _ in range(self.max_iters):
            # Sample random point (biased toward goal)
            if rng.random() < self.goal_sample_rate:
                rand_r, rand_c = gr, gc
            else:
                rand_r = rng.integers(0, GRID_RES)
                rand_c = rng.integers(0, GRID_RES)

            # Find nearest node
            min_dist = float('inf')
            nearest_id = 0
            for nid, (nr, nc, _, _) in enumerate(nodes):
                d = (nr - rand_r)**2 + (nc - rand_c)**2
                if d < min_dist:
                    min_dist = d
                    nearest_id = nid

            nr, nc = nodes[nearest_id][0], nodes[nearest_id][1]

            # Steer toward random point
            dr = rand_r - nr
            dc = rand_c - nc
            dist = math.sqrt(dr**2 + dc**2)
            if dist < 1:
                continue
            new_r = int(nr + self.step_size * dr / dist)
            new_c = int(nc + self.step_size * dc / dist)

            if not (0 <= new_r < GRID_RES and 0 <= new_c < GRID_RES):
                continue

            # Collision check along edge
            if not self._edge_free(costmap, nr, nc, new_r, new_c):
                continue

            step_cost = math.sqrt((new_r - nr)**2 + (new_c - nc)**2)
            new_cost = nodes[nearest_id][3] + step_cost

            # RRT* rewire: check nearby nodes for better parent
            best_parent = nearest_id
            best_cost = new_cost
            nearby = []
            for nid, (nnr, nnc, _, ncost) in enumerate(nodes):
                d = math.sqrt((nnr - new_r)**2 + (nnc - new_c)**2)
                if d < self.neighbor_radius:
                    nearby.append(nid)
                    candidate_cost = ncost + d
                    if candidate_cost < best_cost:
                        if self._edge_free(costmap, nnr, nnc, new_r, new_c):
                            best_parent = nid
                            best_cost = candidate_cost

            new_id = len(nodes)
            nodes.append((new_r, new_c, best_parent, best_cost))

            # Rewire nearby nodes through new node if cheaper
            for nid in nearby:
                nnr, nnc = nodes[nid][0], nodes[nid][1]
                d = math.sqrt((nnr - new_r)**2 + (nnc - new_c)**2)
                if best_cost + d < nodes[nid][3]:
                    if self._edge_free(costmap, new_r, new_c, nnr, nnc):
                        nodes[nid] = (nnr, nnc, new_id, best_cost + d)

            # Check if new node reaches goal
            gdist = math.sqrt((new_r - gr)**2 + (new_c - gc)**2)
            if gdist < self.step_size and best_cost < best_goal_cost:
                if self._edge_free(costmap, new_r, new_c, gr, gc):
                    best_goal_node = new_id
                    best_goal_cost = best_cost

        if best_goal_node is None:
            return None

        # Extract path
        path = [(gr, gc)]
        nid = best_goal_node
        while nid != -1:
            path.append((nodes[nid][0], nodes[nid][1]))
            nid = nodes[nid][2]
        path.reverse()
        return path

    def _edge_free(self, costmap, r1, c1, r2, c2, threshold=0.4):
        """Check if the line between two grid cells is obstacle-free."""
        steps = max(abs(r2 - r1), abs(c2 - c1))
        if steps == 0:
            return is_free(costmap, r1, c1, threshold)
        for i in range(steps + 1):
            t = i / steps
            r = int(r1 + t * (r2 - r1))
            c = int(c1 + t * (c2 - c1))
            if not is_free(costmap, r, c, threshold):
                return False
        return True

    def compute_action(self, costmap, goal_local_x, goal_local_y, env_id=0):
        """Plan path and follow first waypoint."""
        count = self.step_counts.get(env_id, 0)
        path = self.paths.get(env_id, None)

        # Replan if needed
        if path is None or count >= self.replan_interval or len(path) < 2:
            start_rc = (CENTER, CENTER)
            goal_rc = local_to_grid(goal_local_x, goal_local_y)
            path = self._plan(costmap, start_rc, goal_rc)
            self.paths[env_id] = path
            self.step_counts[env_id] = 0

        self.step_counts[env_id] = count + 1

        if path is None or len(path) < 2:
            # No path found — drive straight toward goal
            angle = math.atan2(goal_local_y, goal_local_x)
            w_cmd = np.clip(angle / (math.pi / 2), -1.0, 1.0)
            v_cmd = 0.3 * max(0, math.cos(angle))
            return float(v_cmd), float(w_cmd)

        # Follow next waypoint (skip first which is robot position)
        # Find the lookahead waypoint (skip nearby ones)
        lookahead_dist = 8  # grid cells (~0.4m)
        wp_idx = 1
        for i in range(1, len(path)):
            wr, wc = path[i]
            d = math.sqrt((wr - CENTER)**2 + (wc - CENTER)**2)
            if d > lookahead_dist:
                wp_idx = i
                break
            wp_idx = i

        target_r, target_c = path[wp_idx]
        target_lx, target_ly = grid_to_local(target_r, target_c)

        # Pure pursuit
        angle = math.atan2(target_ly, target_lx)
        w_cmd = np.clip(angle / (math.pi / 3), -1.0, 1.0)
        alignment = max(0.0, math.cos(angle))
        v_cmd = alignment * 0.8

        return float(np.clip(v_cmd, -1.0, 1.0)), float(np.clip(w_cmd, -1.0, 1.0))


# =====================================================================
#  3c. DWA — Dynamic Window Approach
# =====================================================================
class DWAPlanner:
    """Dynamic Window Approach: samples (v, w) pairs, simulates trajectories, picks best scored."""
    def __init__(self, n_v=9, n_w=21, sim_time=1.5, sim_dt=0.1,
                 w_goal=3.0, w_clearance=2.0, w_speed=0.5):
        self.n_v = n_v
        self.n_w = n_w
        self.sim_time = sim_time
        self.sim_dt = sim_dt
        self.w_goal = w_goal
        self.w_clearance = w_clearance
        self.w_speed = w_speed

    def compute_action(self, costmap, goal_local_x, goal_local_y):
        """Evaluate candidate trajectories (vectorized) and return best (v_cmd, w_cmd)."""
        goal_dist = math.sqrt(goal_local_x**2 + goal_local_y**2)

        # Create all (v, w) combinations: (N,) where N = n_v * n_w
        v_vals = np.linspace(0.0, 1.0, self.n_v)
        w_vals = np.linspace(-1.0, 1.0, self.n_w)
        v_grid, w_grid = np.meshgrid(v_vals, w_vals, indexing='ij')
        v_flat = v_grid.ravel()  # (N,)
        w_flat = w_grid.ravel()  # (N,)
        N = len(v_flat)

        n_steps = int(self.sim_time / self.sim_dt)

        # Simulate all N trajectories in parallel
        x = np.zeros(N)
        y = np.zeros(N)
        theta = np.zeros(N)
        min_clearance = np.ones(N)
        alive = np.ones(N, dtype=bool)

        v_mps = v_flat * MAX_V  # (N,)
        w_rps = w_flat * MAX_W  # (N,)

        for _ in range(n_steps):
            x += alive * v_mps * np.cos(theta) * self.sim_dt
            y += alive * v_mps * np.sin(theta) * self.sim_dt
            theta += alive * w_rps * self.sim_dt

            cols = ((x + MAP_RANGE_M) / RESOLUTION).astype(int)
            rows = ((y + MAP_RANGE_M) / RESOLUTION).astype(int)

            # Out-of-bounds → collision
            oob = alive & ((rows < 0) | (rows >= GRID_RES) | (cols < 0) | (cols >= GRID_RES))
            alive[oob] = False

            # In-bounds occupancy check
            in_bounds = alive & ~oob
            if np.any(in_bounds):
                ib_idx = np.where(in_bounds)[0]
                r_ib = np.clip(rows[ib_idx], 0, GRID_RES - 1)
                c_ib = np.clip(cols[ib_idx], 0, GRID_RES - 1)
                occ = costmap[r_ib, c_ib]

                collided = occ > 0.5
                alive[ib_idx[collided]] = False

                surv_idx = ib_idx[~collided]
                min_clearance[surv_idx] = np.minimum(min_clearance[surv_idx], 1.0 - occ[~collided])

        # Score all trajectories (only alive ones get real scores, dead ones get -inf)
        scores = np.full(N, -np.inf)

        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            # All collide — pick the one with smallest w (go straight slowly)
            return 0.1, 0.0

        # Goal heading alignment
        end_angle = np.arctan2(goal_local_y - y[alive_idx], goal_local_x - x[alive_idx])
        heading_err = np.abs(end_angle - theta[alive_idx])
        heading_err = np.minimum(heading_err, 2 * np.pi - heading_err)
        score_goal = 1.0 - heading_err / np.pi

        # Clearance
        score_clearance = min_clearance[alive_idx]

        # Speed
        score_speed = v_flat[alive_idx]

        # Progress
        end_dist = np.sqrt((goal_local_x - x[alive_idx])**2 + (goal_local_y - y[alive_idx])**2)
        score_progress = (goal_dist - end_dist) / max(goal_dist, 0.1)

        scores[alive_idx] = (self.w_goal * score_goal +
                             self.w_clearance * score_clearance +
                             self.w_speed * score_speed +
                             1.5 * score_progress)

        best_idx = np.argmax(scores)
        return float(v_flat[best_idx]), float(w_flat[best_idx])


# =====================================================================
#  3d. MPPI — Model Predictive Path Integral Control
# =====================================================================
class MPPIPlanner:
    """MPPI: samples K noisy trajectories, evaluates costmap costs, returns cost-weighted action.

    Reference: Williams et al., ICRA 2017.
    """
    def __init__(self, K=256, T=20, dt=0.1, lambda_=10.0,
                 noise_v=0.3, noise_w=0.5,
                 w_goal=5.0, w_obs=50.0, w_speed=0.5):
        self.K = K           # number of sampled trajectories
        self.T = T           # planning horizon (steps)
        self.dt = dt         # time step for simulation
        self.lambda_ = lambda_  # temperature (inverse)
        self.noise_v = noise_v  # std dev for linear vel noise
        self.noise_w = noise_w  # std dev for angular vel noise
        self.w_goal = w_goal
        self.w_obs = w_obs
        self.w_speed = w_speed
        # Warm-start: previous nominal control sequence per env
        self.prev_v = {}
        self.prev_w = {}

    def reset(self, env_id=0):
        self.prev_v.pop(env_id, None)
        self.prev_w.pop(env_id, None)

    def compute_action(self, costmap, goal_local_x, goal_local_y, env_id=0):
        goal_dist = math.sqrt(goal_local_x**2 + goal_local_y**2)
        if goal_dist < 0.01:
            return 0.0, 0.0

        rng = np.random.default_rng()
        K, T = self.K, self.T

        # Nominal control sequence (warm-start from previous or default)
        if env_id in self.prev_v:
            nom_v = np.roll(self.prev_v[env_id], -1)
            nom_w = np.roll(self.prev_w[env_id], -1)
            nom_v[-1] = nom_v[-2]
            nom_w[-1] = nom_w[-2]
        else:
            nom_v = np.full(T, 0.5)
            nom_w = np.zeros(T)

        # Sample noise perturbations: (K, T)
        eps_v = rng.normal(0, self.noise_v, (K, T))
        eps_w = rng.normal(0, self.noise_w, (K, T))

        # Perturbed control sequences: (K, T)
        v_samples = np.clip(nom_v[None, :] + eps_v, 0.0, 1.0)
        w_samples = np.clip(nom_w[None, :] + eps_w, -1.0, 1.0)

        # === VECTORIZED trajectory simulation across all K samples ===
        # State arrays: (K,)
        x = np.zeros(K)
        y = np.zeros(K)
        theta = np.zeros(K)
        costs = np.zeros(K)
        alive = np.ones(K, dtype=bool)  # tracks which trajectories haven't collided

        for t in range(T):
            v_mps = v_samples[:, t] * MAX_V    # (K,)
            w_rps = w_samples[:, t] * MAX_W    # (K,)

            x += alive * v_mps * np.cos(theta) * self.dt
            y += alive * v_mps * np.sin(theta) * self.dt
            theta += alive * w_rps * self.dt

            # Grid coords for all K samples
            cols = ((x + MAP_RANGE_M) / RESOLUTION).astype(int)
            rows = ((y + MAP_RANGE_M) / RESOLUTION).astype(int)

            # Out-of-bounds check
            oob = alive & ((rows < 0) | (rows >= GRID_RES) | (cols < 0) | (cols >= GRID_RES))
            costs[oob] += self.w_obs * 5.0
            alive[oob] = False

            # In-bounds occupancy check
            in_bounds = alive & ~oob
            if np.any(in_bounds):
                ib_idx = np.where(in_bounds)[0]
                r_ib = np.clip(rows[ib_idx], 0, GRID_RES - 1)
                c_ib = np.clip(cols[ib_idx], 0, GRID_RES - 1)
                occ = costmap[r_ib, c_ib]

                # Collision
                collided = occ > 0.5
                col_idx = ib_idx[collided]
                costs[col_idx] += self.w_obs * 10.0
                alive[col_idx] = False

                # Obstacle proximity cost for survivors
                surv_idx = ib_idx[~collided]
                costs[surv_idx] += self.w_obs * occ[~collided]

            # Speed reward
            costs[alive] -= self.w_speed * v_samples[alive, t]

        # Goal cost at trajectory end
        end_dist = np.sqrt((goal_local_x - x)**2 + (goal_local_y - y)**2)
        costs += self.w_goal * end_dist

        # Heading cost at trajectory end
        goal_angle = np.arctan2(goal_local_y - y, goal_local_x - x)
        heading_err = np.abs(goal_angle - theta)
        heading_err = np.minimum(heading_err, 2 * np.pi - heading_err)
        costs += 2.0 * heading_err

        # MPPI weighting: softmin
        costs_shifted = costs - np.min(costs)
        weights = np.exp(-costs_shifted / self.lambda_)
        weight_sum = np.sum(weights)
        if weight_sum < 1e-10:
            weights = np.ones(K) / K
        else:
            weights /= weight_sum

        # Weighted average control sequence
        opt_v = np.sum(weights[:, None] * v_samples, axis=0)
        opt_w = np.sum(weights[:, None] * w_samples, axis=0)

        # Save for warm-start
        self.prev_v[env_id] = opt_v
        self.prev_w[env_id] = opt_w

        return float(np.clip(opt_v[0], 0.0, 1.0)), float(np.clip(opt_w[0], -1.0, 1.0))


# =====================================================================
#  3e. TEB — Timed Elastic Band
# =====================================================================
class TEBPlanner:
    """
    Simplified Timed Elastic Band planner.
    Initializes a band of poses from robot to subgoal, then iteratively
    optimizes for: shortest path, obstacle clearance, and smoothness.

    Reference: Rösmann et al., MMAR 2012.
    """
    def __init__(self, n_poses=12, iters=30, lr=0.03,
                 w_obstacle=8.0, w_path=1.0, w_smooth=2.0, w_goal=3.0):
        self.n_poses = n_poses
        self.iters = iters
        self.lr = lr
        self.w_obstacle = w_obstacle
        self.w_path = w_path
        self.w_smooth = w_smooth
        self.w_goal = w_goal

    def compute_action(self, costmap, goal_local_x, goal_local_y):
        goal_dist = math.sqrt(goal_local_x**2 + goal_local_y**2)
        if goal_dist < 0.01:
            return 0.0, 0.0

        # Subgoal: clamp to costmap range if goal is too far
        max_range = MAP_RANGE_M * 0.85  # stay within costmap
        if goal_dist > max_range:
            scale = max_range / goal_dist
            sg_x = goal_local_x * scale
            sg_y = goal_local_y * scale
        else:
            sg_x = goal_local_x
            sg_y = goal_local_y

        # Initialize band: straight line from robot (0,0) to subgoal
        n = self.n_poses
        band_x = np.linspace(0.0, sg_x, n)
        band_y = np.linspace(0.0, sg_y, n)

        # Optimize band via gradient descent
        for _ in range(self.iters):
            grad_x = np.zeros(n)
            grad_y = np.zeros(n)

            for i in range(1, n - 1):  # don't move start or end
                # --- Obstacle gradient ---
                row, col = local_to_grid(band_x[i], band_y[i])
                if 0 <= row < GRID_RES and 0 <= col < GRID_RES:
                    occ = costmap[row, col]
                    if occ > 0.1:
                        # Numerical gradient of obstacle cost
                        delta = RESOLUTION
                        for dx_sign, dy_sign, gx_arr, gy_arr in [
                            (delta, 0, grad_x, None),
                            (-delta, 0, grad_x, None),
                            (0, delta, None, grad_y),
                            (0, -delta, None, grad_y),
                        ]:
                            nr, nc = local_to_grid(band_x[i] + dx_sign, band_y[i] + dy_sign)
                            if 0 <= nr < GRID_RES and 0 <= nc < GRID_RES:
                                neighbor_occ = costmap[nr, nc]
                            else:
                                neighbor_occ = 1.0
                            if dx_sign != 0 and gx_arr is not None:
                                grad_x[i] += self.w_obstacle * (neighbor_occ - occ) / delta * (1 if dx_sign > 0 else -1)
                            if dy_sign != 0 and gy_arr is not None:
                                grad_y[i] += self.w_obstacle * (neighbor_occ - occ) / delta * (1 if dy_sign > 0 else -1)

                # --- Path length gradient (pull toward neighbors) ---
                grad_x[i] += self.w_path * ((band_x[i-1] - band_x[i]) + (band_x[i+1] - band_x[i]))
                grad_y[i] += self.w_path * ((band_y[i-1] - band_y[i]) + (band_y[i+1] - band_y[i]))

                # --- Smoothness gradient (minimize curvature) ---
                accel_x = band_x[i-1] - 2*band_x[i] + band_x[i+1]
                accel_y = band_y[i-1] - 2*band_y[i] + band_y[i+1]
                grad_x[i] += self.w_smooth * accel_x
                grad_y[i] += self.w_smooth * accel_y

            # Update band (keep start and end fixed)
            band_x[1:-1] -= self.lr * grad_x[1:-1]
            band_y[1:-1] -= self.lr * grad_y[1:-1]

        # Follow first optimized waypoint with pure pursuit
        # Find a lookahead point along the band
        lookahead_m = 0.4
        target_x, target_y = band_x[1], band_y[1]
        cumulative_dist = 0.0
        for i in range(1, n):
            dx = band_x[i] - band_x[i-1]
            dy = band_y[i] - band_y[i-1]
            seg_len = math.sqrt(dx**2 + dy**2)
            cumulative_dist += seg_len
            if cumulative_dist >= lookahead_m:
                target_x, target_y = band_x[i], band_y[i]
                break

        # Convert to velocity commands
        angle = math.atan2(target_y, target_x)
        w_cmd = np.clip(angle / (math.pi / 3), -1.0, 1.0)
        alignment = max(0.0, math.cos(angle))
        v_cmd = alignment * 0.8

        # Slow down if first band segment has obstacle
        row1, col1 = local_to_grid(band_x[1], band_y[1])
        if 0 <= row1 < GRID_RES and 0 <= col1 < GRID_RES:
            if costmap[row1, col1] > 0.4:
                v_cmd *= 0.3

        return float(np.clip(v_cmd, -1.0, 1.0)), float(np.clip(w_cmd, -1.0, 1.0))


def generate_trial_params(num_trials, eval_seed, distances, tile_size=60.0, border=3.0):
    """Pre-generate deterministic trial parameters per distance."""
    rng = np.random.default_rng(eval_seed)
    safe_min = border
    safe_max = tile_size - border

    all_params = {}
    for dist in distances:
        positions = np.zeros((num_trials, 2), dtype=np.float32)
        yaws = np.zeros(num_trials, dtype=np.float32)
        angles = np.zeros(num_trials, dtype=np.float32)

        for i in range(num_trials):
            for _ in range(10000):
                x = rng.uniform(safe_min, safe_max)
                y = rng.uniform(safe_min, safe_max)
                a = rng.uniform(0, 2 * math.pi)
                gx = x + dist * math.cos(a)
                gy = y + dist * math.sin(a)
                if safe_min <= gx <= safe_max and safe_min <= gy <= safe_max:
                    positions[i] = [x, y]
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
    """
    import omni.physx as _physx
    physx_query = _physx.get_physx_scene_query_interface()

    num_rays = 8
    ray_angles = [i * 2 * math.pi / num_rays for i in range(num_rays)]
    ray_height = 0.3

    ox = env_origin[0].item()
    oy = env_origin[1].item()
    safe_min = border
    safe_max = tile_size - border

    rng = np.random.default_rng(12345)

    def is_position_clear(tile_x, tile_y):
        wx = ox + tile_x - tile_size / 2.0
        wy = oy + tile_y - tile_size / 2.0

        hit = physx_query.raycast_closest((wx, wy, 2.0), (0.0, 0.0, -1.0), 3.0, False)
        if hit["hit"] and hit["position"][2] > 0.1:
            return False

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


def yaw_to_quat_batch(yaws):
    half = yaws * 0.5
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(yaws)
    return torch.stack([w, zeros, zeros, z], dim=-1)


def setup_trial(env, env_id, trial_idx, target_dist, trial_params, device):
    """Override robot spawn position and goal for a specific trial."""
    robot = env.scene["robot"]
    env_origin = env.scene.env_origins[env_id]

    tile_pos = torch.tensor(trial_params["spawn_positions"][trial_idx], device=device)
    spawn_pos = env_origin.clone().to(device)
    spawn_pos[0] += tile_pos[0] - 30.0
    spawn_pos[1] += tile_pos[1] - 30.0
    spawn_pos[2] += 0.5

    yaw = torch.tensor(trial_params["spawn_yaws"][trial_idx], device=device)
    quat = yaw_to_quat_batch(yaw.unsqueeze(0)).squeeze(0)

    root_state = torch.zeros(1, 13, device=device)
    root_state[0, :3] = spawn_pos
    root_state[0, 3:7] = quat
    robot.write_root_state_to_sim(root_state, torch.tensor([env_id], device=device))

    angle = trial_params["goal_angles"][trial_idx]
    goal_x = spawn_pos[0].item() + target_dist * math.cos(angle)
    goal_y = spawn_pos[1].item() + target_dist * math.sin(angle)
    env.goal_pos[env_id, 0] = goal_x
    env.goal_pos[env_id, 1] = goal_y
    env.goal_pos[env_id, 2] = env_origin[2].item()

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


def run_classical_evaluation(env, wrapped_env, planner, planner_name, device,
                             distances, all_trial_params, num_trials):
    """Run fixed-trial evaluation with a classical planner."""
    num_envs = env.num_envs
    robot = env.scene["robot"]
    eval_goal_threshold = env.cfg.goal_threshold + 0.1
    num_batches = math.ceil(num_trials / num_envs)

    print(f"\n[INFO] === Classical Planner Evaluation: {planner_name.upper()} ===")
    print(f"[INFO] Distances: {distances}m")
    print(f"[INFO] Trials per distance: {num_trials}")
    print(f"[INFO] Parallel envs: {num_envs}, Batches per distance: {num_batches}")

    scenario_results = {}

    for target_dist in distances:
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {target_dist:.0f}m  ({num_trials} trials) — {planner_name.upper()}")
        print(f"{'='*70}")

        dist_params = all_trial_params[target_dist]

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

            # Reset
            wrapped_env.reset()

            env_to_trial = {}
            with torch.inference_mode():
                for local_i in range(batch_size):
                    trial_idx = batch_start + local_i
                    setup_trial(env, local_i, trial_idx, target_dist, dist_params, device)
                    env_to_trial[local_i] = trial_idx

            # Reset RRT* path cache for this batch
            if hasattr(planner, 'reset'):
                for i in range(num_envs):
                    planner.reset(i)

            # Zero-action step to refresh observations
            with torch.inference_mode():
                zero_actions = torch.zeros(num_envs, 2, device=device)
                obs, _, _, _, _ = wrapped_env.step(zero_actions)

            ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
            ep_last_pos = robot.data.root_pos_w[:, :3].detach().clone()
            ep_path_length = torch.zeros(num_envs, device=device)

            initial_goal_dist = torch.norm(
                env.goal_pos[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
            ).detach()

            env_done = [False] * num_envs
            for i in range(batch_size, num_envs):
                env_done[i] = True

            step_count = 0
            max_steps = 10000

            while not all(env_done[:batch_size]) and step_count < max_steps and simulation_app.is_running():
                with torch.inference_mode():
                    pre_robot_pos = robot.data.root_pos_w[:, :3].detach().clone()
                    pre_goal_pos = env.goal_pos.clone()
                    pre_stall_timer = env.stall_timer.clone()

                    step_dist = torch.norm(pre_robot_pos - ep_last_pos, dim=-1)
                    ep_path_length += step_dist
                    ep_last_pos[:] = pre_robot_pos

                    # Compute actions for each active env using the planner
                    actions = torch.zeros(num_envs, 2, device=device)

                    # Batch GPU→CPU transfer (one call instead of per-env)
                    all_cmaps = batch_costmaps_to_numpy(env, batch_size)
                    all_pos, all_yaw, all_goals = batch_robot_states(env, batch_size)

                    for i in range(batch_size):
                        if env_done[i]:
                            continue

                        goal_lx, goal_ly = world_to_local(
                            all_pos[i], all_yaw[i], all_goals[i])
                        cmap = all_cmaps[i]

                        if isinstance(planner, (RRTStarPlanner, MPPIPlanner)):
                            v, w = planner.compute_action(cmap, goal_lx, goal_ly, env_id=i)
                        else:
                            v, w = planner.compute_action(cmap, goal_lx, goal_ly)

                        actions[i, 0] = v
                        actions[i, 1] = w

                    obs, rewards, terminated, truncated, infos = wrapped_env.step(actions)
                    ep_steps += 1
                    step_count += 1

                    terminated_flat = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
                    truncated_flat = truncated.squeeze(-1) if truncated.dim() > 1 else truncated
                    done = terminated_flat | truncated_flat

                    for i in range(batch_size):
                        if env_done[i] or not done[i].item():
                            continue

                        trial_idx = env_to_trial[i]
                        robot_pos_i = pre_robot_pos[i, :2]
                        goal_pos_i = pre_goal_pos[i, :2]
                        nav_error = torch.norm(goal_pos_i - robot_pos_i).item()
                        is_truncated = truncated_flat[i].item()
                        was_stalled = pre_stall_timer[i].item() > 2.9

                        goal_reached = nav_error < eval_goal_threshold
                        if goal_reached:
                            outcome = "success"
                        elif is_truncated:
                            outcome = "timeout"
                        elif was_stalled:
                            outcome = "stall"
                        else:
                            outcome = "collision"

                        trial_outcomes[trial_idx] = outcome
                        trial_nav_errors[trial_idx] = nav_error
                        trial_ep_steps[trial_idx] = ep_steps[i].item()
                        trial_path_lengths[trial_idx] = ep_path_length[i].item()
                        trial_goal_dists[trial_idx] = initial_goal_dist[i].item()

                        env_done[i] = True
                        completed_count += 1

                        # Reset RRT* cache for this env
                        if hasattr(planner, 'reset'):
                            planner.reset(i)

            n_success_so_far = sum(1 for o in trial_outcomes if o == "success")
            print(f"  [{target_dist:.0f}m] Batch {batch_idx+1}/{num_batches} done | "
                  f"Completed: {completed_count}/{num_trials} | "
                  f"SR so far: {n_success_so_far/max(completed_count,1):.1%}")

        # Compute metrics
        n_ep = sum(1 for o in trial_outcomes if o is not None)
        if n_ep == 0:
            scenario_results[target_dist] = {
                'n_episodes': 0, 'sr': 0, 'cr': 0, 'str': 0, 'tr': 0,
                'ne': 0, 'spl': 0, 'avg_steps': 0,
            }
            continue

        n_s = sum(1 for o in trial_outcomes if o == "success")
        n_c = sum(1 for o in trial_outcomes if o == "collision")
        n_st = sum(1 for o in trial_outcomes if o == "stall")
        n_t = sum(1 for o in trial_outcomes if o == "timeout")

        nav_errors = np.array(trial_nav_errors[:n_ep])
        goal_dists = np.array(trial_goal_dists[:n_ep])
        path_lengths = np.array(trial_path_lengths[:n_ep])

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
#  6. Results Display & CSV Export
# =====================================================================
def print_results_table(scenario_results, distances, planner_name):
    print("\n" + "=" * 82)
    print(f"  CLASSICAL PLANNER EVALUATION — {planner_name.upper()}")
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


def save_results_csv(all_results, distances, csv_path):
    """Save results for all planners to a single CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Planner', 'Distance_m', 'Trials', 'SR', 'CR', 'StR', 'TR',
            'NE', 'NE_std', 'SPL', 'Avg_Steps'
        ])
        for planner_name, scenario_results in all_results.items():
            for dist in distances:
                r = scenario_results.get(dist, {})
                writer.writerow([
                    planner_name, dist,
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
#  7. Main
# =====================================================================
def main():
    device = "cuda:0"

    # 1. Environment setup
    env_cfg = EvalNavigationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    print(f"[INFO] Setting up evaluation environment ({args_cli.num_envs} envs)...")
    env = NavigationEnv(cfg=env_cfg)
    wrapped_env = wrap_env(env, wrapper="isaaclab")

    # 2. Generate trial parameters (identical to RL eval)
    all_trial_params = generate_trial_params(
        args_cli.num_trials, args_cli.eval_seed,
        args_cli.distances, tile_size=60.0, border=3.0,
    )
    for dist in args_cli.distances:
        params = all_trial_params[dist]
        print(f"[INFO] Distance {dist:.0f}m: {len(params['spawn_positions'])} trials generated")

    # 3. Warmup
    print("[INFO] Running warmup (50 steps)...")
    obs, _ = wrapped_env.reset()
    zero_actions = torch.zeros(args_cli.num_envs, 2, device=device)
    for _ in range(50):
        wrapped_env.step(zero_actions)

    # 3.5. Validate spawn/goal positions against obstacles
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

    # 4. Select planners
    all_planners = {
        "apf": APFPlanner,
        "dwa": DWAPlanner,
        "mppi": MPPIPlanner,
        "teb": TEBPlanner,
    }
    if args_cli.planner == "all":
        planners = {name: cls() for name, cls in all_planners.items()}
    else:
        planners = {args_cli.planner: all_planners[args_cli.planner]()}

    # 5. Run evaluation for each planner
    all_results = {}
    for name, planner in planners.items():
        results = run_classical_evaluation(
            env, wrapped_env, planner, name, device,
            args_cli.distances, all_trial_params, args_cli.num_trials
        )
        all_results[name] = results
        print_results_table(results, args_cli.distances, name)

    # 6. Save CSV
    if args_cli.save_csv:
        save_results_csv(all_results, args_cli.distances, args_cli.save_csv)

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
