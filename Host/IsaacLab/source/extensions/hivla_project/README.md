# hivla_project — RL Navigation

Scout 2.0 robot point-goal navigation using Isaac Lab (PPO).

---

## Project Structure

```
hivla_project/
├── hivla.usd                  # Robot USD (referenced by nav_env.py)
├── checkpoint/best_agent.pt   # Saved checkpoint
├── hivla_project/             # Importable package
│   ├── nav_env.py             # Robot & sensor definitions
│   ├── nav_task.py            # Task: reward, termination, curriculum
│   ├── nav_env_eval.py        # Eval-mode environment
│   ├── core_map.py            # GPU costmap implementation
│   └── terrain_eval.py        # Custom terrain with obstacle clearance
└── scripts/                   # Execution entry points
    ├── train_single.py
    ├── train_multi.py
    ├── test_single.py
    ├── test_multi.py
    ├── evaluate_policy.py
    └── evaluate_classical.py
```

---

## 1. Setup

```bash
./isaaclab.sh -p -m pip install -e source/extensions/hivla_project
```

---

## 2. Training

### Single GPU

```bash
./isaaclab.sh -p source/extensions/hivla_project/scripts/train_single.py \
  --num_envs 1024 --headless
```

### Multi GPU (2× GPU, distributed)

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 \
  ./isaaclab.sh -p -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 \
  source/extensions/hivla_project/scripts/train_multi.py \
  --headless --num_envs 2048
```

Logs are saved to `logs/navigation_ppo/<timestamp>/`.

```bash
# Monitor training
tensorboard --logdir logs/navigation_ppo
```

---

## 3. Inference (visual check)

```bash
# Single GPU
./isaaclab.sh -p source/extensions/hivla_project/scripts/test_single.py \
  --checkpoint logs/navigation_ppo/<run>/single_gpu_run/checkpoints/agent_50000.pt

# Multi GPU
./isaaclab.sh -p source/extensions/hivla_project/scripts/test_multi.py \
  --checkpoint logs/navigation_ppo/<run>/multigpu_run/checkpoints/agent_50000.pt
```

---

## 4. Evaluation

### 4-1. Policy Evaluation (SR / CR / StR / TR / NE / SPL)

```bash
./isaaclab.sh -p source/extensions/hivla_project/scripts/evaluate_policy.py \
  --checkpoint logs/navigation_ppo/<run>/checkpoints/best_agent.pt \
  --export_jit \
  --headless \
  --num_trials 512
```

Evaluates at 4 goal distances (5 m / 10 m / 15 m / 20 m), 512 trials each.

| Metric | Description |
|--------|-------------|
| SR     | Success Rate |
| CR     | Collision Rate |
| StR    | Stuck Rate |
| TR     | Timeout Rate |
| NE     | Navigation Error (m) |
| SPL    | Success weighted by Path Length |

### 4-2. Classical Baseline Evaluation

Run all planners sequentially (slow):

```bash
./isaaclab.sh -p source/extensions/hivla_project/scripts/evaluate_classical.py \
  --planner all --headless --save_csv classical_results.csv
```

Or run in parallel across terminals for speed:

```bash
# Terminal 1 — GPU 0
./isaaclab.sh -p source/extensions/hivla_project/scripts/evaluate_classical.py \
  --planner apf --headless --save_csv results_apf.csv

# Terminal 2 — GPU 0 (APF is lightweight, can share)
./isaaclab.sh -p source/extensions/hivla_project/scripts/evaluate_classical.py \
  --planner teb --headless --save_csv results_teb.csv

# Terminal 3 — GPU 1
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p source/extensions/hivla_project/scripts/evaluate_classical.py \
  --planner dwa --headless --save_csv results_dwa.csv

# Terminal 4 — GPU 1
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p source/extensions/hivla_project/scripts/evaluate_classical.py \
  --planner mppi --headless --save_csv results_mppi.csv
```

---

## 5. Results

**FIXED-TRIAL POINT-GOAL NAVIGATION EVALUATION**  
Checkpoint: `hivla_project/checkpoint/best_agent.pt`  
Trials per distance: 512 | Seed: 42

### Overall

| Distance | Trials | SR ↑ | CR ↓ | StR ↓ | TR ↓ | NE ↓ | SPL ↑ |
|----------|--------|------|------|-------|------|------|-------|
| 5        | 512    | 74.2% | 6.2% | 1.0% | 18.6% | 0.39m | 0.509 |
| 10       | 512    | 88.5% | 5.1% | 0.6% |  5.9% | 0.31m | 0.777 |
| 15       | 512    | 85.0% | 7.0% | 0.0% |  8.0% | 0.63m | 0.774 |
| 20       | 512    | 89.3% | 6.2% | 0.4% |  4.1% | 0.66m | 0.821 |
| **OVERALL** | **2048** | **87.2%** | **5.3%** | **0.1%** | **7.4%** | **0.44m** | **0.742** |

| Method  |  SR ↑  |  CR ↓  |  TR ↓  |
|---------|--------|--------|--------|
| APF     | 47.4%  |  8.6%  | 43.9%  |
| DWA     | 17.8%  | 79.6%  | **2.6%** |
| MPPI    |  2.5%  | 91.2%  |  6.3%  |
| TEB     | 47.2%  | 43.6%  |  9.2%  |
| Ours    | **87.2%** | **5.3%** |  7.5%  |

### Per-distance Breakdown (SR / CR / TR, %)

<table>
  <thead>
    <tr>
      <th>Distance</th>
      <th colspan="3" align="center">5m</th>
      <th colspan="3" align="center">10m</th>
      <th colspan="3" align="center">15m</th>
      <th colspan="3" align="center">20m</th>
    </tr>
    <tr>
      <th>Method</th>
      <th>SR↑</th><th>CR↓</th><th>TR↓</th>
      <th>SR↑</th><th>CR↓</th><th>TR↓</th>
      <th>SR↑</th><th>CR↓</th><th>TR↓</th>
      <th>SR↑</th><th>CR↓</th><th>TR↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>APF</td>
      <td>60.6%</td><td><strong>5.9%</strong></td><td>33.6%</td>
      <td>48.4%</td><td>10.0%</td><td>41.6%</td>
      <td>41.4%</td><td>10.0%</td><td>48.6%</td>
      <td>39.3%</td><td>8.8%</td><td>51.9%</td>
    </tr>
    <tr>
      <td>DWA</td>
      <td>17.6%</td><td>75.8%</td><td>6.6%</td>
      <td>16.2%</td><td>82.8%</td><td><strong>1.0%</strong></td>
      <td>21.1%</td><td>78.1%</td><td><strong>0.8%</strong></td>
      <td>16.2%</td><td>81.8%</td><td><strong>2.0%</strong></td>
    </tr>
    <tr>
      <td>MPPI</td>
      <td>3.7%</td><td>90.0%</td><td><strong>6.3%</strong></td>
      <td>2.7%</td><td>91.6%</td><td>5.7%</td>
      <td>1.6%</td><td>91.6%</td><td>6.8%</td>
      <td>2.0%</td><td>91.4%</td><td>6.6%</td>
    </tr>
    <tr>
      <td>TEB</td>
      <td>65.6%</td><td>24.0%</td><td>10.4%</td>
      <td>51.0%</td><td>40.6%</td><td>8.4%</td>
      <td>41.0%</td><td>50.4%</td><td>8.6%</td>
      <td>31.1%</td><td>59.4%</td><td>9.6%</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td><strong>83.8%</strong></td><td>6.4%</td><td>9.8%</td>
      <td><strong>88.1%</strong></td><td><strong>4.1%</strong></td><td>7.8%</td>
      <td><strong>90.0%</strong></td><td><strong>5.1%</strong></td><td>4.9%</td>
      <td><strong>86.7%</strong></td><td><strong>5.5%</strong></td><td>7.8%</td>
    </tr>
  </tbody>
</table>

---

## 6. ROS 2 / Visualization (optional)

```bash
# ZED X camera driver (sim mode)
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx sim_mode:=true

# RViz2
ros2 launch zed_display_rviz2 display_zed_cam.launch.py camera_model:=zedx sim_mode:=true

# TF frame alignment (ZED_X → zed_camera_link)
ros2 run tf2_ros static_transform_publisher \
  --frame-id ZED_X --child-frame-id zed_camera_link \
  --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1

# Keyboard teleoperation
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
