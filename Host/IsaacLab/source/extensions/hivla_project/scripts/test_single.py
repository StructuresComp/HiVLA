import os
import sys
import argparse
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Navigation Agent (Single GPU checkpoint)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.models.torch import GaussianMixin, Model
from skrl.envs.wrappers.torch import wrap_env

from hivla_project.nav_env_eval import EvalNavigationEnvCfg
from hivla_project.nav_task import NavigationEnv


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


def main():
    device = "cuda:0"

    env_cfg = EvalNavigationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.sim.render_interval = 1

    print(f"[INFO] Setting up environment with {args_cli.num_envs} instances...")
    raw_env = NavigationEnv(cfg=env_cfg)
    env = wrap_env(raw_env, wrapper="isaaclab")

    policy = NavigationPolicy(env.observation_space, env.action_space, device)
    policy.eval()
    policy.to(device)

    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        raw_state_dict = checkpoint["policy"]
    else:
        raw_state_dict = checkpoint

    clean_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw_state_dict.items()
    }
    try:
        policy.load_state_dict(clean_state_dict, strict=True)
        print("[INFO] Successfully loaded policy weights.")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        sys.exit(1)

    print("[INFO] Starting simulation loop... (Press Ctrl+C to stop)")
    obs, _ = env.reset()
    step_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions, _, _ = policy.compute({"states": obs}, role="policy")
            obs, rewards, terminated, truncated, infos = env.step(actions)
            if step_count % 10 == 0:
                raw_env._save_debug_costmap(step_count, env_id=0)
            step_count += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        simulation_app.close()
