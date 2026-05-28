import os
import argparse
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Navigation Agent (Multi-GPU checkpoint)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.models.torch import GaussianMixin, Model
from skrl.envs.wrappers.torch import wrap_env

from hivla_project.nav_env import NavigationEnvCfg
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
        self.net_state = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
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

    env_cfg = NavigationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.sim.render_interval = 1

    print(f"[INFO] Setting up environment with {args_cli.num_envs} instances...")
    env = NavigationEnv(cfg=env_cfg)
    env = wrap_env(env, wrapper="isaaclab")

    policy = NavigationPolicy(env.observation_space, env.action_space, device)
    models = {"policy": policy}

    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    policy_state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint

    clean_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in policy_state_dict.items()
    }
    policy.load_state_dict(clean_state_dict)
    print("[INFO] Successfully loaded policy weights.")

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = False

    agent = PPO(
        models=models,
        memory=None,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    agent.models["policy"].eval()

    print("[INFO] Starting simulation loop...")
    obs, _ = env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions, _, _ = agent.models["policy"].compute({"states": obs}, role="policy")
            obs, rewards, terminated, truncated, infos = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
