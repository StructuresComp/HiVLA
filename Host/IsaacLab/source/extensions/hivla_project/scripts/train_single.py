import argparse
import os
import datetime
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Navigation Agent with SKRL (Single GPU)")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from")
parser.add_argument("--resume_from", type=int, default=0, help="Timestep the checkpoint was saved at")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
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
            visual_obs = data[:, 2:].view(-1, 1, 128, 128)
        feat_visual = self.net_cnn(visual_obs)
        feat_state = self.net_state(state_obs)
        mu = self.net_fusion(torch.cat([feat_visual, feat_state], dim=1))
        return mu, self.log_std_parameter, {}


class NavigationValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.net_state = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.net_fusion = nn.Sequential(
            nn.Linear(9216 + 64, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        data = inputs["states"]
        if isinstance(data, dict):
            state_obs = data["state"]
            visual_obs = data["visual"]
        else:
            state_obs = data[:, 0:2]
            visual_obs = data[:, 2:].view(-1, 1, 128, 128)
        fused = torch.cat([self.net_cnn(visual_obs), self.net_state(state_obs)], dim=1)
        return self.net_fusion(fused), {}


def main():
    BASE_LOG_DIR = "logs/navigation_ppo"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(BASE_LOG_DIR, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"[INFO] Log directory: {experiment_dir}")

    TOTAL_TIMESTEPS = 100000
    remaining_timesteps = TOTAL_TIMESTEPS - args_cli.resume_from

    env_cfg = NavigationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.curriculum_steps = TOTAL_TIMESTEPS
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.seed = 42

    env = NavigationEnv(cfg=env_cfg)
    env = wrap_env(env, wrapper="isaaclab")
    device = env.device

    models = {
        "policy": NavigationPolicy(env.observation_space, env.action_space, device),
        "value": NavigationValue(env.observation_space, env.action_space, device),
    }
    memory = RandomMemory(memory_size=96, num_envs=env.num_envs, device=device)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 96
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["state_preprocessor"] = False
    cfg["entropy_loss_scale"] = 0.01
    cfg["grad_norm_clip"] = 1.0
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1}
    cfg["experiment"] = {
        "directory": experiment_dir,
        "write_interval": 100,
        "checkpoint_interval": 5000,
        "experiment_name": "single_gpu_run",
    }

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args_cli.checkpoint:
        agent.load(args_cli.checkpoint)
        print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint} (resuming from step {args_cli.resume_from})")

    trainer = SequentialTrainer(
        cfg={"timesteps": remaining_timesteps, "headless": args_cli.headless},
        env=env,
        agents=agent,
    )

    print(f"[INFO] Training on {device} with {env.num_envs} environments ({remaining_timesteps} steps remaining)...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    hrs, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    print(f"[INFO] Training completed in {hrs}h {mins}m {secs}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        simulation_app.close()
        os._exit(0)
