import os
import datetime

local_rank = int(os.getenv("LOCAL_RANK", "0"))

import argparse
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Navigation Agent with SKRL (Multi-GPU)")
parser.add_argument("--num_envs", type=int, default=2048)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.device = f"cuda:{local_rank}"
args_cli.distributed = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
        self.net_state = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
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
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    device = f"cuda:{app_launcher.device_id}"
    torch.cuda.set_device(device)

    import torch.distributed as torch_dist

    BASE_LOG_DIR = "logs/navigation_ppo"
    MAX_PATH_LENGTH = 256

    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(BASE_LOG_DIR, timestamp)
        os.makedirs(experiment_dir, exist_ok=True)
        dir_bytes = experiment_dir.encode('utf-8')
        shared_dir_tensor = torch.zeros(MAX_PATH_LENGTH, dtype=torch.uint8, device=device)
        shared_dir_tensor[:len(dir_bytes)] = torch.tensor(list(dir_bytes), device=device)
    else:
        shared_dir_tensor = torch.zeros(MAX_PATH_LENGTH, dtype=torch.uint8, device=device)
        experiment_dir = ""

    if world_size > 1:
        torch_dist.broadcast(shared_dir_tensor, src=0)
    if rank != 0:
        dir_bytes_list = shared_dir_tensor.cpu().numpy().tolist()
        experiment_dir = bytes(dir_bytes_list).strip(b'\x00').decode('utf-8')

    print(f"[INFO] Rank {rank}/{world_size} | log dir: {experiment_dir}")

    TOTAL_TIMESTEPS = 200000

    env_cfg = NavigationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs // world_size
    env_cfg.curriculum_steps = TOTAL_TIMESTEPS
    env_cfg.sim.device = device
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.seed = 42 + rank

    env = NavigationEnv(cfg=env_cfg)
    env = wrap_env(env, wrapper="isaaclab")

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
        "experiment_name": "multigpu_run" if rank == 0 else f"worker_{rank}",
    }

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    trainer = SequentialTrainer(
        cfg={"timesteps": TOTAL_TIMESTEPS, "headless": args_cli.headless},
        env=env,
        agents=agent,
    )

    print(f"[INFO] Rank {rank}: training on {device} with {env.num_envs} envs...")
    trainer.train()

    if world_size > 1 and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[INFO] Training finished. Force exiting to prevent Isaac Sim hang...")
        os._exit(0)
