from .nav_env import NavigationEnvCfg, HiVLARobotCfg
from .nav_env_eval import EvalNavigationEnvCfg
from .nav_task import NavigationEnv
from .core_map import BatchGPULocalCostmapCore
from .terrain_eval import (
    ClearanceCylindersTerrainCfg,
    ClearanceBoxesTerrainCfg,
    MixedObstaclesTerrainCfg,
)

import gymnasium as gym

gym.register(
    id="Isaac-Navigation-HiVLA-v0",
    entry_point="hivla_project.nav_task:NavigationEnv",
    disable_env_checker=True,
    kwargs={"cfg": NavigationEnvCfg()},
)