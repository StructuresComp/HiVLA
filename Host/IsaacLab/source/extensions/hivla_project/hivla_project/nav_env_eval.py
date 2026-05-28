"""
Evaluation Environment Configuration for Point-Goal Navigation.

Dedicated test environment with:
  - Single 60m x 60m terrain tile (shared by all envs)
  - Both cylinders and boxes on the same tile
  - Minimum 2m clearance between obstacles
  - Fixed seed for reproducibility
  - No curriculum (static difficulty)
"""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg

from .nav_env import (
    NavigationEnvCfg,
    HiVLARobotCfg,
    HeliosLidarCfg,
    RobotContactSensorCfg,
)
from .terrain_eval import MixedObstaclesTerrainCfg


@configclass
class EvalNavigationEnvCfg(NavigationEnvCfg):
    """
    Environment configuration for evaluation.

    Differences from training:
      - Single 60m x 60m tile (all envs share the same terrain)
      - Mixed obstacles: 80 cylinders + 60 boxes on the same tile
      - 2m minimum clearance between obstacles
      - No curriculum
      - Fixed seed=42 for reproducibility
    """

    def __post_init__(self):
        # Robot & Sensors (same as training)
        self.scene.robot = HiVLARobotCfg()
        self.scene.lidar = HeliosLidarCfg()
        self.scene.contact_sensor = RobotContactSensorCfg()

        # Lighting
        self.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(2.0, 2.0, 2.0)),
        )

        # Evaluation terrain: single 60m x 60m tile, mixed obstacles
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=10,
                curriculum=False,
                size=(60.0, 60.0),
                border_width=2.5,
                num_rows=1,
                num_cols=1,

                sub_terrains={
                    "eval_mixed": MixedObstaclesTerrainCfg(
                        proportion=1.0,
                        min_clearance=2.0,
                    ),
                },
            ),
        )
