import math
import torch
import os

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    ActionTerm,
    ActionTermCfg
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains.trimesh.mesh_terrains_cfg as mesh_cfg
from .terrain_eval import ClearanceCylindersTerrainCfg, ClearanceBoxesTerrainCfg, MixedObstaclesTerrainCfg


@configclass
class HiVLARobotCfg(ArticulationCfg):
    prim_path = "{ENV_REGEX_NS}/robot"
    spawn = UsdFileCfg(
        usd_path="source/extensions/hivla_project/hivla.usd",
        activate_contact_sensors=True,
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=100.0,
        ),
    )
    # lifted 50cm to prevent ground clipping at spawn
    init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5))
    actuators = {
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=1.0e12,
            velocity_limit_sim=1000.0,
            stiffness=0.0,    # velocity control mode
            damping=1.0e7,
        ),
    }


@configclass
class HeliosLidarCfg(RayCasterCfg):
    prim_path = "{ENV_REGEX_NS}/robot/hivla_robot/Helios"
    offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    pattern_cfg = patterns.LidarPatternCfg(
        channels=32,
        vertical_fov_range=(-55.0, 15.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=0.2,
    )
    max_distance = 10.0
    data_type = "distance"
    debug_vis = False
    mesh_prim_paths = ["/World/ground"]


@configclass
class RobotContactSensorCfg(ContactSensorCfg):
    prim_path = "{ENV_REGEX_NS}/robot/hivla_robot/scout_v2/(base_link|.*_wheel_link)"
    history_length = 3
    track_air_time = False
    debug_vis = False


class DifferentialDriveAction(ActionTerm):
    """Skid-steer action term for AgileX Scout 2.0.

    Converts [linear_velocity, angular_velocity] commands into per-wheel
    velocity targets using differential drive kinematics.
    """
    cfg: "DifferentialDriveActionCfg"

    def __init__(self, cfg: "DifferentialDriveActionCfg", env):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self.joint_ids, _ = self._asset.find_joints(self.cfg.joint_names)
        self.num_wheels = len(self.joint_ids)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.num_wheels, device=self.device)

        if isinstance(self.cfg.scale, (tuple, list)):
            if len(self.cfg.scale) == 3:
                self.scale_vals = torch.tensor(self.cfg.scale, device=self.device)
            elif len(self.cfg.scale) == 2:
                val = self.cfg.scale
                self.scale_vals = torch.tensor([val[0], val[0], val[1]], device=self.device)
            else:
                raise ValueError(f"Scale must be tuple of length 2 or 3, got {len(self.cfg.scale)}")
        else:
            val = self.cfg.scale
            self.scale_vals = torch.tensor([val, val, val], device=self.device)

        self.left_wheel_indices = []
        self.right_wheel_indices = []
        all_joint_names = self._asset.joint_names
        for i, j_idx in enumerate(self.joint_ids):
            idx_val = j_idx.item() if isinstance(j_idx, torch.Tensor) else j_idx
            name = all_joint_names[idx_val]
            if "left" in name.lower():
                self.left_wheel_indices.append(i)
            elif "right" in name.lower():
                self.right_wheel_indices.append(i)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        actions = torch.clamp(actions, -1.0, 1.0)

        fwd_max = self.scale_vals[0]
        bwd_max = self.scale_vals[1]
        ang_max = self.scale_vals[2]

        raw_lin = actions[:, 0]
        raw_ang = actions[:, 1]

        final_lin = torch.where(raw_lin > 0, raw_lin * fwd_max, raw_lin * bwd_max)
        final_ang = raw_ang * ang_max

        # v_wheel = (v_linear ± v_angular * wheelbase/2) / wheel_radius
        v_left = (final_lin - final_ang * self.cfg.wheel_base / 2) / self.cfg.wheel_radius
        v_right = (final_lin + final_ang * self.cfg.wheel_base / 2) / self.cfg.wheel_radius

        if self.left_wheel_indices:
            self._processed_actions[:, self.left_wheel_indices] = v_left.unsqueeze(1)
        if self.right_wheel_indices:
            self._processed_actions[:, self.right_wheel_indices] = v_right.unsqueeze(1)

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self.joint_ids)


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    class_type = DifferentialDriveAction
    joint_names: list[str] = []
    wheel_radius: float = 0.165
    wheel_base: float = 0.582
    scale: float | tuple[float, float] | tuple[float, float, float] = (0.5, 0.5, 1.0)


@configclass
class ActionsCfg:
    body_cmd = DifferentialDriveActionCfg(
        asset_name="robot",
        # joint order must match kinematics: left wheels before right
        joint_names=[
            'front_left_wheel_joint', 'rear_left_wheel_joint',
            'front_right_wheel_joint', 'rear_right_wheel_joint'
        ],
        wheel_radius=0.165,
        wheel_base=0.582,
        scale=(0.5, 0.5, 1.0),
    )


@configclass
class PolicyObsGroupCfg(ObservationGroupCfg):
    visual = ObservationTermCfg(func="hivla_project.nav_task:compute_visual_obs")
    state = ObservationTermCfg(func="hivla_project.nav_task:compute_state_obs")
    concatenate_terms = False


@configclass
class ObservationsCfg:
    policy = PolicyObsGroupCfg()


@configclass
class RewardsCfg:
    total_reward = RewardTermCfg(func="hivla_project.nav_task:compute_reward", weight=1.0)


@configclass
class TerminationsCfg:
    collision = TerminationTermCfg(func="hivla_project.nav_task:check_collision_termination")
    reach_goal = TerminationTermCfg(func="hivla_project.nav_task:check_goal_reached")
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    stall = TerminationTermCfg(func="hivla_project.nav_task:check_stall_termination")


@configclass
class EventsCfg:
    pass


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=5.0)

    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()

    decimation = 4
    render_interval = 4
    episode_length_s = 150.0
    curriculum_steps = 0
    goal_threshold = 0.5

    def __post_init__(self):
        # --- Robot & Sensors ---
        self.scene.robot = HiVLARobotCfg()
        self.scene.lidar = HeliosLidarCfg()
        self.scene.contact_sensor = RobotContactSensorCfg()

        # --- Lighting ---
        self.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(2.0, 2.0, 2.0)),
        )

        # --- Terrain ---
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=42,
                curriculum=False,
                size=(60.0, 60.0),
                border_width=2.5,
                num_rows=5,
                num_cols=4,
                sub_terrains={
                    "mixed": MixedObstaclesTerrainCfg(
                        proportion=1.0,
                        min_clearance=1.8,
                        platform_margin=1.0,
                    ),
                },
            )
        )
