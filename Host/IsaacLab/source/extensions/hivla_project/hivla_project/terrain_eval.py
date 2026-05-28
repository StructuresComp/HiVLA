"""
Custom terrain generation with minimum clearance between obstacles.

Provides modified versions of IsaacLab's repeated-object terrain functions
that enforce a minimum gap between obstacles via rejection sampling.
Used for creating dedicated, reproducible evaluation environments.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.utils import make_plane, make_cylinder, make_box
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
import isaaclab.terrains.trimesh.mesh_terrains_cfg as mesh_cfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


def repeated_objects_terrain_clearance(
    difficulty: float,
    cfg: mesh_cfg.MeshRepeatedObjectsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate terrain with repeated objects and enforced minimum clearance.

    Same as IsaacLab's ``repeated_objects_terrain`` but adds rejection sampling
    to guarantee a minimum gap between obstacle surfaces.
    """
    min_clearance = getattr(cfg, "min_clearance", 3.0)

    if isinstance(cfg.object_type, str):
        object_func = getattr(mesh_utils_terrains, f"make_{cfg.object_type}", None)
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"object_type must be a string or callable. Got: {cfg.object_type}")

    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    platform_height = cfg.platform_height if cfg.platform_height >= 0.0 else height

    if isinstance(cfg, (ClearanceBoxesTerrainCfg, mesh_cfg.MeshRepeatedBoxesTerrainCfg)):
        length = cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0])
        width = cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1])
        object_kwargs = {
            "length": length,
            "width": width,
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
        effective_radius = max(length, width) / 2.0
    elif isinstance(cfg, (ClearanceCylindersTerrainCfg, mesh_cfg.MeshRepeatedCylindersTerrainCfg)):
        radius = cp_0.radius + difficulty * (cp_1.radius - cp_0.radius)
        object_kwargs = {
            "radius": radius,
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
        effective_radius = radius
    else:
        raise ValueError(f"Unsupported terrain config type: {type(cfg)}")

    min_center_dist = min_clearance + 2.0 * effective_radius

    platform_margin = getattr(cfg, "platform_margin", 0.1)
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * platform_height))
    platform_half = cfg.platform_width / 2
    platform_corners = np.asarray([
        [origin[0] - platform_half - platform_margin, origin[1] - platform_half - platform_margin],
        [origin[0] + platform_half + platform_margin, origin[1] + platform_half + platform_margin],
    ])

    placed_centers = []
    for _ in range(num_objects):
        for _ in range(1000):
            x = np.random.uniform(0, cfg.size[0])
            y = np.random.uniform(0, cfg.size[1])
            if (platform_corners[0, 0] <= x <= platform_corners[1, 0] and
                    platform_corners[0, 1] <= y <= platform_corners[1, 1]):
                continue
            if placed_centers:
                centers_arr = np.array(placed_centers)
                dists = np.sqrt((centers_arr[:, 0] - x) ** 2 + (centers_arr[:, 1] - y) ** 2)
                if np.any(dists < min_center_dist):
                    continue
            placed_centers.append((x, y))
            break

    n_placed = len(placed_centers)
    if n_placed < num_objects:
        print(f"[terrain_eval] Placed {n_placed}/{num_objects} objects "
              f"(min_clearance={min_clearance}m, tile={cfg.size[0]}x{cfg.size[1]}m)")

    meshes_list = []
    for cx, cy in placed_centers:
        abs_noise = np.random.uniform(cfg.abs_height_noise[0], cfg.abs_height_noise[1])
        rel_noise = np.random.uniform(cfg.rel_height_noise[0], cfg.rel_height_noise[1])
        ob_height = height * rel_noise + abs_noise
        if ob_height > 0.0:
            center = np.array([cx, cy, 0.0])
            meshes_list.append(object_func(center=center, height=ob_height, **object_kwargs))

    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    dim = (cfg.platform_width, cfg.platform_width, 0.5 * platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.25 * platform_height)
    meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    return meshes_list, origin


@configclass
class ClearanceCylindersTerrainCfg(mesh_cfg.MeshRepeatedCylindersTerrainCfg):
    """Cylinder terrain with enforced minimum clearance between obstacles."""
    function = repeated_objects_terrain_clearance
    min_clearance: float = 3.0


@configclass
class ClearanceBoxesTerrainCfg(mesh_cfg.MeshRepeatedBoxesTerrainCfg):
    """Box terrain with enforced minimum clearance between obstacles."""
    function = repeated_objects_terrain_clearance
    min_clearance: float = 3.0


def mixed_objects_terrain_clearance(
    difficulty: float,
    cfg: "MixedObstaclesTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate terrain with both cylinders and boxes, enforced min clearance."""
    min_clearance = cfg.min_clearance

    cylinder_eff_radius = cfg.cylinder_radius
    box_eff_radius = max(cfg.box_length, cfg.box_width) / 2.0
    max_eff_radius = max(cylinder_eff_radius, box_eff_radius)
    min_center_dist = min_clearance + 2.0 * max_eff_radius

    platform_margin = cfg.platform_margin
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))
    platform_half = cfg.platform_width / 2
    platform_corners = np.asarray([
        [origin[0] - platform_half - platform_margin, origin[1] - platform_half - platform_margin],
        [origin[0] + platform_half + platform_margin, origin[1] + platform_half + platform_margin],
    ])

    obj_queue = [("cylinder",)] * cfg.num_cylinders + [("box",)] * cfg.num_boxes
    total_objects = len(obj_queue)

    placed = []
    for (obj_type,) in obj_queue:
        for _ in range(1000):
            x = np.random.uniform(0, cfg.size[0])
            y = np.random.uniform(0, cfg.size[1])
            if (platform_corners[0, 0] <= x <= platform_corners[1, 0] and
                    platform_corners[0, 1] <= y <= platform_corners[1, 1]):
                continue
            if placed:
                centers = np.array([(p[0], p[1]) for p in placed])
                dists = np.sqrt((centers[:, 0] - x) ** 2 + (centers[:, 1] - y) ** 2)
                if np.any(dists < min_center_dist):
                    continue
            placed.append((x, y, obj_type))
            break

    n_placed = len(placed)
    if n_placed < total_objects:
        n_cyl = sum(1 for p in placed if p[2] == "cylinder")
        n_box = sum(1 for p in placed if p[2] == "box")
        print(f"[terrain_eval] Mixed terrain: placed {n_cyl} cylinders + {n_box} boxes "
              f"= {n_placed}/{total_objects} (min_clearance={min_clearance}m)")

    meshes_list = []
    for cx, cy, obj_type in placed:
        center = np.array([cx, cy, 0.0])
        if obj_type == "cylinder":
            meshes_list.append(make_cylinder(
                radius=cfg.cylinder_radius, height=cfg.cylinder_height, center=center,
            ))
        else:
            meshes_list.append(make_box(
                length=cfg.box_length, width=cfg.box_width, height=cfg.box_height, center=center,
            ))

    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    dim = (cfg.platform_width, cfg.platform_width, 0.01)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.005)
    meshes_list.append(
        trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    )

    return meshes_list, origin


@configclass
class MixedObstaclesTerrainCfg(SubTerrainBaseCfg):
    """Single terrain tile with both cylinders and boxes, enforced min clearance."""

    function = mixed_objects_terrain_clearance

    min_clearance: float = 3.0
    platform_width: float = 3.0
    platform_margin: float = 0.5

    num_cylinders: int = 80
    cylinder_height: float = 2.0
    cylinder_radius: float = 0.25

    num_boxes: int = 60
    box_height: float = 2.0
    box_length: float = 1.5
    box_width: float = 0.75
