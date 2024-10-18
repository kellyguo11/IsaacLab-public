# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, AssetBase
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[env_ids, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    # Sample new light intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])

    # Set light intensity to light prim
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = []

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Sample a pose
            for j in range(max_sample_tries):
                sample = [random.uniform(range[0], range[1]) for range in range_list]

                # Accept pose if it is the first one, or if reached max num tries
                if len(pose_list) == 0 or j == max_sample_tries - 1:
                    pose_list.append(sample)
                    # Write pose to simulation
                    pose_tensor = torch.tensor([pose_list[i]], device=env.device)
                    positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
                    orientations = math_utils.quat_from_euler_xyz(
                        pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
                    )
                    asset.write_root_pose_to_sim(
                        torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
                    )
                    break

                # Check if pose of object of sufficiently far away from all other objects
                separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
                if False not in separation_check:
                    pose_list.append(sample)
                    # Write pose to simulation
                    pose_tensor = torch.tensor([pose_list[i]], device=env.device)
                    positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
                    orientations = math_utils.quat_from_euler_xyz(
                        pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
                    )
                    asset.write_root_pose_to_sim(
                        torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
                    )
                    break
