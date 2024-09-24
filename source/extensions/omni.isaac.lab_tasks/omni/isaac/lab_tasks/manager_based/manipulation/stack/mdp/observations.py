# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def cube_positions_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """The position of the cubes in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    cube_1_pos_w = cube_1.data.root_pos_w[:, :3]
    cube_1_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], cube_1_pos_w
    )
    cube_2_pos_w = cube_2.data.root_pos_w[:, :3]
    cube_2_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], cube_2_pos_w
    )
    cube_3_pos_w = cube_3.data.root_pos_w[:, :3]
    cube_3_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], cube_3_pos_w
    )

    return torch.cat((cube_1_pos_b, cube_2_pos_b, cube_3_pos_b), dim=1)
