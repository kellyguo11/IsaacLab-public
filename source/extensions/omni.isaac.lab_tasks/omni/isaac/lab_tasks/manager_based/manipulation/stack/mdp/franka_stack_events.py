# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import random
import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import AssetBase
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    new_intensity = random.uniform(intensity_range[0], intensity_range[1])

    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)
