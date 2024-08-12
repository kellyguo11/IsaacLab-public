# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leap Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents
from .leap_hand_env_cfg import LeapHandEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Repose-Cube-Leap-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeapHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
