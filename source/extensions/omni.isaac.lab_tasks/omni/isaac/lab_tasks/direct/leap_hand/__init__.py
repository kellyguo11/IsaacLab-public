# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leap Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents
from .leap_hand_env_cfg import LeapHandEnvCfg, LeapHandEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Leap-Hand-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leap_hand:LeapHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeapHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
