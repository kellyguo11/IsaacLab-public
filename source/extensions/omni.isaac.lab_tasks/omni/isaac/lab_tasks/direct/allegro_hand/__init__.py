# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Allegro Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents
from .allegro_hand_env_cfg import AllegroHandEnvCfg, AllegroHandOpenAIEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Allegro-Hand-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AllegroHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.AllegroHandPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Allegro-Hand-OpenAI-FF-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AllegroHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_ff_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.AllegroHandAsymFFPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Allegro-Hand-OpenAI-LSTM-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.inhand_manipulation:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AllegroHandOpenAIEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
