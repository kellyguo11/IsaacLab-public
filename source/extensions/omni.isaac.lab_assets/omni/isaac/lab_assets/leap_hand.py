# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Allegro Hand robots from Wonik Robotics.

The following configurations are available:

* :obj:`ALLEGRO_HAND_CFG`: Allegro Hand with implicit actuator model.

Reference:

* https://www.wonikrobotics.com/robot-hand

"""


import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"https://isaac-dev.ov.nvidia.com/omni/web3/omniverse://isaac-dev.ov.nvidia.com/Users/kellyg@nvidia.com/LeapHandV2/leap_hand_v2_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            # retain_accelerations=False,
            # enable_gyroscopic_forces=False,
            # angular_damping=0.01,
            # max_linear_velocity=1000.0,
            # max_angular_velocity=64 / math.pi * 180.0,
            # max_depenetration_velocity=1000.0,
            # max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        rot=(0, 0, 1, 0),
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
            armature=0.002
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Allegro Hand robot."""
