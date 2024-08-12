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


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Users/kellyg@nvidia.com/LeapHandV2/New0812/leapv2_right_full_res/leapv2_right_full_res.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=3.0, damping=0.1),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        rot=(0, 0, 1, 0),
    ),
    actuators={
        "mcps": ImplicitActuatorCfg(
            joint_names_expr=[".*mcps"],
            effort_limit=20,
            velocity_limit=100.0,
            stiffness=3,
            damping=0.1,
            friction=0.01,
            armature=0.002,
        ),
        "mcpf": ImplicitActuatorCfg(
            joint_names_expr=[".*mcpf"],
            effort_limit=20,
            velocity_limit=100,
            stiffness=3,
            damping=0.1,
            friction=0.01,
            armature=0.002,
        ),
        "pip": ImplicitActuatorCfg(
            joint_names_expr=[".*pip"],
            effort_limit=20,
            velocity_limit=100.0,
            stiffness=3,
            damping=0.1,
            friction=0.01,
            armature=0.002,
        ),
        "dip": ImplicitActuatorCfg(
            joint_names_expr=[".*dip"],
            effort_limit=20,
            velocity_limit=100.0,
            stiffness=3,
            damping=0.1,
            friction=0.01,
            armature=0.002,
        ),
        "palm_4_finger": ImplicitActuatorCfg(
            joint_names_expr=["palm_4_finger"],
            effort_limit=20,
            velocity_limit=100.0,
            stiffness=3,
            damping=0.1,
            friction=0.01,
            armature=0.002,
        ),
        "palm_thumb": ImplicitActuatorCfg(
            joint_names_expr=["palm_thumb"],
            effort_limit=20,
            velocity_limit=100.0,
            stiffness=3,
            damping=0.1,
            friction=0.01,
            armature=0.002,
        ),
        "thumb_ip": ImplicitActuatorCfg(
            joint_names_expr=["thumb_ip"],
            effort_limit=20,
            velocity_limit=100.0,
            stiffness=3,
            damping=0.1,
            friction=0.01,
            armature=0.002,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Leap Hand robot."""
