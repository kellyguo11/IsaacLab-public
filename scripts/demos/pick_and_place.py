# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Automated Pick and Place for Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.pick_and_place import PICK_AND_PLACE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
    SurfaceGripper,
    SurfaceGripperCfg,
)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class PickAndPlaceEnvCfg(DirectRLEnvCfg):
    """Example configuration for a PickAndPlace robot using suction-cups.

    This example follows what would be typically done in a DirectRL pipeline.
    """

    # env
    decimation = 4
    episode_length_s = 240.0
    action_space = 4
    observation_space = 6
    state_space = 0

    # Simulation cfg. Note that we are forcing the simulation to run on CPU.
    # This is because the surface gripper API is only supported on CPU backend for now.
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, device=args_cli.device, render_interval=decimation, use_fabric=True)
    debug_vis = True

    # robot
    robot_cfg: ArticulationCfg = PICK_AND_PLACE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    x_dof_name = "x_axis"
    y_dof_name = "y_axis"
    z_dof_name = "z_axis"

    # We add a cube to pick-up
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # Surface Gripper, the prim_expr need to point to a unique surface gripper per environment.
    gripper = SurfaceGripperCfg(
        prim_path="/World/envs/env_.*/Robot/picker_head/SurfaceGripper",
        max_grip_distance=0.1,
        shear_force_limit=500.0,
        coaxial_force_limit=500.0,
        retry_interval=0.2,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=12.0, replicate_physics=True)

    # reset logic
    # Initial position of the robot
    initial_x_pos_range = [-2.0, 2.0]
    initial_y_pos_range = [-2.0, 2.0]
    initial_z_pos_range = [0.0, 0.5]

    # Initial position of the cube
    initial_object_x_pos_range = [-2.0, 2.0]
    initial_object_y_pos_range = [-2.0, -0.5]
    initial_object_z_pos = 0.2

    # Target position of the cube
    target_x_pos_range = [-2.0, 2.0]
    target_y_pos_range = [2.0, 0.5]
    target_z_pos = 0.2


class PickAndPlaceEnv(DirectRLEnv):
    """Example environment for a PickAndPlace robot using suction-cups.

    This example follows what would be typically done in a DirectRL pipeline.
    Here we substitute the policy by keyboard inputs.
    """

    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Indices used to control the different axes of the gantry
        self._x_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.x_dof_name)
        self._y_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.y_dof_name)
        self._z_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.z_dof_name)

        # joints info
        self.joint_pos = self.pick_and_place.data.joint_pos
        self.joint_vel = self.pick_and_place.data.joint_vel

        # Buffers
        self.go_to_cube = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.go_to_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.instant_controls = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.permanent_controls = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        # Visual marker for the target
        self.set_debug_vis(self.cfg.debug_vis)

        # Set up automated sequence control
        self.set_up_automation()

    def set_up_automation(self):
        """Sets up automated sequence control for pick and place operations."""
        # Automation state machine - now supports multiple environments with OUT-OF-SYNC execution
        self.automation_phase = torch.full((self.num_envs,), 0, dtype=torch.long, device=self.device)  # 0 = move_up
        
        # Randomize start times so environments begin at different times (staggered start)
        # Each environment starts 0.0 to 0.5 seconds apart
        start_time_offsets = torch.rand(self.num_envs, device=self.device) * 0.5
        self.phase_start_time = -1.0 - start_time_offsets  # Will be set to current_time + offset
        
        # Per-environment phase durations with randomization (0.8 to 1.2 seconds per phase)
        # This ensures each environment transitions at different rates
        base_duration = 1.0
        duration_variance = 0.2
        self.phase_duration = base_duration + (torch.rand(self.num_envs, device=self.device) - 0.5) * 2 * duration_variance
        
        # Phase mapping for easier handling
        self.phase_to_int = {
            "move_up": 0,
            "track_cube": 1, 
            "move_down": 2,
            "close_gripper": 3,
            "move_up_again": 4,
            "move_to_target": 5,
            "open_gripper": 6
        }
        self.int_to_phase = {v: k for k, v in self.phase_to_int.items()}
        
        # Phase-specific settings
        self.move_up_force = -200.0
        self.move_down_force = 100.0
        self.gripper_close_command = 1.0
        self.gripper_open_command = -1.0
        self.gripper_idle_command = 0.0
        
        # Position tracking thresholds
        self.position_tolerance = 0.1
        self.z_height_threshold = 0.3
        
        print("Automated pick and place sequence initialized with OUT-OF-SYNC execution!")
        print(f"Phase duration range: {self.phase_duration.min().item():.3f}s - {self.phase_duration.max().item():.3f}s")
        print(f"Start time offset range: {start_time_offsets.min().item():.3f}s - {start_time_offsets.max().item():.3f}s")
        print("Sequence: Move up -> Track cube -> Move down -> Close gripper -> Move up -> Move to target -> Open gripper")

    def update_automation(self, current_time: float):
        """Updates the automation state machine based on current time and robot state."""
        # Initialize phase start time on first call for each environment
        not_started = self.phase_start_time < 0
        if not_started.any():
            self.phase_start_time[not_started] = current_time
            print(f"Starting automation for {not_started.sum().item()} environments with phase: move_up")
        
        # Check if it's time to transition to the next phase for each environment
        # Each environment has its own duration, so they transition independently
        elapsed_time = current_time - self.phase_start_time
        time_to_transition = elapsed_time >= self.phase_duration
        
        if time_to_transition.any():
            self.transition_to_next_phase(time_to_transition, current_time)
            # Randomize next phase duration for environments that transitioned
            # This adds continuous variability and keeps them out of sync
            self.phase_duration[time_to_transition] = 1.0 + (torch.rand(time_to_transition.sum(), device=self.device) - 0.5) * 0.4
        
        # Execute current phase logic for all environments - FULLY VECTORIZED
        # Initialize all controls to zero
        self.permanent_controls.zero_()
        self.instant_controls.zero_()
        self.go_to_cube.zero_()
        self.go_to_target.zero_()
        
        # Create phase masks (all computed in parallel)
        phase_0 = self.automation_phase == 0  # move_up
        phase_1 = self.automation_phase == 1  # track_cube
        phase_2 = self.automation_phase == 2  # move_down
        phase_3 = self.automation_phase == 3  # close_gripper
        phase_4 = self.automation_phase == 4  # move_up_again
        phase_5 = self.automation_phase == 5  # move_to_target
        phase_6 = self.automation_phase == 6  # open_gripper
        
        # Apply phase actions using vectorized operations
        # Phases that use move_up_force: 0, 1, 4, 5, 6
        move_up_phases = phase_0 | phase_1 | phase_4 | phase_5 | phase_6
        self.permanent_controls[move_up_phases, 0] = self.move_up_force
        
        # Phase that uses move_down_force: 2
        self.permanent_controls[phase_2, 0] = self.move_down_force
        
        # Phases that use gripper_idle_command: 0, 1, 2, 4, 5
        idle_phases = phase_0 | phase_1 | phase_2 | phase_4 | phase_5
        self.instant_controls[idle_phases, 2] = self.gripper_idle_command
        
        # Phase that uses gripper_close_command: 3
        self.instant_controls[phase_3, 2] = self.gripper_close_command
        
        # Phase that uses gripper_open_command: 6
        self.instant_controls[phase_6, 2] = self.gripper_open_command
        
        # Set tracking flags
        self.go_to_cube[phase_1] = True  # track_cube phase
        self.go_to_target[phase_5] = True  # move_to_target phase
    
    def transition_to_next_phase(self, env_ids, current_time):
        """Transitions to the next phase in the automation sequence for specified environments."""
        # Move to next phase (with wraparound) - VECTORIZED
        self.automation_phase[env_ids] = (self.automation_phase[env_ids] + 1) % len(self.int_to_phase)
        self.phase_start_time[env_ids] = current_time
        
        # Print transitions for debugging - VECTORIZED (batch print)
        transitioning_env_indices = torch.where(env_ids)[0]
        if len(transitioning_env_indices) > 0:
            # Gather all phase names at once
            phase_values = self.automation_phase[transitioning_env_indices]
            phase_names = [self.int_to_phase[phase.item()] for phase in phase_values]
            env_indices = transitioning_env_indices.tolist()
            
            # # Print all transitions in a single operation
            # transitions = ", ".join([f"Env {env_id}: {phase}" for env_id, phase in zip(env_indices, phase_names)])
            # print(f"Transitions -> {transitions}")

    def _setup_scene(self):
        self.pick_and_place = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.gripper = SurfaceGripper(self.cfg.gripper)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["pick_and_place"] = self.pick_and_place
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.surface_grippers["gripper"] = self.gripper
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Store the actions
        self.actions = actions.clone()

    @carb.profiler.profile
    def _apply_action(self) -> None:
        # Update automation state machine
        current_time = self.episode_length_buf[0].item() * self.cfg.sim.dt
        self.update_automation(current_time)
        
        # Apply the automation outputs as actions
        if self.go_to_cube.any():
            # Effort based proportional controller to track the cube position
            head_pos_x = self.pick_and_place.data.joint_pos[:, self._x_dof_idx[0]]
            head_pos_y = self.pick_and_place.data.joint_pos[:, self._y_dof_idx[0]]
            cube_pos_x = self.cube.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
            cube_pos_y = self.cube.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
            d_cube_robot_x = cube_pos_x - head_pos_x
            d_cube_robot_y = cube_pos_y - head_pos_y
            # Only update controls for environments that are tracking the cube
            self.instant_controls[self.go_to_cube, 0] = d_cube_robot_x[self.go_to_cube] * 5.0
            self.instant_controls[self.go_to_cube, 1] = d_cube_robot_y[self.go_to_cube] * 5.0
            
        if self.go_to_target.any():
            # Effort based proportional controller to track the target position
            head_pos_x = self.pick_and_place.data.joint_pos[:, self._x_dof_idx[0]]
            head_pos_y = self.pick_and_place.data.joint_pos[:, self._y_dof_idx[0]]
            target_pos_x = self.target_pos[:, 0]
            target_pos_y = self.target_pos[:, 1]
            d_target_robot_x = target_pos_x - head_pos_x
            d_target_robot_y = target_pos_y - head_pos_y
            # Only update controls for environments that are tracking the target
            self.instant_controls[self.go_to_target, 0] = d_target_robot_x[self.go_to_target] * 5.0
            self.instant_controls[self.go_to_target, 1] = d_target_robot_y[self.go_to_target] * 5.0
        # Set the joint effort targets for the picker
        self.pick_and_place.set_joint_effort_target(
            self.instant_controls[:, 0].unsqueeze(dim=1), joint_ids=self._x_dof_idx
        )
        self.pick_and_place.set_joint_effort_target(
            self.instant_controls[:, 1].unsqueeze(dim=1), joint_ids=self._y_dof_idx
        )
        self.pick_and_place.set_joint_effort_target(
            self.permanent_controls[:, 0].unsqueeze(dim=1), joint_ids=self._z_dof_idx
        )
        # Set the gripper command
        import time
        time0 = time.time()
        self.gripper.set_grippers_command(self.instant_controls[:, 2])
        time1 = time.time()
        # print(f"[INFO]: set command application: {time1 - time0}")

    @carb.profiler.profile
    def _get_observations(self) -> dict:
        # Get the observations
        # print(self.cube.data.root_pos_w)
        # print(self.joint_pos)
        gripper_state = self.gripper.state.clone()
        obs = torch.cat(
            (
                self.joint_pos[:, self._x_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._x_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._y_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._y_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._z_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._z_dof_idx[0]].unsqueeze(dim=1),
                self.target_pos[:, 0].unsqueeze(dim=1),
                self.target_pos[:, 1].unsqueeze(dim=1),
                gripper_state.unsqueeze(dim=1),
            ),
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.reset_terminated, dtype=torch.float32)

    @carb.profiler.profile
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Dones
        self.joint_pos = self.pick_and_place.data.joint_pos
        self.joint_vel = self.pick_and_place.data.joint_vel
        # Check for time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Check if the cube reached the target
        cube_to_target_x_dist = self.cube.data.root_pos_w[:, 0] - self.target_pos[:, 0] - self.scene.env_origins[:, 0]
        cube_to_target_y_dist = self.cube.data.root_pos_w[:, 1] - self.target_pos[:, 1] - self.scene.env_origins[:, 1]
        cube_to_target_z_dist = self.cube.data.root_pos_w[:, 2] - self.target_pos[:, 2] - self.scene.env_origins[:, 2]
        cube_to_target_distance = torch.norm(
            torch.stack((cube_to_target_x_dist, cube_to_target_y_dist, cube_to_target_z_dist), dim=1), dim=1
        )
        self.target_reached = cube_to_target_distance < 0.3
        # Check if the cube is out of bounds (that is outside of the picking area)
        cube_to_origin_xy_diff = self.cube.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        cube_to_origin_x_dist = torch.abs(cube_to_origin_xy_diff[:, 0])
        cube_to_origin_y_dist = torch.abs(cube_to_origin_xy_diff[:, 1])
        self.cube_out_of_bounds = (cube_to_origin_x_dist > 2.5) | (cube_to_origin_y_dist > 2.5)

        time_out = time_out | self.target_reached
        return self.cube_out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.pick_and_place._ALL_INDICES
        # Reset the environment, this must be done first! As it releases the objects held by the grippers.
        # (And that's an operation that should be done before the gripper or the gripped objects are moved)
        super()._reset_idx(env_ids)
        
        # Reset automation state with randomization to keep environments out of sync
        self.automation_phase[env_ids] = 0  # move_up
        
        # Randomize start time offsets for reset environments (staggered restart)
        start_time_offsets = torch.rand(len(env_ids), device=self.device) * 0.5
        self.phase_start_time[env_ids] = -1.0 - start_time_offsets
        
        # Randomize phase durations for reset environments (0.8 to 1.2 seconds)
        base_duration = 1.0
        duration_variance = 0.2
        self.phase_duration[env_ids] = base_duration + (torch.rand(len(env_ids), device=self.device) - 0.5) * 2 * duration_variance
        
        num_resets = len(env_ids)

        # Set a target position for the cube
        self.target_pos[env_ids, 0] = sample_uniform(
            self.cfg.target_x_pos_range[0],
            self.cfg.target_x_pos_range[1],
            num_resets,
            self.device,
        )
        self.target_pos[env_ids, 1] = sample_uniform(
            self.cfg.target_y_pos_range[0],
            self.cfg.target_y_pos_range[1],
            num_resets,
            self.device,
        )
        self.target_pos[env_ids, 2] = self.cfg.target_z_pos

        # Set the initial position of the cube
        cube_pos = self.cube.data.default_root_state[env_ids, :7]
        cube_pos[:, 0] = sample_uniform(
            self.cfg.initial_object_x_pos_range[0],
            self.cfg.initial_object_x_pos_range[1],
            cube_pos[:, 0].shape,
            self.device,
        )
        cube_pos[:, 1] = sample_uniform(
            self.cfg.initial_object_y_pos_range[0],
            self.cfg.initial_object_y_pos_range[1],
            cube_pos[:, 1].shape,
            self.device,
        )
        cube_pos[:, 2] = self.cfg.initial_object_z_pos
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(cube_pos, env_ids)

        # Set the initial position of the robot
        joint_pos = self.pick_and_place.data.default_joint_pos[env_ids]
        joint_pos[:, self._x_dof_idx] += sample_uniform(
            self.cfg.initial_x_pos_range[0],
            self.cfg.initial_x_pos_range[1],
            joint_pos[:, self._x_dof_idx].shape,
            self.device,
        )
        joint_pos[:, self._y_dof_idx] += sample_uniform(
            self.cfg.initial_y_pos_range[0],
            self.cfg.initial_y_pos_range[1],
            joint_pos[:, self._y_dof_idx].shape,
            self.device,
        )
        joint_pos[:, self._z_dof_idx] += sample_uniform(
            self.cfg.initial_z_pos_range[0],
            self.cfg.initial_z_pos_range[1],
            joint_pos[:, self._z_dof_idx].shape,
            self.device,
        )
        joint_vel = self.pick_and_place.data.default_joint_vel[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.pick_and_place.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.25
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self.target_pos + self.scene.env_origins)


def main():
    """Main function."""
    # create environment
    pick_and_place = PickAndPlaceEnv(PickAndPlaceEnvCfg())
    obs, _ = pick_and_place.reset()
    
    # Initialize timing and step counter for FPS calculation
    import time
    start_time = time.time()
    step_count = 0
    max_steps = 1000
    
    print(f"Starting pick and place demo for {max_steps} steps...")
    
    while simulation_app.is_running() and step_count < max_steps:
        # check for selected robots
        with torch.inference_mode():
            actions = torch.zeros((pick_and_place.num_envs, 4), device=pick_and_place.device, dtype=torch.float32)
            pick_and_place.step(actions)
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                elapsed = time.time() - start_time
                current_fps = step_count / elapsed
                print(f"Step {step_count}/{max_steps} - Current FPS: {current_fps:.2f}")
    
    # Calculate and print final statistics
    end_time = time.time()
    total_time = end_time - start_time
    average_fps = step_count / total_time
    
    print(f"\n=== PERFORMANCE RESULTS ===")
    print(f"Total steps completed: {step_count}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average FPS: {average_fps:.4f}")
    print(f"Average effective FPS: {average_fps * pick_and_place.num_envs}")
    print(f"Average step time: {(total_time / step_count) * 1000:.4f} ms")
    print(f"===========================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()