#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to test for non-determinism in vision-based Isaac Lab environments.

This script runs a vision-based environment twice with the same seed and compares:
1. Camera image observations (byte-wise comparison)
2. Physics state observations (joint positions, velocities, etc.)

The script applies identical actions until the first divergence is detected in camera observations,
and saves all images to disk for visual comparison.

Usage:
    ./isaaclab.sh -p scripts/test_vision_determinism.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --num_envs 1 --num_steps 10
"""

import argparse
import os
from datetime import datetime

import torch
import numpy as np

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test vision environment determinism.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Cartpole-RGB-Camera-Direct-v0",
    help="Name of the vision-based task to test.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=100,
    help="Number of environments to simulate. Images will be saved in a grid showing all environments.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=10,
    help="Number of steps to run for each trial.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save output images. Defaults to ./vision_determinism_test/<timestamp>",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Override default to run headless by default (can still be disabled with --no-headless if supported)
parser.set_defaults(headless=True)

args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import save_images_to_file
from isaaclab.sim import SimulationContext
from isaaclab.utils.seed import configure_seed
from isaaclab_tasks.utils import parse_env_cfg


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def save_images(images: torch.Tensor, output_dir: str, run_id: int, step: int, env_id: int = 0):
    """Save camera images to disk in a grid format showing all environments.
    
    Uses Isaac Lab's save_images_to_file utility which applies global normalization
    (not per-image normalization) to preserve relative differences.
    
    Args:
        images: Tensor of shape (num_envs, H, W, C) containing the images (in range [0, 255])
        output_dir: Directory to save images
        run_id: Run identifier (1 or 2)
        step: Step number
        env_id: Environment ID to save individually (deprecated, kept for compatibility)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to float and normalize to [0, 1] for save_images_to_file
    # Use global min/max to preserve relative differences across all environments
    images_float = images.float()
    images_normalized = images_float / 255.0
    
    # Save grid image using Isaac Lab's utility
    filename = os.path.join(output_dir, f"run{run_id}_step{step:03d}_grid.png")
    save_images_to_file(images_normalized, filename)
    

def get_physics_state(env) -> dict:
    """Extract physics state from the environment.
    
    Returns a dictionary containing relevant physics state information.
    """
    state = {}
    
    # Try to get articulation data (for environments with articulated robots)
    if hasattr(env.unwrapped, "_cartpole"):
        robot = env.unwrapped._cartpole
        state["joint_pos"] = tensor_to_numpy(robot.data.joint_pos)
        state["joint_vel"] = tensor_to_numpy(robot.data.joint_vel)
        state["root_pos"] = tensor_to_numpy(robot.data.root_pos_w)
        state["root_quat"] = tensor_to_numpy(robot.data.root_quat_w)
        state["root_lin_vel"] = tensor_to_numpy(robot.data.root_lin_vel_w)
        state["root_ang_vel"] = tensor_to_numpy(robot.data.root_ang_vel_w)
    
    return state


def compare_states(state1: dict, state2: dict, tolerance: float = 1e-6) -> tuple[bool, list]:
    """Compare two physics states.
    
    Returns:
        Tuple of (is_equal, differences) where differences is a list of mismatched keys
    """
    differences = []
    
    for key in state1.keys():
        if key not in state2:
            differences.append(f"{key}: missing in state2")
            continue
        
        arr1 = state1[key]
        arr2 = state2[key]
        
        if not np.allclose(arr1, arr2, atol=tolerance):
            max_diff = np.max(np.abs(arr1 - arr2))
            differences.append(f"{key}: max_diff={max_diff:.6e}")
    
    return len(differences) == 0, differences


def compare_images(img1: torch.Tensor, img2: torch.Tensor) -> tuple[bool, dict]:
    """Compare two image tensors byte-wise across all environments.
    
    Args:
        img1: First image tensor of shape (num_envs, H, W, C)
        img2: Second image tensor of shape (num_envs, H, W, C)
    
    Returns:
        Tuple of (is_equal, stats) where stats contains comparison statistics
    """
    # Convert to numpy for easier comparison
    arr1 = tensor_to_numpy(img1)
    arr2 = tensor_to_numpy(img2)
    
    # Byte-wise comparison across all environments
    are_equal = np.array_equal(arr1, arr2)
    
    # Calculate statistics
    diff = np.abs(arr1 - arr2)
    
    # Per-environment statistics
    num_envs = arr1.shape[0]
    per_env_equal = []
    per_env_max_diff = []
    per_env_mean_diff = []
    per_env_num_different = []
    
    for i in range(num_envs):
        env_equal = np.array_equal(arr1[i], arr2[i])
        per_env_equal.append(env_equal)
        per_env_max_diff.append(np.max(diff[i]))
        per_env_mean_diff.append(np.mean(diff[i]))
        per_env_num_different.append(np.sum(~np.isclose(arr1[i], arr2[i])))
    
    stats = {
        "byte_equal": are_equal,
        "max_diff": np.max(diff),
        "mean_diff": np.mean(diff),
        "num_different_pixels": np.sum(~np.isclose(arr1, arr2)),
        "total_pixels": arr1.size,
        "num_envs": num_envs,
        "per_env_equal": per_env_equal,
        "per_env_max_diff": per_env_max_diff,
        "per_env_mean_diff": per_env_mean_diff,
        "per_env_num_different": per_env_num_different,
        "num_envs_equal": sum(per_env_equal),
    }
    
    return are_equal, stats


def run_trial(task_name: str, num_envs: int, num_steps: int, seed: int, output_dir: str, run_id: int, 
              actions_list: list = None) -> tuple[list, list, list]:
    """Run a single trial of the environment.
    
    Args:
        task_name: Name of the environment task
        num_envs: Number of parallel environments
        num_steps: Number of steps to run
        seed: Random seed
        output_dir: Directory to save images
        run_id: Run identifier (1 or 2)
        actions_list: Optional list of actions to apply. If None, random actions are generated.
    
    Returns:
        Tuple of (observations_list, states_list, actions_list)
    """
    print(f"\n{'='*80}")
    print(f"RUN {run_id}: Starting trial")
    print(f"{'='*80}")
    
    # Configure seed
    configure_seed(seed, torch_deterministic=True)
    
    # Parse environment config
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Set seed in config
    env_cfg.seed = seed
    
    # Create environment
    env = gym.make(task_name, cfg=env_cfg)
    
    print(f"[INFO] Environment created: {task_name}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Device: {env.unwrapped.device}")
    
    # Reset environment with seed
    obs, _ = env.reset(seed=seed)
    
    observations_list = []
    states_list = []
    recorded_actions = []
    
    # Run for specified number of steps
    for step in range(num_steps):
        # Get raw camera data directly from the sensor (before normalization)
        # This ensures we save the actual RGB values, not the normalized observations
        camera_sensor = env.unwrapped.scene.sensors["tiled_camera"]
        data_type = env.unwrapped.cfg.tiled_camera.data_types[0]
        raw_camera_data = camera_sensor.data.output[data_type].clone()
        
        # Save raw image (grid of all environments)
        save_images(raw_camera_data, output_dir, run_id, step)
        
        # Get observation (camera image) - this is the normalized version for comparison
        if isinstance(obs, dict):
            # Extract policy observation (typically contains the camera image)
            camera_obs = obs["policy"]
        else:
            camera_obs = obs
        
        # Get physics state
        physics_state = get_physics_state(env)
        
        # Store observation and state (using normalized observation for consistency)
        observations_list.append(camera_obs.clone())
        states_list.append(physics_state)
        
        # Generate or use provided action
        if actions_list is not None and step < len(actions_list):
            action = actions_list[step]
        else:
            # Generate random action
            action = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        
        recorded_actions.append(action.clone())
        
        print(f"[RUN {run_id}] Step {step}: action={tensor_to_numpy(action[0])}")
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Close environment
    env.close()
    
    # Clear simulation context to ensure clean state for next run
    SimulationContext.clear_instance()
    
    print(f"\n[RUN {run_id}] Trial completed: {num_steps} steps")
    
    return observations_list, states_list, recorded_actions


def main():
    """Main function to test vision environment determinism."""
    
    # Setup output directory
    if args_cli.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), "vision_determinism_test", timestamp)
    else:
        output_dir = args_cli.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("VISION ENVIRONMENT DETERMINISM TEST")
    print("="*80)
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Number of steps: {args_cli.num_steps}")
    print(f"Seed: {args_cli.seed}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Run first trial
    print("\n" + "="*80)
    print("RUNNING TRIAL 1")
    print("="*80)
    obs1_list, states1_list, actions_list = run_trial(
        args_cli.task,
        args_cli.num_envs,
        args_cli.num_steps,
        args_cli.seed,
        output_dir,
        run_id=1,
        actions_list=None  # Generate random actions
    )
    
    # Run second trial with same actions
    print("\n" + "="*80)
    print("RUNNING TRIAL 2 (with same actions)")
    print("="*80)
    obs2_list, states2_list, actions2_list = run_trial(
        args_cli.task,
        args_cli.num_envs,
        args_cli.num_steps,
        args_cli.seed,
        output_dir,
        run_id=2,
        actions_list=actions_list  # Use same actions as trial 1
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    first_divergence_step = None
    all_images_equal = True
    all_states_equal = True
    
    for step in range(args_cli.num_steps):
        print(f"\n--- Step {step} ---")
        
        # Compare actions
        action1 = tensor_to_numpy(actions_list[step][0])  # First environment
        action2 = tensor_to_numpy(actions2_list[step][0])  # First environment
        actions_equal = np.allclose(action1, action2)
        
        print(f"Actions:")
        print(f"  Run 1: {action1}")
        print(f"  Run 2: {action2}")
        print(f"  Equal: {actions_equal}")
        if not actions_equal:
            print(f"  ⚠️  WARNING: Actions differ between runs!")
        
        # Compare images
        img_equal, img_stats = compare_images(obs1_list[step], obs2_list[step])
        
        print(f"Camera observations (across {img_stats['num_envs']} environments):")
        print(f"  All environments byte-wise equal: {img_stats['byte_equal']}")
        print(f"  Environments with equal images: {img_stats['num_envs_equal']} / {img_stats['num_envs']}")
        print(f"  Max pixel difference (across all envs): {img_stats['max_diff']:.6e}")
        print(f"  Mean pixel difference (across all envs): {img_stats['mean_diff']:.6e}")
        print(f"  Different pixels (across all envs): {img_stats['num_different_pixels']} / {img_stats['total_pixels']}")
        
        # Show per-environment details for first few environments
        if img_stats['num_envs'] <= 5:
            for env_id in range(img_stats['num_envs']):
                print(f"    Env {env_id}: equal={img_stats['per_env_equal'][env_id]}, "
                      f"max_diff={img_stats['per_env_max_diff'][env_id]:.6e}")
        else:
            # Show first 3 environments
            print(f"  Sample environments:")
            for env_id in range(min(3, img_stats['num_envs'])):
                print(f"    Env {env_id}: equal={img_stats['per_env_equal'][env_id]}, "
                      f"max_diff={img_stats['per_env_max_diff'][env_id]:.6e}")
        
        if not img_equal:
            all_images_equal = False
            if first_divergence_step is None:
                first_divergence_step = step
                print(f"  ⚠️  FIRST DIVERGENCE DETECTED AT STEP {step}")
        
        # Compare physics states
        state_equal, state_diffs = compare_states(states1_list[step], states2_list[step])
        
        print(f"Physics state:")
        if state_equal:
            print(f"  ✓ All physics states equal")
        else:
            all_states_equal = False
            print(f"  ✗ Physics states differ:")
            for diff in state_diffs:
                print(f"    - {diff}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Task: {args_cli.task}")
    print(f"Total steps compared: {args_cli.num_steps}")
    print(f"All camera observations equal: {all_images_equal}")
    print(f"All physics states equal: {all_states_equal}")
    
    if first_divergence_step is not None:
        print(f"\n⚠️  NON-DETERMINISM DETECTED:")
        print(f"  First image divergence at step: {first_divergence_step}")
    else:
        print(f"\n✓ Environment appears deterministic for {args_cli.num_steps} steps")
    
    print(f"\nImages saved to: {output_dir}")
    print("  - run1_stepXXX_grid.png: Grid view of all environments from first run")
    print("  - run2_stepXXX_grid.png: Grid view of all environments from second run")
    print("\nImages use Isaac Lab's save_images_to_file utility (no per-image normalization).")
    print("You can visually compare the grid images to see differences across all environments.")
    print("="*80)
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("VISION ENVIRONMENT DETERMINISM TEST\n")
        f.write("="*80 + "\n")
        f.write(f"Task: {args_cli.task}\n")
        f.write(f"Seed: {args_cli.seed}\n")
        f.write(f"Number of environments: {args_cli.num_envs}\n")
        f.write(f"Number of steps: {args_cli.num_steps}\n")
        f.write(f"Device: {args_cli.device}\n")
        f.write("="*80 + "\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"All camera observations equal: {all_images_equal}\n")
        f.write(f"All physics states equal: {all_states_equal}\n")
        
        if first_divergence_step is not None:
            f.write(f"\nFirst image divergence at step: {first_divergence_step}\n")
        else:
            f.write(f"\nEnvironment appears deterministic for {args_cli.num_steps} steps\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED COMPARISON:\n")
        f.write("="*80 + "\n")
        
        for step in range(args_cli.num_steps):
            f.write(f"\nStep {step}:\n")
            
            # Action comparison
            action1 = tensor_to_numpy(actions_list[step][0])  # First environment
            action2 = tensor_to_numpy(actions2_list[step][0])  # First environment
            actions_equal = np.allclose(action1, action2)
            
            f.write(f"  Actions:\n")
            f.write(f"    Run 1: {action1}\n")
            f.write(f"    Run 2: {action2}\n")
            f.write(f"    Equal: {actions_equal}\n")
            if not actions_equal:
                max_action_diff = np.max(np.abs(action1 - action2))
                f.write(f"    Max difference: {max_action_diff:.6e}\n")
            
            # Image comparison
            img_equal, img_stats = compare_images(obs1_list[step], obs2_list[step])
            f.write(f"  Camera observations:\n")
            f.write(f"    All environments byte-wise equal: {img_stats['byte_equal']}\n")
            f.write(f"    Environments with equal images: {img_stats['num_envs_equal']} / {img_stats['num_envs']}\n")
            f.write(f"    Max pixel difference (across all envs): {img_stats['max_diff']:.6e}\n")
            f.write(f"    Mean pixel difference (across all envs): {img_stats['mean_diff']:.6e}\n")
            f.write(f"    Different pixels (across all envs): {img_stats['num_different_pixels']} / {img_stats['total_pixels']}\n")
            
            # State comparison
            state_equal, state_diffs = compare_states(states1_list[step], states2_list[step])
            f.write(f"  Physics state:\n")
            if state_equal:
                f.write(f"    All states equal\n")
            else:
                f.write(f"    States differ:\n")
                for diff in state_diffs:
                    f.write(f"      - {diff}\n")
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation app
        simulation_app.close()
