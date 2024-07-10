# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import time

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--num_frames", type=int, default=100, help="Number of environment frames to run benchmark for.")
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="JSONFileMetrics",
    choices=["LocalLogMetrics", "JSONFileMetrics", "OsmoKPIFile"],
    help="Benchmarking backend options, defaults OsmoKPIFile",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

app_start_time_begin = time.perf_counter_ns()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

app_start_time_end = time.perf_counter_ns()

"""Rest everything follows."""

# enable benchmarking extension
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.benchmark.services")
from omni.isaac.benchmark.services import BaseIsaacBenchmark

imports_time_begin = time.perf_counter_ns()

import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

imports_time_end = time.perf_counter_ns()


# Create the benchmark
benchmark = BaseIsaacBenchmark(
    benchmark_name="benchmark_non_rl",
    workflow_metadata={
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "num_frames", "data": args_cli.num_frames},
        ]
    },
    backend_type=args_cli.benchmark_backend,
)


def main():
    """Train with RL-Games agent."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        log_root_path = os.path.abs(f"benchmark/{args_cli.task}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    task_startup_time_end = time.perf_counter_ns()

    env.reset()

    reset_time_end = time.perf_counter_ns()

    benchmark.set_phase("benchmark")

    # counter for number of frames to run for
    num_frames = 0
    # log frame times
    step_times = []
    while simulation_app.is_running():
        while num_frames < args_cli.num_frames:
            # get upper and lower bounds of action space, sample actions randomly on this interval
            action_high = env.single_action_space.high[0]
            action_low = env.single_action_space.low[0]
            actions = (action_high - action_low) * torch.rand(
                env.num_envs, env.single_action_space.shape[0], device=env.device
            ) - action_high

            # env stepping
            env_step_time_begin = time.perf_counter_ns()
            _ = env.step(actions)
            end_step_time_end = time.perf_counter_ns()
            step_times.append(end_step_time_end - env_step_time_begin)

            num_frames += 1

        # terminate
        break

    benchmark.store_measurements()
    benchmark.stop()

    # compute stats
    step_times = np.array(step_times) / 1e6  # ns to ms
    fps = 1.0 / (step_times / 1000)
    effective_fps = fps * env.num_envs

    stats = dict()
    stats["App launch time"] = (app_start_time_end - app_start_time_begin) / 1e6
    stats["Python imports time"] = (imports_time_end - imports_time_begin) / 1e6
    stats["Task startup time"] = {
        "Total task startup": (task_startup_time_end - task_startup_time_begin) / 1e6,
        "scene_creation_time": env.scene_creation_time * 1000,
        "simulation_start_time": env.simulation_start_time * 1000,
        "reset_time": (reset_time_end - task_startup_time_end) / 1e6,
    }
    # stats["Task startup time"] = (task_startup_time_end - task_startup_time_begin) / 1e6
    stats["Total startup time (Launch to train)"] = (task_startup_time_end - app_start_time_begin) / 1e6
    stats["Environment step"] = {
        "Environment Step time": step_times.tolist(),
        "FPS": fps.tolist(),
        "Effective FPS": (fps * env.num_envs).tolist(),
        "Environment Step time (min)": step_times.min(),
        "Environment Step time (max)": step_times.max(),
        "Environment Step time (mean)": step_times.mean(),
        "FPS (min)": fps.min(),
        "FPS (max)": fps.max(),
        "FPS (mean)": fps.mean(),
        "Effective FPS (min)": effective_fps.min(),
        "Effective FPS (max)": effective_fps.max(),
        "Effective FPS (mean)": effective_fps.mean(),
    }

    print(json.dumps(stats, indent=4))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
