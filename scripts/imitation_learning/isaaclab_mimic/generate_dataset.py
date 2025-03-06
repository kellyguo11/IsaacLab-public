# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import contextlib
import gymnasium as gym
import inspect
import numpy as np
import os
import random
import torch

import isaaclab_mimic.envs  # noqa: F401
import omni.log
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# global variable to keep track of the data generation statistics
num_success = 0
num_failures = 0
num_attempts = 0


async def run_data_generator(
    env: ManagerBasedRLMimicEnv,
    env_id: int,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term: TerminationTermCfg,
    pause_subtask: bool = False,
):
    """Run mimic data generation from the given data generator in the specified environment index.

    Args:
        env: The environment to run the data generator on.
        env_id: The environment index to run the data generation on.
        env_reset_queue: The asyncio queue to send environment (for this particular env_id) reset requests to.
        env_action_queue: The asyncio queue to send actions to for executing actions.
        data_generator: The data generator instance to use.
        success_term: The success termination term to use.
        pause_subtask: Whether to pause the subtask during generation.
    """
    global num_success, num_failures, num_attempts
    while True:
        results = await data_generator.generate(
            env_id=env_id,
            success_term=success_term,
            env_reset_queue=env_reset_queue,
            env_action_queue=env_action_queue,
            pause_subtask=pause_subtask,
        )
        if bool(results["success"]):
            num_success += 1
        else:
            num_failures += 1
        num_attempts += 1


def env_loop(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    shared_datagen_info_pool: DataGenInfoPool,
    asyncio_event_loop: asyncio.AbstractEventLoop,
):
    """Main asyncio loop for the environment.

    Args:
        env: The environment to run the main step loop on.
        env_reset_queue: The asyncio queue to handle reset request the environment.
        env_action_queue: The asyncio queue to handle actions to for executing actions.
        shared_datagen_info_pool: The shared datagen info pool that stores source demo info.
        asyncio_event_loop: The main asyncio event loop.
    """
    global num_success, num_failures, num_attempts
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:

            # check if any environment needs to be reset while waiting for actions
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            actions = torch.zeros(env.action_space.shape)

            # get actions from all the data generators
            for i in range(env.num_envs):
                # an async-blocking call to get an action from a data generator
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            # perform action on environment
            env.step(actions)

            # mark done so the data generators can continue with the step results
            for i in range(env.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != num_attempts:
                prev_num_attempts = num_attempts
                generated_sucess_rate = 100 * num_success / num_attempts if num_attempts > 0 else 0.0
                print("")
                print("*" * 50, "\033[K")
                print(
                    f"{num_success}/{num_attempts} ({generated_sucess_rate:.1f}%) successful demos generated by"
                    " mimic\033[K"
                )
                print("*" * 50, "\033[K")

                # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
                generation_guarantee = env.cfg.datagen_config.generation_guarantee
                generation_num_trials = env.cfg.datagen_config.generation_num_trials
                check_val = num_success if generation_guarantee else num_attempts
                if check_val >= generation_num_trials:
                    print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                    break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

    env.close()


def main():
    num_envs = args_cli.num_envs

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get env name from the dataset file
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # parse configuration
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Override the datagen_config.generation_num_trials with the value from the command line arg
    if args_cli.generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = args_cli.generation_num_trials

    env_cfg.env_name = env_name

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # data generator is in charge of resetting the environment
    env_cfg.terminations = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # reset before starting
    env.reset()

    # Set up asyncio stuff
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()

    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(env, env.cfg, env.device, asyncio_lock=shared_datagen_info_pool_lock)
    shared_datagen_info_pool.load_from_dataset_file(args_cli.input_file)
    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")

    # make data generator object
    data_generator = DataGenerator(env=env, src_demo_datagen_info_pool=shared_datagen_info_pool)
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        data_generator_asyncio_tasks.append(
            asyncio_event_loop.create_task(
                run_data_generator(
                    env,
                    i,
                    env_reset_queue,
                    env_action_queue,
                    data_generator,
                    success_term,
                    pause_subtask=args_cli.pause_subtask,
                )
            )
        )

    try:
        asyncio.ensure_future(asyncio.gather(*data_generator_asyncio_tasks))
    except asyncio.CancelledError:
        print("Tasks were cancelled.")

    env_loop(env, env_reset_queue, env_action_queue, shared_datagen_info_pool, asyncio_event_loop)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
