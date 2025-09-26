# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# Set up paths
newton_dir = "/home/kellyg/Documents/isaac/IsaacLab/newton_benchmarks"
physx_dir = "/home/kellyg/Documents/isaac/IsaacLab/physx_benchmarks"


def parse_benchmark_files(directory):
    """Parse all JSON files in the given directory and extract task, num_envs, and FPS data."""
    data = []
    json_files = glob.glob(os.path.join(directory, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path) as f:
                content = json.load(f)

            # Extract relevant data from runtime section
            runtime = content.get("runtime", {})
            task = runtime.get("task", "Unknown")
            num_envs = runtime.get("num_envs", 0)
            fps = runtime.get("Mean Environment step effective FPS", 0)

            if task != "Unknown" and num_envs > 0 and fps > 0:
                data.append({"task": task, "num_envs": num_envs, "fps": fps, "file": os.path.basename(file_path)})

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

    return data


def organize_data_by_num_envs(newton_data, physx_data):
    """Organize data by num_envs for comparison plots."""
    # Group data by num_envs
    newton_by_envs = defaultdict(dict)
    physx_by_envs = defaultdict(dict)

    for entry in newton_data:
        newton_by_envs[entry["num_envs"]][entry["task"]] = entry["fps"]

    for entry in physx_data:
        physx_by_envs[entry["num_envs"]][entry["task"]] = entry["fps"]

    return newton_by_envs, physx_by_envs


def calculate_percentage_difference(newton_fps, physx_fps):
    """Calculate percentage difference: (newton - physx) / physx * 100"""
    if physx_fps == 0:
        return 0
    return ((newton_fps - physx_fps) / physx_fps) * 100


def create_comparison_plot(num_envs, newton_data, physx_data, save_path):
    """Create a comparison bar plot for the given num_envs."""
    # Find common tasks
    newton_tasks = set(newton_data.keys())
    physx_tasks = set(physx_data.keys())
    common_tasks = newton_tasks.intersection(physx_tasks)

    if not common_tasks:
        print(f"No common tasks found for {num_envs} environments")
        return

    # Sort tasks alphabetically for consistent ordering
    tasks = sorted(list(common_tasks))

    # Extract FPS values
    newton_fps = [newton_data[task] for task in tasks]
    physx_fps = [physx_data[task] for task in tasks]

    # Calculate percentage differences
    percentages = [calculate_percentage_difference(n, p) for n, p in zip(newton_fps, physx_fps)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))

    x = np.arange(len(tasks))
    width = 0.35

    # Create bars
    newton_bars = ax.bar(x - width / 2, newton_fps, width, label="Newton", color="blue", alpha=0.8)
    physx_bars = ax.bar(x + width / 2, physx_fps, width, label="PhysX", color="gold", alpha=0.8)

    # Add percentage labels on top of Newton bars
    for i, (bar, pct) in enumerate(zip(newton_bars, percentages)):
        height = bar.get_height()
        sign = "+" if pct >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(newton_fps + physx_fps) * 0.01,
            f"{sign}{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Customize the plot
    ax.set_xlabel("Tasks", fontsize=12, fontweight="bold")
    ax.set_ylabel("FPS", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Mean Environment step FPS, {num_envs} envs, NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")

    # Add legend
    ax.legend(loc="upper right")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot for {num_envs} environments to {save_path}")
    print(f"Tasks included: {len(tasks)}")
    print(f"Average performance difference: {np.mean(percentages):.1f}%")
    print()


def main():
    print("Parsing Newton benchmark files...")
    newton_data = parse_benchmark_files(newton_dir)
    print(f"Found {len(newton_data)} Newton benchmark entries")

    print("Parsing PhysX benchmark files...")
    physx_data = parse_benchmark_files(physx_dir)
    print(f"Found {len(physx_data)} PhysX benchmark entries")

    # Organize data by num_envs
    newton_by_envs, physx_by_envs = organize_data_by_num_envs(newton_data, physx_data)

    # Find common num_envs values
    newton_envs = set(newton_by_envs.keys())
    physx_envs = set(physx_by_envs.keys())
    common_envs = newton_envs.intersection(physx_envs)

    print(f"Common num_envs values: {sorted(common_envs)}")

    # Create plots for each num_envs
    for num_envs in sorted(common_envs):
        save_path = os.path.join(newton_dir, f"fps_comparison_{num_envs}_envs.png")
        create_comparison_plot(num_envs, newton_by_envs[num_envs], physx_by_envs[num_envs], save_path)

    print("All plots have been generated and saved!")


if __name__ == "__main__":
    main()
