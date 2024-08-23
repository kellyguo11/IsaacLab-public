# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """Custom CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self):
        super().__init__()
        num_channel = 8
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 110, 110]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 54, 54]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 26, 26]),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 12, 12]),
            nn.AvgPool2d(12),
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )

    def forward(self, x):
        cnn_x = self.cnn(x.permute(0, 3, 1, 2))
        out = self.linear(cnn_x.view(-1, 128))
        return out


class Trainer:
    """Class for training a custom CNN from image data during the rollout process."""

    def __init__(self, device, inference=False):
        self.device = device
        self.inference = inference

        self.cnn_model = CustomCNN()
        self.cnn_model.to(self.device)

        self.rgb_optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=1e-4)
        self.l2_loss = nn.MSELoss()

        self.step_count = 0
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if inference:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            self.cnn_model.load_state_dict(torch.load(checkpoint))
        else:
            self.cnn_model.train()

    def step(self, image, gt_pose):
        """Training step for regressing keypoint positions using a custom CNN."""

        if self.inference:
            with torch.inference_mode():
                predicted_pose = self.cnn_model(image).squeeze()
                return predicted_pose, None
        else:
            self.rgb_optimizer.zero_grad()

            predicted_pose = self.cnn_model(image).squeeze()
            pose_loss = self.l2_loss(predicted_pose, gt_pose) * 100

            pose_loss.backward()
            self.rgb_optimizer.step()

            self.step_count += 1

            if self.step_count % 100000 == 0:
                torch.save(
                    self.cnn_model.state_dict(),
                    os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                )

        return pose_loss, predicted_pose
