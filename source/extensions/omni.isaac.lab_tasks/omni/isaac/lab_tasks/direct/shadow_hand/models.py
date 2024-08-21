# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, depth=False):
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
    def __init__(self, device):
        self.device = device

        self.cnn_model = CustomCNN()
        self.cnn_model.to(self.device)
        self.cnn_model.train()

        self.rgb_optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=1e-4)
        self.l2_loss = nn.MSELoss()

        self.step_count = 0

    def step(self, image, gt_pose):
        self.rgb_optimizer.zero_grad()

        predicted_pose = self.cnn_model(image).squeeze()
        pose_loss = self.l2_loss(predicted_pose, gt_pose) * 100

        pose_loss.backward()
        self.rgb_optimizer.step()

        self.step_count += 1

        if self.step_count % 100000 == 0:
            torch.save(self.cnn_model.state_dict(), f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth")

        return pose_loss, predicted_pose
