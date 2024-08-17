import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from omni.isaac.lab.sensors import save_images_to_file

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        weights = models.ResNet18_Weights.DEFAULT
        self.pretrain_transforms = weights.transforms()
        self.resnet18 = models.resnet18(weights=weights)
        modules = list(self.resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False

        self.resnet18.eval()

        self.postprocess = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, x):
        save_images_to_file(x, f"shadow_hand_untransformed.png")
        x = x.permute(0, 3, 1, 2)
        transformed_img = self.pretrain_transforms(x)
        # save_images_to_file(transformed_img.permute(0, 2, 3, 1), f"shadow_hand_transformed.png")
        with torch.no_grad():
            x = self.resnet18(transformed_img)
        # x = self.postprocess(x.squeeze())
        return x

class CustomCNN(nn.Module):
    def __init__(self, device, depth=False):
        self.device = device
        super().__init__()
        num_channel = 1 if depth else 4
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.LayerNorm([16, 110, 110]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.LayerNorm([32, 54, 54]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.LayerNorm([64, 26, 26]),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.LayerNorm([128, 12, 12]),
            nn.AvgPool2d(12)
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 27), 
            # nn.ReLU(),
            # nn.Linear(256, 512), 
            # nn.ReLU(),
        )

        self.resnet18_mean = torch.tensor([0.485, 0.0456, 0.0406], device=self.device)
        self.resnet18_std = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        self.resnet_transform = transforms.Normalize(self.resnet18_mean, self.resnet18_std)

    def forward(self, x):
        # save_images_to_file(x, f"shadow_hand_transformed.png")
        cnn_x = self.cnn(x.permute(0, 3, 1, 2))
        # print("cnn", cnn_x.requires_grad)
        out = self.linear(cnn_x.view(-1, 128))
        # print("linear", out.requires_grad)
        return out


class Trainer:
    def __init__(self, device):
        self.device = device

        distributed = False
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        if world_size > 1:
            distributed = True

        # self.rgb_model = ResNet18()
        self.rgb_model = CustomCNN(self.device)
        if distributed:
            self.rgb_model = DDP(self.rgb_model)
        self.rgb_model.to(self.device)
        self.rgb_model.train()
        # self.depth_model = CustomCNN(depth=True)
        # self.depth_model.to(self.device)

        if distributed:
            for param in self.rgb_model.parameters():
                dist.broadcast(param.data, src=0)

        self.rgb_optimizer = torch.optim.Adam(self.rgb_model.parameters(), lr=1e-4)
        self.l2_loss = nn.MSELoss()

        self.horizon_length = 1
        self.batch_loss = 0
        self.step_count = 1

    def step(self, rgb_image, gt_pose):
        self.rgb_optimizer.zero_grad()

        predicted_pose = self.rgb_model(rgb_image).squeeze()
        pose_loss = self.l2_loss(predicted_pose, gt_pose)
        #self.batch_loss += pose_loss

        if self.step_count % self.horizon_length == 0:
            # self.batch_loss /= self.horizon_length
            # self.batch_loss.backward()
            pose_loss.backward()
            self.rgb_optimizer.step()
            # self.batch_loss = torch.zeros_like(self.batch_loss)

        self.step_count += 1

        if self.step_count % 25000 == 0:
            torch.save(self.rgb_model.state_dict(), f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth")

        return pose_loss, predicted_pose