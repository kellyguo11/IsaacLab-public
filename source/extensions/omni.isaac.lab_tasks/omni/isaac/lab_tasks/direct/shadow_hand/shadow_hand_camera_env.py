# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

import omni.usd
from pxr import Semantics

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_conjugate, quat_mul

from omni.isaac.lab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale

from .models import Trainer
from .shadow_hand_env_cfg import ShadowHandEnvCfg


@configclass
class ShadowHandVisionEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=5.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, -0.35, 1.0), rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=224,
        height=224,
    )
    write_image_to_file = False

    # env
    num_channels = 3
    num_observations = 188 + 27  # state observation + vision CNN embedding
    num_states = 187 + 27  # asymettric states + vision CNN embedding


class ShadowHandVisionEnv(InHandManipulationEnv):
    cfg: ShadowHandVisionEnvCfg

    def __init__(self, cfg: ShadowHandVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._get_embeddings_model()
        # hide goal cubes
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 10.0], device=self.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add semantics for in-hand cube
        prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/object")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_embeddings_model(self):
        self.trainer = Trainer(self.device)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def compute_embeddings_observations(self):
        # process RGB image
        rgb_img = self._tiled_camera.data.output["rgb"].clone() / 255.0
        self._save_images("rgb", rgb_img)
        mean_tensor = torch.mean(rgb_img, dim=(1, 2), keepdim=True)
        rgb_img -= mean_tensor
        # process depth image
        depth_img = self._tiled_camera.data.output["depth"].clone()
        depth_img[depth_img == float("inf")] = 0
        depth_img /= 5.0
        depth_img /= torch.max(depth_img)
        self._save_images("depth", depth_img)
        # process segmentation image
        segmentation_img = self._tiled_camera.data.output["semantic_segmentation"].clone() / 255.0
        mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
        segmentation_img -= mean_tensor
        self._save_images("segmentation", segmentation_img)
        # combine all image input in channel dimension
        img_obs = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)

        # generate ground truth keypoints for in-hand cube
        gt_keypoints = gen_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1))
        for i in range(3):
            gt_keypoints.view(-1, 24)[:, i] -= self.object.data.default_root_state[:, i]
        object_pose = torch.cat([self.object_pos, gt_keypoints.view(-1, 24)], dim=-1)

        # train CNN to regress on keypoint positions
        pose_loss, embeddings = self.trainer.step(img_obs, object_pose)

        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["pose_loss"] = pose_loss

        self.cnn_embeddings = embeddings.clone().detach()
        # compute keypoint states based on CNN output
        goal_keypoints = gen_keypoints(pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=-1))
        predicted_cube_pos = self.cnn_embeddings[:, :3]
        zero_pos_obj_keypoints = self.cnn_embeddings[:, 3:]
        for i in range(3):
            zero_pos_obj_keypoints[:, i] -= predicted_cube_pos[:, i]

        obs = torch.cat(
            (
                self.cnn_embeddings,
                goal_keypoints.view(-1, 24),
                goal_keypoints.view(-1, 24) - zero_pos_obj_keypoints,
            ),
            dim=-1,
        )

        return obs

    def compute_state_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),  # 0:24
                self.cfg.vel_obs_scale * self.hand_dof_vel,  # 24:48
                # goal
                self.in_hand_pos,  # 61:64
                self.goal_rot,  # 64:68
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),  # 72:87
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),  # 87:107
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),  # 107:137
                # actions
                self.actions,  # 137:157
            ),
            dim=-1,
        )
        return obs

    def compute_states(self):
        sim_states = self.compute_full_state()
        state = torch.cat((sim_states, self.cnn_embeddings), dim=-1)
        return state

    def _get_observations(self) -> dict:
        state_obs = self.compute_state_observations()
        embedding_obs = self.compute_embeddings_observations()
        obs = torch.cat((state_obs, embedding_obs), dim=-1)

        self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        state = self.compute_states()
        observations = {"policy": obs, "critic": state}
        return observations

    def _save_images(self, data_type, camera_data):
        if self.cfg.write_image_to_file:
            save_images_to_file(camera_data, f"shadow_hand_{data_type}.png")


@torch.jit.script
def local_to_world_space(pos_offset_local: torch.Tensor, pose_global: torch.Tensor):
    """Convert a point from the local frame to the global frame
    Args:
        pos_offset_local: Point in local frame. Shape: [N, 3]
        pose_global: The spatial pose of this point. Shape: [N, 7]
    Returns:
        Position in the global frame. Shape: [N, 3]
    """
    quat_pos_local = torch.cat(
        [
            torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device),
            pos_offset_local,
        ],
        dim=-1,
    )
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(quat_pos_local, quat_global_conj))[:, 1:4]

    result_pos_gloal = pos_offset_global + pose_global[:, 0:3]

    return result_pos_gloal


@torch.jit.script
def gen_keypoints(
    pose: torch.Tensor, num_keypoints: int = 8, size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03)
):

    num_envs = pose.shape[0]
    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf
