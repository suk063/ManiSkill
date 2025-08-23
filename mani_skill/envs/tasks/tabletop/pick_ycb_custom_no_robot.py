from typing import Any, Dict, List, Union
import numpy as np
import sapien
import torch
import gymnasium as gym

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig

from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor, GPUMemoryConfig, SimConfig
from .pick_ycb_sequential import PickYCBSequentialEnv


@register_env("PickYCBCustomNoRobot-v1", max_episode_steps=50)
class PickYCBCustomNoRobotEnv(PickYCBSequentialEnv):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids=[], robot_init_qpos_noise=0.0, **kwargs)

    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        return []

    @property
    def _default_human_render_camera_configs(self):
        moving_camera = CameraConfig(
            "moving_camera", pose=sapien.Pose(), width=224, height=224,
            fov=np.pi / 3, near=0.01, far=100, mount=self.cam_mount
        )
        fixed_cam_pose = sapien_utils.look_at(eye=[0.508, -0.5, 0.42], target=[-0.522, 0.2, 0])
        return [
            CameraConfig("hand_cam", pose=fixed_cam_pose, width=224, height=224, fov=np.pi / 3, near=0.01, far=100),
            moving_camera
        ]

    def _load_agent(self, options: dict):
        self.agent = None

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")

    def evaluate(self):
        if not hasattr(self, "stage1_done") or self.stage1_done.shape[0] != self.num_envs:
            self.stage1_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        z_margin = 0.01

        pos_obj_1 = self.pick_obj_1.pose.p
        pos_obj_2 = self.pick_obj_2.pose.p
        pos_basket_bottom = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
        
        # XY-plane check
        xy_flag_1 = torch.all(torch.abs(pos_obj_1[..., :2] - pos_basket_bottom[..., :2]) <= 0.12, dim=1)
        xy_flag_2 = torch.all(torch.abs(pos_obj_2[..., :2] - pos_basket_bottom[..., :2]) <= 0.12, dim=1)
        
        # Z-axis checks
        obj_bottom_z_1 = pos_obj_1[..., 2] - self.env_target_obj_half_height_1
        obj_top_z_1 = pos_obj_1[..., 2] + self.env_target_obj_half_height_1
        obj_bottom_z_2 = pos_obj_2[..., 2] - self.env_target_obj_half_height_2
        obj_top_z_2 = pos_obj_2[..., 2] + self.env_target_obj_half_height_2
        basket_bottom_z = pos_basket_bottom[..., 2]
        basket_top_z = pos_basket_bottom[..., 2] + 2 * self.basket_half_size
        
        entering_basket_z_flag_1 = obj_bottom_z_1 < (basket_top_z - z_margin)
        entering_basket_z_flag_2 = obj_bottom_z_2 < (basket_top_z - z_margin)
        
        placed_in_basket_z_flag_1 = (obj_bottom_z_1 > basket_bottom_z) & (obj_top_z_1 < basket_top_z - z_margin)
        placed_in_basket_z_flag_2 = (obj_bottom_z_2 > basket_bottom_z) & (obj_top_z_2 < basket_top_z - z_margin)
        
        is_entering_basket_obj_1 = torch.logical_and(xy_flag_1, entering_basket_z_flag_1)
        is_entering_basket_obj_2 = torch.logical_and(xy_flag_2, entering_basket_z_flag_2)
        is_placed_in_basket_obj_1 = torch.logical_and(xy_flag_1, placed_in_basket_z_flag_1)
        is_placed_in_basket_obj_2 = torch.logical_and(xy_flag_2, placed_in_basket_z_flag_2)

        is_grasped_obj_1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        is_grasped_obj_2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        is_static_obj_1 = self.pick_obj_1.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_static_obj_2 = self.pick_obj_2.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_robot_static = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        success_1 = is_placed_in_basket_obj_1 & is_static_obj_1 & (~is_grasped_obj_1)
        prev_stage1_done = self.stage1_done.clone()
        current_stage1_done = prev_stage1_done | success_1
        success_2 = current_stage1_done & is_placed_in_basket_obj_2 & is_static_obj_2 & (~is_grasped_obj_2)
        
        self.stage1_done = current_stage1_done
        success = success_2 & is_robot_static

        return {
            "prev_stage1_done": prev_stage1_done,
            "env_target_obj_idx_1": self.env_target_obj_idx_1,
            "env_target_obj_idx_2": self.env_target_obj_idx_2,
            "is_grasped_obj_1": is_grasped_obj_1,
            "is_grasped_obj_2": is_grasped_obj_2,
            "is_entering_basket_obj_1": is_entering_basket_obj_1,
            "is_entering_basket_obj_2": is_entering_basket_obj_2,
            "is_placed_in_basket_obj_1": is_placed_in_basket_obj_1,
            "is_placed_in_basket_obj_2": is_placed_in_basket_obj_2,
            "is_static_obj_1": is_static_obj_1,
            "is_static_obj_2": is_static_obj_2,
            "is_robot_static": is_robot_static,
            "success_obj_1": success_1,
            "success_obj_2": success_2,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            basket_pos=self.basket.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj1_pose=self.pick_obj_1.pose.raw_pose,
                obj2_pose=self.pick_obj_2.pose.raw_pose,
                obj1_to_basket_pos=self.basket.pose.p - self.pick_obj_1.pose.p,
                obj2_to_basket_pos=self.basket.pose.p - self.pick_obj_2.pose.p,
            )
        return obs

    def compute_dense_reward(self, *_, **__):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_obs_agent(self):
        return {}

    def render_at_pose(self, pose: sapien.Pose = None) -> dict:
        camera = self.scene.human_render_cameras["moving_camera"]
        self.cam_mount.set_pose(pose)
        self.scene.update_render()
        camera.camera.take_picture()
        obs = {k: v[0] for k, v in camera.get_obs(position=False).items()}
        if "position" in obs:
            obs["position"][..., 1] *= -1
            obs["position"][..., 2] *= -1
        obs["cam_pose"] = np.concatenate([pose.p, pose.q])
        obs["extrinsic_cv"] = camera.camera.get_extrinsic_matrix()[0]
        obs["intrinsic_cv"] = camera.camera.get_intrinsic_matrix()[0]
        return obs

    def _step_action(self, action):
        return action

    def get_state_dict(self):
        return self.scene.get_sim_state()

    def set_state_dict(self, state):
        return self.scene.set_sim_state(state)