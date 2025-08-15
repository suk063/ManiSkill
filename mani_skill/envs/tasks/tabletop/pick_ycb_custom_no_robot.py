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
from .pick_ycb_custom import PickYCBCustomEnv


@register_env("PickYCBCustomNoRobot-v1", max_episode_steps=50)
class PickYCBCustomNoRobotEnv(PickYCBCustomEnv):
    
    def __init__(self, *args, grid_dim: int = 15, **kwargs):
        super().__init__(*args, grid_dim=grid_dim, robot_uids=[], robot_init_qpos_noise=0.0, **kwargs)

    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        return []

    @property
    def _default_human_render_camera_configs(self):
        moving_camera = CameraConfig(
            "moving_camera", pose=sapien.Pose(), width=224, height=224,
            fov=np.pi * 0.4, near=0.01, far=100, mount=self.cam_mount
        )
        fixed_cam_pose = sapien_utils.look_at(eye=[0.508, -0.5, 0.42], target=[-0.522, 0.2, 0])
        return [
            CameraConfig("hand_cam", pose=fixed_cam_pose, width=224, height=224, fov=np.pi/2, near=0.01, far=100),
            moving_camera
        ]

    def _load_agent(self, options: dict):
        self.agent = None

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")

    def evaluate(self):
        pos_obj = self.pick_obj.pose.p
        pos_basket_bottom = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
        
        xy_flag = torch.linalg.norm(pos_obj[..., :2] - pos_basket_bottom[..., :2], axis=1) <= 0.12
        
        obj_bottom_z = pos_obj[..., 2] - self.env_target_obj_half_height
        obj_top_z = pos_obj[..., 2] + self.env_target_obj_half_height
        basket_bottom_z = pos_basket_bottom[..., 2]
        basket_top_z = pos_basket_bottom[..., 2] + 2 * self.basket_half_size
        
        entering_basket_z_flag = obj_bottom_z < basket_top_z
        placed_in_basket_z_flag = (obj_bottom_z > basket_bottom_z) & (obj_top_z < basket_top_z)
        
        is_obj_entering_basket = torch.logical_and(xy_flag, entering_basket_z_flag)
        is_obj_placed_in_basket = torch.logical_and(xy_flag, placed_in_basket_z_flag)
        is_obj_static = self.pick_obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        success = is_obj_placed_in_basket & is_obj_static

        return {
            "env_target_obj_idx": self.env_target_obj_idx,
            "is_obj_entering_basket": is_obj_entering_basket,
            "is_obj_placed_in_basket": is_obj_placed_in_basket,
            "is_obj_static": is_obj_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            basket_pos=self.basket.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.pick_obj.pose.raw_pose,
                obj_to_basket_pos=self.basket.pose.p - self.pick_obj.pose.p,
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