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


@register_env("PickYCBCustomNoRobot-v1", max_episode_steps=50)
class PickYCBCustomNoRobotEnv(BaseEnv):
    model_ids = ["005_tomato_soup_can", "009_gelatin_box", "010_potted_meat_can", "013_apple", "011_banana"]
    obj_half_size = 0.025
    basket_half_size = 0.012
    
    human_cam_eye_pos = [0.6, 0.7, 0.6]
    human_cam_target_pos = [0.0, 0.0, 0.35]

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)
        )

    def __init__(self, *args, grid_dim: int = 15, **kwargs):
        self.grid_dim = grid_dim
        self.init_obj_orientations = {}
        self.ycb_half_heights_m = {
            "005_tomato_soup_can": 0.101 / 2.0,
            "010_potted_meat_can":  0.0423,
            "009_gelatin_box":     0.028 / 2.0,
            "013_apple":           0.07 / 2.0,
            "011_banana":          0.036 / 2.0,
        }
        self.spawn_z_clearance = 0.001
        super().__init__(*args, robot_uids=[], **kwargs)

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
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=0.0, custom_table=True, custom_basket=True, basket_color="orange"
        )
        self.table_scene.build()
        self.basket = self.table_scene.basket
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")

        all_ycb_objects = []
        for env_i in range(self.num_envs):
            env_ycb_objects = []
            for model_i, model_id in enumerate(self.model_ids):
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.set_scene_idxs([env_i])
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1 * (model_i + 1)])
                env_ycb_objects.append(builder.build(name=f"ycb_{model_id}_env_{env_i}"))
            all_ycb_objects.append(env_ycb_objects)

        self.ycb_objects = []
        for obj_idx in range(len(self.model_ids)):
            obj_type_actors = []
            for env_i in range(self.num_envs):
                obj_type_actors.append(all_ycb_objects[env_i][obj_idx])
            merged_actor = Actor.merge(obj_type_actors, name=f"ycb_{self.model_ids[obj_idx]}")
            self.ycb_objects.append(merged_actor)

        pick_objs = []
        self.env_target_obj_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.env_target_obj_half_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            obj_idx = i % len(self.model_ids)
            model_id = self.model_ids[obj_idx]
            half_height = self.ycb_half_heights_m[model_id]
            self.env_target_obj_half_height[i] = half_height
            env_ycb_obj = all_ycb_objects[i][obj_idx]
            self.env_target_obj_idx[i] = obj_idx
            pick_objs.append(env_ycb_obj)

        self.pick_obj = Actor.merge(pick_objs, name="pick_obj")

        # These are for clutter environment where all the objects will appear in all parallel environment
        self.clutter_objects = []
        clutter_model_ids = [
            "002_master_chef_can", "003_cracker_box", "004_sugar_box",
            "006_mustard_bottle", "007_tuna_fish_can", "024_bowl", "025_mug",
            "015_peach", "008_pudding_box", "051_large_clamp"
        ]

        clutter_poses = [
            sapien.Pose(p=[-0.15, -0.4, 0.05]),
            sapien.Pose(p=[0.0, -0.45, 0.1081]),
            sapien.Pose(p=[-0.21, 0.55, 0.0888]),
            sapien.Pose(p=[-0.12, 0.65, 0.1945 / 2.0]),
            sapien.Pose(p=[-0.23, -0.6, 0.013]),
            sapien.Pose(p=[-0.33, -0.65, 0.028]),
            sapien.Pose(p=[-0.27, 0.65, 0.04]),
            sapien.Pose(p=[0.1, -0.25, 0.032]), # Need to change
            sapien.Pose(p=[-0.29, 0.25, 0.015]), # Need to change
            sapien.Pose(p=[0.08, 0.4, 0.019]),
        ]

        for i, model_id in enumerate(clutter_model_ids):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            
            # Set the fixed initial pose for this clutter object
            builder.initial_pose = clutter_poses[i]

            clutter_obj = builder.build(name=f"clutter_{model_id}_{i}")
            self.clutter_objects.append(clutter_obj)

    def _initialize_ycb_objects(self, env_idx: torch.Tensor):
        b = len(env_idx)

        self.scene_x_offset = 0.0
        self.scene_y_offset = 0.0
        self.scene_z_offset = 0.0

        circle_center_x = -0.2 + self.scene_x_offset
        circle_center_y = 0.0 + self.scene_y_offset
        
        radii = torch.tensor([0.25, 0.28], device=self.device)
        num_radii = len(radii)
        num_orderings = 10
        num_angles = 5
        
        radius_idx = env_idx // (num_orderings * num_angles)
        ordering_idx = (env_idx % (num_orderings * num_angles)) // num_angles
        angle_idx = env_idx % num_angles
        
        radius = radii[radius_idx]
        starting_angle = angle_idx * 72
        
        positions = self._generate_circular_positions_vectorized(
            circle_center_x, circle_center_y, self.scene_z_offset,
            radius, starting_angle, ordering_idx
        )
        
        for obj_idx, model_id in enumerate(self.model_ids):
            half_h = self.ycb_half_heights_m.get(model_id, self.obj_half_size)
            z_val = half_h + self.spawn_z_clearance
            positions[:, obj_idx, 2] = torch.full((b,), z_val, device=self.device)

        for obj_idx, obj in enumerate(self.ycb_objects):
            obj_positions = positions[:, obj_idx, :]
            if obj_idx not in self.init_obj_orientations:
                self.init_obj_orientations[obj_idx] = obj.pose.q
                
            obj_orientations = self.init_obj_orientations[obj_idx]
            if model_id == "011_banana":
                angle_rad = np.pi / 4.0
                apply_rotation = True
            elif model_id == "009_gelatin_box":
                angle_rad = np.pi / 2.0
                apply_rotation = True

            if apply_rotation:
                # Define the rotation quaternion around the Z-axis
                rot_w = np.cos(angle_rad / 2.0)
                rot_z = np.sin(angle_rad / 2.0)
                q_rot = torch.tensor([rot_w, 0, 0, rot_z], device=self.device).unsqueeze(0)

                q_orig = obj_orientations
                w1, x1, y1, z1 = q_rot[:, 0], q_rot[:, 1], q_rot[:, 2], q_rot[:, 3]
                w2, x2, y2, z2 = q_orig[:, 0], q_orig[:, 1], q_orig[:, 2], q_orig[:, 3]
                
                w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                
                obj_orientations = torch.stack([w_new, x_new, y_new, z_new], dim=-1)
            obj.set_pose(Pose.create_from_pq(obj_positions, obj_orientations))

    def _generate_circular_positions_vectorized(self, center_x, center_y, center_z, radius, starting_angle, ordering_idx):
        b = len(radius)
        num_objects = len(self.ycb_objects)
        
        orderings = torch.tensor([
            [0, 1, 2, 3, 4], [0, 2, 4, 1, 3], [0, 3, 1, 4, 2], [0, 4, 3, 2, 1], [1, 0, 2, 4, 3],
            [1, 3, 0, 2, 4], [2, 0, 1, 4, 3], [2, 4, 1, 3, 0], [3, 1, 4, 0, 2], [4, 2, 0, 1, 3],
        ], device=self.device)
        
        ordering = orderings[ordering_idx]
        
        angles = starting_angle.unsqueeze(1) + torch.arange(5, device=self.device) * 72
        angles_rad = torch.deg2rad(angles)
        
        x = center_x + radius.unsqueeze(1) * torch.cos(angles_rad)
        y = center_y + radius.unsqueeze(1) * torch.sin(angles_rad)
        z = torch.full((b, 5), center_z, device=self.device)
        
        positions = torch.stack([x, y, z], dim=2)
        
        batch_indices = torch.arange(b, device=self.device).unsqueeze(1).expand(-1, 5)
        reordered_positions = positions[batch_indices, ordering]
        
        return reordered_positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._initialize_ycb_objects(env_idx)

    def evaluate(self):
        pos_obj = self.pick_obj.pose.p
        pos_basket = self.basket.pose.p
        offset = pos_obj - pos_basket
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        
        entering_basket_z_flag = (
            offset[..., 2] - self.env_target_obj_half_height < self.basket_half_size
        )
        placed_in_basket_z_flag = (
            offset[..., 2] - self.env_target_obj_half_height <= self.basket_half_size * 0.5
        )
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
            env_target_obj_idx=self.env_target_obj_idx,
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