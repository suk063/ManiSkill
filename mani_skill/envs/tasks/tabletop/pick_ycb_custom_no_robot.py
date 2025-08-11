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
    goal_thresh = 0.025
    human_cam_eye_pos = [0.6, 0.7, 0.6]
    human_cam_target_pos = [0.0, 0.0, 0.35]

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)
        )

    def __init__(self, *args, grid_dim: int = 15, **kwargs):
        self.grid_dim = grid_dim
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
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

        model_ids = ["005_tomato_soup_can", "003_cracker_box", "006_mustard_bottle", "013_apple", "011_banana"]

        self._ycb_actors: List[Actor] = []
        all_ycb_objects = []
        for env_i in range(self.num_envs):
            env_ycb_objects = []
            for model_id in model_ids:
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.set_scene_idxs([env_i])
                actor = builder.build(name=f"ycb_{model_id}_env_{env_i}")
                env_ycb_objects.append(actor)
                self._ycb_actors.append(actor)
            all_ycb_objects.append(env_ycb_objects)

        self.ycb_objects = []
        for obj_idx in range(len(model_ids)):
            obj_type_actors = []
            for env_i in range(self.num_envs):
                obj_type_actors.append(all_ycb_objects[env_i][obj_idx])
            merged_actor = Actor.merge(obj_type_actors, name=f"ycb_{model_ids[obj_idx]}")
            self.ycb_objects.append(merged_actor)

        pick_objs = []
        for i in range(self.num_envs):
            obj_idx = i % len(model_ids)
            env_ycb_obj = all_ycb_objects[i][obj_idx]
            pick_objs.append(env_ycb_obj)

        self.pick_obj = Actor.merge(pick_objs, name="pick_obj")

    def _reconfigure(self, options=dict()):
        if hasattr(self, '_ycb_actors'):
            for actor in self._ycb_actors:
                if hasattr(actor, 'entity') and actor.entity is not None:
                    self.scene.remove_actor(actor)
            self._ycb_actors.clear()
        if hasattr(self, 'table_scene'):
            self.table_scene.cleanup()
        super()._reconfigure(options)

    def _initialize_ycb_objects(self, eff_idx: torch.Tensor):
        b = len(eff_idx)

        self.scene_x_offset = 0.0
        self.scene_y_offset = 0.0
        self.scene_z_offset = 0.0

        circle_center_x = -0.2 + self.scene_x_offset
        circle_center_y = 0.0 + self.scene_y_offset
        circle_center_z = 0.015 + self.scene_z_offset
        
        radii = torch.tensor([0.18, 0.22], device=self.device)
        num_radii = len(radii)
        num_orderings = 10
        num_angles = 5
        
        radius_idx = eff_idx // (num_orderings * num_angles)
        ordering_idx = (eff_idx % (num_orderings * num_angles)) // num_angles
        angle_idx = eff_idx % num_angles
        
        radius = radii[radius_idx]
        starting_angle = angle_idx * 72
        
        positions = self._generate_circular_positions_vectorized(
            circle_center_x, circle_center_y, circle_center_z,
            radius, starting_angle, ordering_idx
        )
        
        for obj_idx, obj in enumerate(self.ycb_objects):
            obj_positions = positions[:, obj_idx, :]
            obj.set_pose(Pose.create_from_pq(obj_positions, obj.pose.q))

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
            if options is not None and "global_idx" in options:
                gidx = options["global_idx"]
                if isinstance(gidx, torch.Tensor):
                    gidx = gidx.to(env_idx.device)
                else:
                    gidx = torch.as_tensor(gidx, device=env_idx.device)
                assert len(gidx) == len(env_idx), "global_idx length mismatch"
                eff_idx = gidx.long()
            else:
                eff_idx = env_idx.long()
            self.table_scene.initialize(env_idx)
            self._initialize_ycb_objects(eff_idx)
            
            pick_obj_pos = self.pick_obj.pose.p
            goal_xyz = pick_obj_pos.clone()
            goal_xyz[:, 2] = pick_obj_pos[:, 2] + 0.2
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def _get_obs_extra(self, info: Dict):
        return {}

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