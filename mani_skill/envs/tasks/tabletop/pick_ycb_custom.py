from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig

from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor

from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent



@register_env("PickYCBCustom-v1", max_episode_steps=50)
class PickYCBCustomEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, XArm6Robotiq]
    
    # Match PickCube configs
    goal_thresh = 0.025
    sensor_cam_eye_pos = [0.3, 0, 0.6]
    sensor_cam_target_pos = [-0.1, 0, 0.1]
    human_cam_eye_pos = [0.6, 0.7, 0.6]
    human_cam_target_pos = [0.0, 0.0, 0.35]

    def __init__(self, *args, grid_dim: int = 15, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.grid_dim = grid_dim
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict, initial_agent_poses = sapien.Pose(p=[-0.615, 0, 0]), build_separate: bool = False):
        super()._load_agent(options, initial_agent_poses, build_separate)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise, custom_table=True, custom_basket=True, basket_color="orange"
        )
        self.table_scene.build()

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
        
        # Create ALL 5 YCB objects for EACH environment
        # Each environment gets its own set of 5 YCB objects
        all_ycb_objects = []
        for env_i in range(self.num_envs):
            env_ycb_objects = []
            for model_id in model_ids:
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                # Set scene_idxs to only include this environment
                builder.set_scene_idxs([env_i])
                env_ycb_objects.append(builder.build(name=f"ycb_{model_id}_env_{env_i}"))
            all_ycb_objects.append(env_ycb_objects)
        
        # Create 5 merged actors, each representing one type of YCB object across all environments
        self.ycb_objects = []
        for obj_idx in range(len(model_ids)):
            obj_type_actors = []
            for env_i in range(self.num_envs):
                obj_type_actors.append(all_ycb_objects[env_i][obj_idx])
            merged_actor = Actor.merge(obj_type_actors, name=f"ycb_{model_ids[obj_idx]}")
            self.ycb_objects.append(merged_actor)
        
        # Create pick_obj as a merged actor representing the pick objects for all environments
        # For each environment i, the pick object is the (i % 5)th object from that environment's set
        pick_objs = []
        for i in range(self.num_envs):
            obj_idx = i % len(model_ids)  # Which object type this environment should pick
            env_ycb_obj = all_ycb_objects[i][obj_idx]  # Get the YCB object of that type from this environment
            # Get the entity for this specific environment
            pick_objs.append(env_ycb_obj)  # Only one entity per environment object
        
        self.pick_obj = Actor.merge(pick_objs, name="pick_obj")

    def _initialize_ycb_objects(self, env_idx: torch.Tensor):
        b = len(env_idx)

        self.scene_x_offset = 0.0
        self.scene_y_offset = 0.0
        self.scene_z_offset = 0.0

        # Define center of the circular layout
        circle_center_x = -0.2 + self.scene_x_offset  # Center of YCB area
        circle_center_y = 0.0 + self.scene_y_offset
        circle_center_z = 0.015 + self.scene_z_offset  # Height above the table
        
        # Method 4: 2 radii × 10 orderings × 5 angles = 100 unique arrangements
        radii = torch.tensor([0.18, 0.22], device=self.device)  # Two different circle sizes
        num_radii = len(radii)
        num_orderings = 10
        num_angles = 5
        
        # Vectorized parameter computation
        radius_idx = env_idx // (num_orderings * num_angles)  # 0 or 1
        ordering_idx = (env_idx % (num_orderings * num_angles)) // num_angles  # 0-9
        angle_idx = env_idx % num_angles  # 0-4
        
        # Get radius for each environment
        radius = radii[radius_idx]  # Shape: (b,)
        starting_angle = angle_idx * 72  # 0°, 72°, 144°, 216°, 288° - Shape: (b,)
        
        # Generate circular positions for all environments at once
        positions = self._generate_circular_positions_vectorized(
            circle_center_x, circle_center_y, circle_center_z,
            radius, starting_angle, ordering_idx
        )
        
        # Apply positions to objects - each merged actor represents one type across all environments
        for obj_idx, obj in enumerate(self.ycb_objects):
            obj_positions = positions[:, obj_idx, :]  # Shape: (b, 3) - positions for this object type across all environments
            obj.set_pose(Pose.create_from_pq(obj_positions, obj.pose.q))

    def _generate_circular_positions_vectorized(self, center_x, center_y, center_z, radius, starting_angle, ordering_idx):
        """Generate circular positions for YCB objects using tensor operations."""
        b = len(radius)
        num_objects = len(self.ycb_objects)
        
        # Define 10 different object orderings (permutations of 5 objects)
        # Each ordering is a list of indices representing the order around the circle
        orderings = torch.tensor([
            [0, 1, 2, 3, 4],  # Original order
            [0, 2, 4, 1, 3],  # Skip by 2
            [0, 3, 1, 4, 2],  # Skip by 3
            [0, 4, 3, 2, 1],  # Reverse order
            [1, 0, 2, 4, 3],  # Start with second object
            [1, 3, 0, 2, 4],  # Another variation
            [2, 0, 1, 4, 3],  # Start with third object
            [2, 4, 1, 3, 0],  # Another variation
            [3, 1, 4, 0, 2],  # Start with fourth object
            [4, 2, 0, 1, 3],  # Start with fifth object
        ], device=self.device)  # Shape: (10, 5)
        
        # Get the ordering for each environment
        ordering = orderings[ordering_idx]  # Shape: (b, 5)
        
        # Calculate positions around the circle for all environments
        # Create angle tensor: (b, 5) - each row has angles for 5 positions
        angles = starting_angle.unsqueeze(1) + torch.arange(5, device=self.device) * 72  # Shape: (b, 5)
        angles_rad = torch.deg2rad(angles)  # Shape: (b, 5)
        
        # Calculate x, y positions on the circle
        x = center_x + radius.unsqueeze(1) * torch.cos(angles_rad)  # Shape: (b, 5)
        y = center_y + radius.unsqueeze(1) * torch.sin(angles_rad)  # Shape: (b, 5)
        z = torch.full((b, 5), center_z, device=self.device)  # Shape: (b, 5)
        
        # Stack into positions tensor: (b, 5, 3)
        positions = torch.stack([x, y, z], dim=2)  # Shape: (b, 5, 3)
        
        # Reorder positions according to the ordering for each environment
        # Use advanced indexing to reorder
        batch_indices = torch.arange(b, device=self.device).unsqueeze(1).expand(-1, 5)  # Shape: (b, 5)
        reordered_positions = positions[batch_indices, ordering]  # Shape: (b, 5, 3)
        
        return reordered_positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._initialize_ycb_objects(env_idx)
            
            # Set goal positions based on pick object positions
            pick_obj_pos = self.pick_obj.pose.p
            goal_xyz = pick_obj_pos.clone()
            goal_xyz[:, 2] = pick_obj_pos[:, 2] + 0.2
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.pick_obj.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.pick_obj)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }
    
    def _get_obs_extra(self, info: Dict):
        # # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        # obs = dict(
        #     is_grasped=info["is_grasped"],
        #     # tcp_pose=self.agent.tcp.pose.raw_pose,
        #     # goal_pos=self.goal_site.pose.p,
        # )
        # if "state" in self.obs_mode:
        #     obs.update(
        #         # obj_pose=self.pick_obj.pose.raw_pose,
        #         tcp_to_obj_pos=self.pick_obj.pose.p - self.agent.tcp_pose.p,
        #         obj_to_goal_pos=self.goal_site.pose.p - self.pick_obj.pose.p,
        #     )
        return {}

    def staged_rewards(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.pick_obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.pick_obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        place_reward *= is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        static_reward *= info["is_obj_placed"]

        return reaching_reward.mean(), is_grasped.mean(), place_reward.mean(), static_reward.mean()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.pick_obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.pick_obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5