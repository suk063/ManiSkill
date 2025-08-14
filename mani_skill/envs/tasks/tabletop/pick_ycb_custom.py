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


@register_env("PickYCBCustom-v1", max_episode_steps=200)
class PickYCBCustomEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, XArm6Robotiq]
    
    # Match PickCube configs
    sensor_cam_eye_pos = [0.3, 0, 0.6]
    sensor_cam_target_pos = [-0.1, 0, 0.1]
    human_cam_eye_pos = [0.6, 0.7, 0.6]
    human_cam_target_pos = [0.0, 0.0, 0.35]
    model_ids = ["005_tomato_soup_can", "009_gelatin_box", "024_bowl", "013_apple", "011_banana"]
    obj_half_size = 0.025
    basket_half_size = 0.0807 # 26.9 (original_size) * 0.006 (scale) / 2.0

    def __init__(self, *args, grid_dim: int = 10, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.grid_dim = grid_dim
        self.init_obj_orientations = {}

        self.ycb_half_heights_m = {
            "005_tomato_soup_can": 0.101 / 2.0,
            "006_mustard_bottle":  0.175 / 2.0,
            "009_gelatin_box":     0.028 / 2.0,
            "013_apple":           0.07 / 2.0,
            "011_banana":          0.036 / 2.0,
            "024_bowl":            0.053 / 2.0,
        }

        self.spawn_z_clearance = 0.001 

        super().__init__(*args, robot_uids=robot_uids, max_episode_steps=max_episode_steps, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        # base_camera_config = CameraConfig(
        #     uid="base_camera", 
        #     pose=sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos), 
        #     width=128, 
        #     height=128, 
        #     fov=np.pi / 2, 
        #     near=0.01, 
        #     far=100,
        # )

        hand_camera_config = CameraConfig(
            uid="hand_camera",
            pose=sapien.Pose(p=[0, 0, -0.05], q=[0.70710678, 0, 0.70710678, 0]),
            width=224,
            height=224,
            fov=np.pi * 0.4,
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["camera_link"],
        )
        return [hand_camera_config]

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
        self.basket = self.table_scene.basket

        # Create ALL 5 YCB objects for EACH environment
        # Each environment gets its own set of 5 YCB objects
        all_ycb_objects = []
        for env_i in range(self.num_envs):
            env_ycb_objects = []
            for model_i, model_id in enumerate(self.model_ids):
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                # Set scene_idxs to only include this environment
                builder.set_scene_idxs([env_i])
                # Collision-free initial poses while spawning. Objects will be set to the correct positions when episode is initialized.
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1 * (model_i + 1)])

                env_ycb_objects.append(builder.build(name=f"ycb_{model_id}_env_{env_i}"))
            all_ycb_objects.append(env_ycb_objects)
        
        # Create 5 merged actors, each representing one type of YCB object across all environments
        self.ycb_objects = []
        for obj_idx in range(len(self.model_ids)):
            obj_type_actors = []
            for env_i in range(self.num_envs):
                obj_type_actors.append(all_ycb_objects[env_i][obj_idx])
            merged_actor = Actor.merge(obj_type_actors, name=f"ycb_{self.model_ids[obj_idx]}")
            self.ycb_objects.append(merged_actor)
        
        # Create pick_obj as a merged actor representing the pick objects for all environments
        # For each environment i, the pick object is the (i % 5)th object from that environment's set
        pick_objs = []
        self.env_target_obj_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.env_target_obj_half_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            obj_idx = i % len(self.model_ids)  # Which object type this environment should pick
            model_id = self.model_ids[obj_idx]
            half_height = self.ycb_half_heights_m[model_id]
            self.env_target_obj_half_height[i] = half_height

            env_ycb_obj = all_ycb_objects[i][obj_idx]  # Get the YCB object of that type from this environment
            self.env_target_obj_idx[i] = obj_idx  # Save the target object index for this environment to be used in the observation
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
        
        # 2 radii × 10 orderings × 5 angles = 100 unique arrangements
        radii = torch.tensor([0.25, 0.28], device=self.device)  # Two different circle sizes
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
            circle_center_x, circle_center_y, self.scene_z_offset,
            radius, starting_angle, ordering_idx
        )

        # ==== CHANGE START: set per-object z using actual half-heights ====
        # positions: (b, 5, 3); overwrite Z channel for each object type
        for obj_idx, model_id in enumerate(self.model_ids):
            half_h = self.ycb_half_heights_m.get(model_id, self.obj_half_size)
            z_val = half_h + self.spawn_z_clearance
            # broadcast to batch
            positions[:, obj_idx, 2] = torch.full((b,), z_val, device=self.device)
        
        # Apply positions to objects - each merged actor represents one type across all environments
        for obj_idx, obj in enumerate(self.ycb_objects):
            obj_positions = positions[:, obj_idx, :]  # Shape: (b, 3) - positions for this object type across all environments
            # Reset to saved initial orientations to avoid constant reconfiguration to pick up fallen objects
            if obj_idx not in self.init_obj_orientations:
                self.init_obj_orientations[obj_idx] = obj.pose.q
            obj.set_pose(Pose.create_from_pq(obj_positions, self.init_obj_orientations[obj_idx]))

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
    
    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            # env_target_obj_idx=self.env_target_obj_idx,
            # is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            # basket_pos=self.basket.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.pick_obj.pose.raw_pose,
                tcp_to_obj_pos=self.pick_obj.pose.p - self.agent.tcp_pose.p,
                obj_to_basket_pos=self.basket.pose.p - self.pick_obj.pose.p,
            )
        return obs

    def evaluate(self):
        pos_obj = self.pick_obj.pose.p
        pos_basket_bottom = self.basket.pose.p.clone()
        
        # XY-plane check
        xy_flag = torch.linalg.norm(pos_obj[..., :2] - pos_basket_bottom[..., :2], axis=1) <= 0.1  # 105.2038 * 0.003 / 2.0 ~ 0.158  
        
        # Z-axis checks based on clearer variable names
        obj_bottom_z = pos_obj[..., 2] - self.env_target_obj_half_height
        obj_top_z = pos_obj[..., 2] + self.env_target_obj_half_height
        basket_bottom_z = pos_basket_bottom[..., 2]
        basket_top_z = pos_basket_bottom[..., 2] + 2 * self.basket_half_size
        
        # entering_basket_z_flag: True if the object's bottom is below the basket's top edge.
        entering_basket_z_flag = obj_bottom_z < basket_top_z
        
        # placed_in_basket_z_flag: True if the object is entirely inside the basket (vertically).
        placed_in_basket_z_flag = (obj_bottom_z > basket_bottom_z) & (obj_top_z < basket_top_z)
        
        # NOTE: This is the original flag from Dwait's code
        # placed_in_basket_z_flag = (
        #     offset[..., 2] - self.env_target_obj_half_height <= self.basket_half_size * 0.5
        # )
        is_obj_entering_basket = torch.logical_and(xy_flag, entering_basket_z_flag)
        is_obj_placed_in_basket = torch.logical_and(xy_flag, placed_in_basket_z_flag)
        is_obj_grasped = self.agent.is_grasping(self.pick_obj)
        is_obj_static = self.pick_obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_robot_static = self.agent.is_static(0.2)
        success = is_obj_placed_in_basket & is_obj_static & (~is_obj_grasped) & is_robot_static
        return {
            "env_target_obj_idx": self.env_target_obj_idx,
            "is_obj_grasped": is_obj_grasped,
            "is_obj_entering_basket": is_obj_entering_basket,
            "is_obj_placed_in_basket": is_obj_placed_in_basket,
            "is_obj_static": is_obj_static,
            "is_robot_static": is_robot_static,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.pick_obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and reach basket top reward
        obj_pos = self.pick_obj.pose.p
        basket_top_pos = self.basket.pose.p.clone() # [-0.1745, 0, 0.001]
        
        # NOTE: Need to tune this to get a z value slightly above the basket top
        basket_top_pos[:, 2] = basket_top_pos[:, 2] + 2 * self.basket_half_size
        obj_to_basket_top_dist = torch.linalg.norm(basket_top_pos + 0.03 - obj_pos, axis=1)
        reach_basket_top_reward = 1 - torch.tanh(5.0 * obj_to_basket_top_dist)
        reward[info["is_obj_grasped"]] = (4 + reach_basket_top_reward)[info["is_obj_grasped"]]

        # NOTE: Need to tune this to get a z value inside the basket
        basket_inside_pos = self.basket.pose.p.clone()
        basket_inside_pos[:, 2] = basket_inside_pos[:, 2] + 0.03 # add 3cm to the basket bottom z
        obj_to_basket_inside_dist = torch.linalg.norm(basket_inside_pos - obj_pos, axis=1)
        reach_inside_basket_reward = 1 - torch.tanh(5.0 * obj_to_basket_inside_dist)
        reward[info["is_obj_entering_basket"]] = (6 + reach_inside_basket_reward)[info["is_obj_entering_basket"]]

        # ungrasp and static reward
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = self.agent.get_gripper_width()
        ungrasp_reward[
            ~is_obj_grasped
        ] = 1.0
        v = torch.linalg.norm(self.pick_obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.pick_obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 5 + av)
        reward[info["is_obj_placed_in_basket"]] = (
            8 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_obj_placed_in_basket"]]

        # go up and stay static reward
        robot_qvel = torch.linalg.norm(self.agent.robot.get_qvel(), axis=1)
        robot_static_reward = 1 - torch.tanh(5.0 * robot_qvel)
        tcp_to_basket_top_dist = torch.linalg.norm(self.agent.tcp.pose.p - self.basket.pose.p, axis=1)
        reach_basket_top_reward = 1 - torch.tanh(5.0 * tcp_to_basket_top_dist)
        
        final_state = info["is_obj_placed_in_basket"] & ~info["is_obj_grasped"]
        final_state_reward = (robot_static_reward + reach_basket_top_reward) / 2.0
        reward[final_state] = (10 + final_state_reward)[final_state]

        # success reward
        reward[info["success"]] = 15
        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalize dense reward to a ~[0, 2] range for stability (adjust the divisor after inspecting logs).
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0