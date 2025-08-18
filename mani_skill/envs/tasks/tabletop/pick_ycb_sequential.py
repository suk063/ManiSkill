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


@register_env("PickYCBSequential-v1", max_episode_steps=500)
class PickYCBSequentialEnv(BaseEnv):

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
    basket_half_size = 0.132 / 2 # 44.2964 (original_size) * 0.003 (scale) / 2.0
    basket_pos_offset = torch.tensor([0, 0, 0.1135])

    def __init__(self, *args, grid_dim: int = 10, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.02, camera_uids: Union[str, List[str]]="base_camera", **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.grid_dim = grid_dim
        if isinstance(camera_uids, str):
            camera_uids = [camera_uids]
        self.camera_uids = camera_uids
        # self.init_obj_orientations = {}

        self.ycb_half_heights_m = {
            "005_tomato_soup_can": 0.101 / 2.0,
            "006_mustard_bottle":  0.175 / 2.0,
            "009_gelatin_box":     0.028 / 2.0,
            "013_apple":           0.07 / 2.0,
            "011_banana":          0.036 / 2.0,
            "024_bowl":            0.053 / 2.0,
        }

        self.spawn_z_clearance = 0.001

        # self.robot_cumulative_force_limit = 5000
        # self.robot_force_mult = 0.001
        # self.robot_force_penalty_min = 0.2 

        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        base_camera_config = CameraConfig(
            uid="base_camera", 
            pose=sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos), 
            width=224, 
            height=224, 
            fov=np.pi / 3, 
            near=0.01, 
            far=100,
        )

        hand_camera_config = CameraConfig(
            uid="hand_camera",
            pose=sapien.Pose(p=[0, 0, -0.05], q=[0.70710678, 0, 0.70710678, 0]),
            width=224,
            height=224,
            fov=np.pi / 3,
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["camera_link"],
        )
        
        sensor_configs = []
        if "base_camera" in self.camera_uids:
            sensor_configs.append(base_camera_config)
        if "hand_camera" in self.camera_uids:
            sensor_configs.append(hand_camera_config)
        
        return sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict, initial_agent_poses = sapien.Pose(p=[-0.615, 0, 0]), build_separate: bool = False):
        super()._load_agent(options, initial_agent_poses, build_separate)

        # Ignore gripper finger pads for collision checking
        # force_rew_ignore_links = [
        #     "left_outer_finger",
        #     "right_outer_finger",
        #     "left_inner_finger",
        #     "right_inner_finger",
        #     "left_inner_finger_pad",
        #     "right_inner_finger_pad",
        # ]
        # self.force_articulation_link_names = [
        #     link.name
        #     for link in self.agent.robot.get_links()
        #     if link.name not in force_rew_ignore_links
        # ]

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
        pick_objs_1 = []
        pick_objs_2 = []
        self.env_target_obj_idx_1 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.env_target_obj_idx_2 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.env_target_obj_half_height_1 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.env_target_obj_half_height_2 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        model_id_1 = "013_apple"
        model_id_2 = "011_banana"

        obj_idx_1 = self.model_ids.index(model_id_1)
        obj_idx_2 = self.model_ids.index(model_id_2)

        half_height_1 = self.ycb_half_heights_m[model_id_1]
        half_height_2 = self.ycb_half_heights_m[model_id_2]

        for i in range(self.num_envs):
            self.env_target_obj_idx_1[i] = obj_idx_1
            self.env_target_obj_half_height_1[i] = half_height_1
            env_ycb_obj_1 = all_ycb_objects[i][obj_idx_1]
            pick_objs_1.append(env_ycb_obj_1)
            
            self.env_target_obj_idx_2[i] = obj_idx_2
            self.env_target_obj_half_height_2[i] = half_height_2
            env_ycb_obj_2 = all_ycb_objects[i][obj_idx_2]
            pick_objs_2.append(env_ycb_obj_2)
        
        self.pick_obj_1 = Actor.merge(pick_objs_1, name="pick_obj_1")
        self.pick_obj_2 = Actor.merge(pick_objs_2, name="pick_obj_2")
        
        self.stage1_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
            if not hasattr(self, "init_obj_orientations"):
                self.init_obj_orientations = torch.empty((len(self.ycb_objects), self.num_envs, 4), device=self.device)
            # When b=num_envs (first reset), save all initial orientations
            if len(env_idx) == self.num_envs:
                self.init_obj_orientations[obj_idx] = obj.pose.q
            
            obj_orientations = self.init_obj_orientations[obj_idx, env_idx]
            obj.set_pose(Pose.create_from_pq(obj_positions, obj_orientations))

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
            # if not hasattr(self, "robot_cumulative_force"):
            #     self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
            self.table_scene.initialize(env_idx)
            self._initialize_ycb_objects(env_idx)
        
            if not hasattr(self, "stage1_done") or self.stage1_done.shape[0] != self.num_envs:
                self.stage1_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self.stage1_done[env_idx] = False
            if not hasattr(self, "returned_to_start_flag") or self.returned_to_start_flag.shape[0] != self.num_envs:
                self.returned_to_start_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self.returned_to_start_flag[env_idx] = False
            # self.robot_cumulative_force[env_idx] = 0.0
    
    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            # env_target_obj_idx=self.env_target_obj_idx,
            is_grasped_obj_1=info["is_grasped_obj_1"],
            is_grasped_obj_2=info["is_grasped_obj_2"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            # basket_pos=self.basket.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.pick_obj.pose.raw_pose,
                tcp_to_obj1_pos=self.pick_obj_1.pose.p - self.agent.tcp.pose.p,
                tcp_to_obj2_pos=self.pick_obj_2.pose.p - self.agent.tcp.pose.p,
                obj1_to_basket_pos=(self.basket.pose.p + self.basket_pos_offset.to(self.device)) - self.pick_obj_1.pose.p,
                obj2_to_basket_pos=(self.basket.pose.p + self.basket_pos_offset.to(self.device)) - self.pick_obj_2.pose.p,
            )
        return obs

    def evaluate(self):
        if not hasattr(self, "stage1_done") or self.stage1_done.shape[0] != self.num_envs:
            self.stage1_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        z_margin = 0.01

        pos_obj_1 = self.pick_obj_1.pose.p
        pos_obj_2 = self.pick_obj_2.pose.p
        pos_basket_bottom = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
        
        # XY-plane check
        xy_flag_1 = torch.all(torch.abs(pos_obj_1[..., :2] - pos_basket_bottom[..., :2]) <= 0.12, dim=1)  # 24cm x 24cm square check
        xy_flag_2 = torch.all(torch.abs(pos_obj_2[..., :2] - pos_basket_bottom[..., :2]) <= 0.12, dim=1)  # 24cm x 24cm square check
        
        # Z-axis checks based on clearer variable names
        obj_bottom_z_1 = pos_obj_1[..., 2] - self.env_target_obj_half_height_1
        obj_top_z_1 = pos_obj_1[..., 2] + self.env_target_obj_half_height_1
        obj_bottom_z_2 = pos_obj_2[..., 2] - self.env_target_obj_half_height_2
        obj_top_z_2 = pos_obj_2[..., 2] + self.env_target_obj_half_height_2
        basket_bottom_z = pos_basket_bottom[..., 2]
        basket_top_z = pos_basket_bottom[..., 2] + 2 * self.basket_half_size
        
        # entering_basket_z_flag: True if the object's bottom is below the basket's top edge.
        entering_basket_z_flag_1 = obj_bottom_z_1 < (basket_top_z - z_margin)
        entering_basket_z_flag_2 = obj_bottom_z_2 < (basket_top_z - z_margin)
        
        # placed_in_basket_z_flag: True if the object is entirely inside the basket (vertically).
        placed_in_basket_z_flag_1 = (obj_bottom_z_1 > basket_bottom_z) & (obj_top_z_1 < basket_top_z - z_margin)
        placed_in_basket_z_flag_2 = (obj_bottom_z_2 > basket_bottom_z) & (obj_top_z_2 < basket_top_z - z_margin)
        
        is_entering_basket_obj_1 = torch.logical_and(xy_flag_1, entering_basket_z_flag_1)
        is_entering_basket_obj_2 = torch.logical_and(xy_flag_2, entering_basket_z_flag_2)
        is_placed_in_basket_obj_1 = torch.logical_and(xy_flag_1, placed_in_basket_z_flag_1)
        is_placed_in_basket_obj_2 = torch.logical_and(xy_flag_2, placed_in_basket_z_flag_2)
        is_grasped_obj_1 = self.agent.is_grasping(self.pick_obj_1)
        is_grasped_obj_2 = self.agent.is_grasping(self.pick_obj_2)
        is_static_obj_1 = self.pick_obj_1.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_static_obj_2 = self.pick_obj_2.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_robot_static = self.agent.is_static(0.2)

        success_1 = is_placed_in_basket_obj_1 & is_static_obj_1 & (~is_grasped_obj_1)
        prev_stage1_done = self.stage1_done.clone()
        current_stage1_done = prev_stage1_done | success_1
        success_2 = current_stage1_done & is_placed_in_basket_obj_2 & is_static_obj_2 & (~is_grasped_obj_2)
        
        self.stage1_done = current_stage1_done
        success = success_2 & is_robot_static

        # calculate and update robot force
        # robot_force_on_links = self.agent.robot.get_net_contact_forces(self.force_articulation_link_names)
        # robot_force = torch.sum(torch.norm(robot_force_on_links, dim=-1), dim=-1)
        # self.robot_cumulative_force += robot_force

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
            # "robot_force": robot_force,
            # "robot_cumulative_force": self.robot_cumulative_force,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # --- Common tensors ---
        tcp_pose = self.agent.tcp.pose.p
        basket_top_pos = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
        basket_top_pos[:, 2] = basket_top_pos[:, 2] + 2 * self.basket_half_size

        basket_inside_pos = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
        basket_inside_pos[:, 2] = basket_inside_pos[:, 2] + 0.03  # add 3cm to the basket bottom z

        basket_top_target = basket_top_pos.clone()
        basket_top_target[:, 2] += 0.03  # slightly above the rim to encourage clearing the edge

        target_qpos = torch.tensor([0, 0.22, -1.23, 0, 1.01, 0, 0, 0, 0, 0, 0, 0]).to(self.device)

        prev_stage1_done = info["prev_stage1_done"]
 
        # Helper to apply monotonic update on a boolean mask
        def update_max(reward_vec: torch.Tensor, mask: torch.Tensor, candidate: torch.Tensor):
            """
            reward <- max(reward, candidate) on masked elements; unchanged elsewhere.
            """
            if isinstance(candidate, float):
                candidate = torch.full_like(reward_vec, candidate)
            return torch.where(mask, torch.maximum(reward_vec, candidate), reward_vec)

        # =========================
        # Object 1: Reach -> Grasp -> Lift -> Approach -> Enter -> Place
        # =========================
        obj_pos_1 = self.pick_obj_1.pose.p
        obj_to_tcp_dist_1 = torch.linalg.norm(tcp_pose - obj_pos_1, dim=1)

        # 1. Reach reward (dense)
        reward = 2.0 * (1.0 - torch.tanh(5.0 * obj_to_tcp_dist_1))

        # 2. Grasp reward
        is_grasped_1 = info["is_grasped_obj_1"]
        cand = 4.0
        reward = update_max(reward, is_grasped_1, cand)
        
        # 3. Lift reward
        obj_bottom_z_1 = obj_pos_1[..., 2] - self.env_target_obj_half_height_1
        lifted_1 = is_grasped_1 & (obj_bottom_z_1 >= 0.01)
        cand = 5.0
        reward = update_max(reward, lifted_1, cand)

        # 4. Approach basket top (while grasped)
        obj_to_basket_top_dist_1 = torch.linalg.norm(basket_top_target - obj_pos_1, dim=1)
        reach_basket_top_reward_1 = 1.0 - torch.tanh(5.0 * obj_to_basket_top_dist_1)
        cand = 5.0 + 3.0 * reach_basket_top_reward_1
        reward = update_max(reward, lifted_1, cand)

        # 5. Enter basket
        obj_to_basket_inside_dist_1 = torch.linalg.norm(basket_inside_pos - obj_pos_1, dim=1)
        reach_inside_basket_reward_1 = 1.0 - torch.tanh(5.0 * obj_to_basket_inside_dist_1)
        cand = 8.0 + reach_inside_basket_reward_1
        reward = update_max(reward, info["is_entering_basket_obj_1"], cand)

        # 6. Place inside basket (ungrasp + static)
        v1 = torch.linalg.norm(self.pick_obj_1.linear_velocity, dim=1)
        av1 = torch.linalg.norm(self.pick_obj_1.angular_velocity, dim=1)
        static_reward_1 = 1.0 - torch.tanh(v1 * 5.0 + av1)
        cand = 9.0 + static_reward_1
        placed_mask_1 = info["is_placed_in_basket_obj_1"] & ~is_grasped_1
        reward = update_max(reward, placed_mask_1, cand)

        # 7. object 1 is placed in basket reward
        cand = 12.0
        reward = update_max(reward, info["success_obj_1"], cand)

        mask_prog1 = prev_stage1_done | info["success_obj_1"]

        # =========================
        # Post-O1: Robot returns to initial pose
        # =========================
        robot_qpos = self.agent.robot.get_qpos()
        qpos_dist = torch.sum(torch.abs(robot_qpos - target_qpos), dim=1)
        
        # Reward for returning to a neutral pose (up to +2)
        return_to_start_reward = 3.0 * (1.0 - torch.tanh(5.0 * qpos_dist))
        cand = 12.0 + return_to_start_reward
        reward = update_max(reward, mask_prog1, cand)
        
        # Gate for starting Object 2 task
        # Update the flag if the condition is met. Once true, it stays true.
        self.returned_to_start_flag = self.returned_to_start_flag | (mask_prog1 & (qpos_dist < 0.05))

        # =========================
        # Object 2: Reach -> Grasp -> Lift -> Approach -> Enter -> Place (gated by obj1 progress)
        # =========================
        obj_pos_2 = self.pick_obj_2.pose.p
        obj_to_tcp_dist_2 = torch.linalg.norm(tcp_pose - obj_pos_2, dim=1)
        reach_obj_2_reward = 2.0 * (1.0 - torch.tanh(5.0 * obj_to_tcp_dist_2))

        if mask_prog1.any():
            # 1. Reach object 2 (dense) - now gated by return to start
            cand = 15.0 + reach_obj_2_reward
            reward = update_max(reward, self.returned_to_start_flag, cand)

            # 2. Grasp reward
            is_grasped_2 = self.returned_to_start_flag & info["is_grasped_obj_2"]
            cand = 19.0
            reward = update_max(reward, is_grasped_2, cand)

            # 3. Lift reward
            obj_bottom_z_2 = obj_pos_2[..., 2] - self.env_target_obj_half_height_2
            lifted_2 = is_grasped_2 & (obj_bottom_z_2 >= 0.01)
            cand = 20.0
            reward = update_max(reward, lifted_2, cand)

            # 4. Approach basket top for O2 (while grasped)
            obj_to_basket_top_dist_2 = torch.linalg.norm(basket_top_target - obj_pos_2, dim=1)
            reach_basket_top_reward_2 = 1.0 - torch.tanh(5.0 * obj_to_basket_top_dist_2)
            cand = 20.0 + 3.0 * reach_basket_top_reward_2
            reward = update_max(reward, lifted_2, cand)

            # 5. Enter basket for O2
            obj_to_basket_inside_dist_2 = torch.linalg.norm(basket_inside_pos - obj_pos_2, dim=1)
            reach_inside_basket_reward_2 = 1.0 - torch.tanh(5.0 * obj_to_basket_inside_dist_2)
            mask_e2 = self.returned_to_start_flag & info["is_entering_basket_obj_2"]
            cand = 23.0 + reach_inside_basket_reward_2
            reward = update_max(reward, mask_e2, cand)

            # 6. Place inside basket for O2 (ungrasp + static)
            v2 = torch.linalg.norm(self.pick_obj_2.linear_velocity, dim=1)
            av2 = torch.linalg.norm(self.pick_obj_2.angular_velocity, dim=1)
            static_reward_2 = 1.0 - torch.tanh(v2 * 5.0 + av2)
            cand = 24.0 + static_reward_2
            placed_mask_2 = self.returned_to_start_flag & info["is_placed_in_basket_obj_2"] & ~info["is_grasped_obj_2"]
            reward = update_max(reward, placed_mask_2, cand)

            # 7. object 2 is placed in basket reward
            cand = 27.0
            reward = update_max(reward, info["success_obj_2"], cand)

        # =========================
        # Final stage: robot goes up and stays static
        # =========================
        robot_qvel = torch.linalg.norm(self.agent.robot.get_qvel(), dim=1)
        robot_static_reward = 1.0 - torch.tanh(5.0 * robot_qvel)

        tcp_to_basket_top_dist = torch.linalg.norm(self.agent.tcp.pose.p - basket_top_target, dim=1)
        reach_basket_top_reward = 1.0 - torch.tanh(5.0 * tcp_to_basket_top_dist)

        final_state = (
            (prev_stage1_done | info["success_obj_1"])
            & info["is_placed_in_basket_obj_2"]
            & info["is_static_obj_2"]
            & (~info["is_grasped_obj_2"])
        )
        final_state_reward = robot_static_reward + reach_basket_top_reward
        cand = 27.0 + final_state_reward
        reward = update_max(reward, final_state, cand)

        # =========================
        # Success bonus
        # =========================
        reward_success = torch.full_like(reward, 32.0)
        reward = update_max(reward, info["success"], reward_success)
        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalize dense reward to a ~[0, 1] range for stability (adjust the divisor after inspecting logs).
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 32.0
