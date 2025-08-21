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
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
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
    
    basket_half_size = 0.132 / 2 # 44.2964 (original_size) * 0.003 (scale) / 2.0
    basket_pos_offset = torch.tensor([0, 0, 0.1135])

    def __init__(self, *args, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.02, camera_uids: Union[str, List[str]]="base_camera", **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if isinstance(camera_uids, str):
            camera_uids = [camera_uids]
        self.camera_uids = camera_uids
        # self.init_obj_orientations = {}

        self.spawn_z_clearance = 0.001

        # self.robot_cumulative_force_limit = 5000
        # self.robot_force_mult = 0.001
        # self.robot_force_penalty_min = 0.2 
        
        # These are for clutter environment where all the objects will appear in all parallel environment
        
        self.object_heights = {
            "013_apple": 0.035,
            "014_lemon": 0.025,
            "002_master_chef_can": 0.05,
            "004_sugar_box": 0.0888,
            "006_mustard_bottle": 0.1945 / 2.0,
            "007_tuna_fish_can": 0.013,
            "024_bowl": 0.028,
            "025_mug": 0.04,
            "015_peach": 0.0296,
            "008_pudding_box": 0.015,
            "051_large_clamp": 0.019,
            "011_banana": 0.025,
            "005_tomato_soup_can": 0.05,
            "009_gelatin_box": 0.014,
            "017_orange": 0.036,
            "012_strawberry": 0.021,
        }
        
        # Define target objects
        self.target_model_ids = ["013_apple", "014_lemon"]
        target_model_xy = [[0.0, -0.1], [0.0, 0.1]]
        self.target_model_poses = [
            sapien.Pose(p=[xy[0], xy[1], self.object_heights[model_id]])
            for model_id, xy in zip(self.target_model_ids, target_model_xy)
        ]

        # Define clutter objects
        self.clutter_model_ids = [
            "002_master_chef_can", "004_sugar_box", "006_mustard_bottle",
            "007_tuna_fish_can", "024_bowl", "025_mug", "015_peach",
            "008_pudding_box", "051_large_clamp", "011_banana",
            "005_tomato_soup_can", "009_gelatin_box", "017_orange","012_strawberry",
        ]
        clutter_model_xy = [
            [-0.096, -0.66], [-0.21, 0.55], [-0.44, 0.34], [-0.39, -0.42],
            [-0.25, -0.35], [-0.3, -0.57], [0.069, -0.45], [-0.29, 0.25],
            [0.0, 0.52], [-0.087, -0.35], [0.03, 0.3], [-0.1, 0.25], [0.12, 0], [-0.15, -0.21]
        ]
        self.clutter_model_poses = [
            sapien.Pose(p=[xy[0], xy[1], self.object_heights[model_id]])
            for model_id, xy in zip(self.clutter_model_ids, clutter_model_xy)
        ]
        
        # Combine them for spawning
        self.model_ids = self.target_model_ids + self.clutter_model_ids
        self.model_poses = self.target_model_poses + self.clutter_model_poses

        # target object poses (-0.4, 0.17), (-0.43, -0.15)

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
        return CameraConfig("render_camera", pose, 448, 448, 1, 0.01, 100)

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
        model_id_2 = "014_lemon"

        obj_idx_1 = self.model_ids.index(model_id_1)
        obj_idx_2 = self.model_ids.index(model_id_2)

        half_height_1 = float(self.model_poses[obj_idx_1].p[2])
        half_height_2 = float(self.model_poses[obj_idx_2].p[2])

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
        
    def _initialize_ycb_objects(self, env_idx: torch.Tensor, options: dict = None):
        b = len(env_idx)

        # Resolve effective indices for layout (for determinism)
        if options is not None and "global_idx" in options:
            gidx = options["global_idx"]
            if isinstance(gidx, torch.Tensor):
                gidx = gidx.to(env_idx.device)
            else:
                gidx = torch.as_tensor(gidx, device=env_idx.device)
            assert len(gidx) == len(env_idx), "global_idx length mismatch"
            eff_idx = gidx.long()
            if not hasattr(self, "_assigned_global_idx") or self._assigned_global_idx.shape[0] != self.num_envs:
                self._assigned_global_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._assigned_global_idx[env_idx] = eff_idx
        else:
            if hasattr(self, "_assigned_global_idx") and self._assigned_global_idx.shape[0] == self.num_envs:
                eff_idx = self._assigned_global_idx[env_idx]
            else:
                eff_idx = env_idx.long()
        
        # Create one generator per env for deterministic randomization
        rngs = [torch.Generator(device=self.device).manual_seed(int(i)) for i in eff_idx]

        num_targets = len(self.target_model_ids)
        num_clutters = len(self.clutter_model_ids)
        num_models = len(self.model_ids)

        # 1. PERMUTATION OF XY POSITIONS
        # Extract base xy and z from self.model_poses
        all_base_p = torch.tensor([p.p for p in self.model_poses], device=self.device, dtype=torch.float32)

        target_base_xy = all_base_p[:num_targets, :2]
        target_base_z = all_base_p[:num_targets, 2:] # Keep Z with the object model

        clutter_base_xy = all_base_p[num_targets:, :2]
        clutter_base_z = all_base_p[num_targets:, 2:] # Keep Z with the object model

        # Generate and apply permutations for xy coordinates for each env
        target_perms = torch.stack([torch.randperm(num_targets, generator=g, device=self.device) for g in rngs])
        clutter_perms = torch.stack([torch.randperm(num_clutters, generator=g, device=self.device) for g in rngs])
        
        permuted_target_xy = target_base_xy[target_perms]
        permuted_clutter_xy = clutter_base_xy[clutter_perms]
        
        # Re-combine with original Z to form the new base positions for the batch
        permuted_target_pos = torch.cat([permuted_target_xy, target_base_z.unsqueeze(0).expand(b, -1, -1)], dim=-1)
        permuted_clutter_pos = torch.cat([permuted_clutter_xy, clutter_base_z.unsqueeze(0).expand(b, -1, -1)], dim=-1)

        batch_base_pos = torch.cat([permuted_target_pos, permuted_clutter_pos], dim=1)

        # 2. XY-OFFSET
        # Generate deterministic xy-offsets with different ranges
        max_offsets = torch.zeros(num_models, 1, device=self.device)
        max_offsets[:num_targets] = 0.05
        max_offsets[num_targets:] = 0.01 

        xy_offsets = torch.stack([
            (torch.rand(num_models, 2, generator=g, device=self.device) * 2 - 1) * max_offsets
            for g in rngs
        ])
        
        final_positions = batch_base_pos
        final_positions[..., :2] += xy_offsets

        # 3. Z-AXIS ROTATION
        # Generate deterministic random orientations
        random_angles = torch.stack([
            torch.rand(num_models, generator=g, device=self.device) * 2 * torch.pi
            for g in rngs
        ])

        cos_half_angle = torch.cos(random_angles / 2)
        sin_half_angle = torch.sin(random_angles / 2)
        z_rot_quats = torch.zeros(b, num_models, 4, device=self.device)
        z_rot_quats[..., 0] = cos_half_angle # w
        z_rot_quats[..., 3] = sin_half_angle # z
        
        # 4. APPLY FINAL POSES
        for obj_idx, obj in enumerate(self.ycb_objects):
            obj_positions = final_positions[:, obj_idx, :]
            
            if not hasattr(self, "init_obj_orientations"):
                self.init_obj_orientations = torch.empty((num_models, self.num_envs, 4), device=self.device)
            if len(env_idx) == self.num_envs:
                self.init_obj_orientations[obj_idx] = obj.pose.q
            
            base_orientations = self.init_obj_orientations[obj_idx, env_idx]
            random_z_rotations = z_rot_quats[:, obj_idx, :]
            
            final_orientations = quaternion_multiply(random_z_rotations, base_orientations)
            obj.set_pose(Pose.create_from_pq(obj_positions, final_orientations))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # if not hasattr(self, "robot_cumulative_force"):
            #     self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
            self.table_scene.initialize(env_idx)
            self._initialize_ycb_objects(env_idx, options)
        
            if not hasattr(self, "stage1_done") or self.stage1_done.shape[0] != self.num_envs:
                self.stage1_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self.stage1_done[env_idx] = False
            # if not hasattr(self, "returned_to_start_flag") or self.returned_to_start_flag.shape[0] != self.num_envs:
            #     self.returned_to_start_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # else:
            #     self.returned_to_start_flag[env_idx] = False
            # self.robot_cumulative_force[env_idx] = 0.0
    
    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            # env_target_obj_idx=self.env_target_obj_idx,
            # is_grasped_obj_1=info["is_grasped_obj_1"],
            # is_grasped_obj_2=info["is_grasped_obj_2"],
            is_grasp=info["is_grasped_obj_1"] | info["is_grasped_obj_2"],
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

        basket_inside_pos_1 = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
        basket_inside_pos_1[:, 2] = basket_inside_pos_1[:, 2] + self.env_target_obj_half_height_1 + 0.01

        basket_top_target = basket_top_pos.clone()
        basket_top_target[:, 2] += 0.05  # slightly above the rim to encourage clearing the edge

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
        reward = 2.0 * (1.0 - torch.tanh(3.0 * obj_to_tcp_dist_1))

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
        reach_basket_top_reward_1 = 1.0 - torch.tanh(3.0 * obj_to_basket_top_dist_1)
        cand = 5.0 + 3.0 * reach_basket_top_reward_1
        reward = update_max(reward, lifted_1, cand)

        # 5. Enter basket
        obj_to_basket_inside_dist_1 = torch.linalg.norm(basket_inside_pos_1 - obj_pos_1, dim=1)
        reach_inside_basket_reward_1 = 1.0 - torch.tanh(3.0 * obj_to_basket_inside_dist_1)
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
        # robot_qpos = self.agent.robot.get_qpos()
        # qpos_dist = torch.sum(torch.abs(robot_qpos - target_qpos), dim=1)
        
        # # Reward for returning to a neutral pose (up to +3)
        # return_to_start_reward = 3.0 * (1.0 - torch.tanh(5.0 * qpos_dist))
        # cand = 12.0 + return_to_start_reward
        # reward = update_max(reward, mask_prog1, cand)
        
        # # Gate for starting Object 2 task
        # # Update the flag if the condition is met. Once true, it stays true.
        # self.returned_to_start_flag = self.returned_to_start_flag | (mask_prog1 & (qpos_dist < 0.05))

        # =========================
        # Object 2: Reach -> Grasp -> Lift -> Approach -> Enter -> Place (gated by obj1 progress)
        # =========================
        if mask_prog1.any():
            obj_pos_2 = self.pick_obj_2.pose.p
            obj_to_tcp_dist_2 = torch.linalg.norm(tcp_pose - obj_pos_2, dim=1)
            reach_obj_2_reward = 2.0 * (1.0 - torch.tanh(3.0 * obj_to_tcp_dist_2))

            # 1. Reach object 2 (dense) - now gated by return to start
            cand = 12.0 + reach_obj_2_reward
            reward = update_max(reward, mask_prog1, cand)

            # 2. Grasp reward
            is_grasped_2 = mask_prog1 & info["is_grasped_obj_2"]
            cand = 16.0
            reward = update_max(reward, is_grasped_2, cand)

            # 3. Lift reward
            obj_bottom_z_2 = obj_pos_2[..., 2] - self.env_target_obj_half_height_2
            lifted_2 = is_grasped_2 & (obj_bottom_z_2 >= 0.01)
            cand = 17.0
            reward = update_max(reward, lifted_2, cand)

            # 4. Approach basket top for O2 (while grasped)
            obj_to_basket_top_dist_2 = torch.linalg.norm(basket_top_target - obj_pos_2, dim=1)
            reach_basket_top_reward_2 = 1.0 - torch.tanh(3.0 * obj_to_basket_top_dist_2)
            cand = 17.0 + 3.0 * reach_basket_top_reward_2
            reward = update_max(reward, lifted_2, cand)

            # 5. Enter basket for O2
            basket_inside_pos_2 = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
            basket_inside_pos_2[:, 2] = basket_inside_pos_2[:, 2] + self.env_target_obj_half_height_2 + 0.01
            obj_to_basket_inside_dist_2 = torch.linalg.norm(basket_inside_pos_2 - obj_pos_2, dim=1)
            reach_inside_basket_reward_2 = 1.0 - torch.tanh(3.0 * obj_to_basket_inside_dist_2)
            mask_e2 = mask_prog1 & info["is_entering_basket_obj_2"]
            cand = 20.0 + reach_inside_basket_reward_2
            reward = update_max(reward, mask_e2, cand)

            # 6. Place inside basket for O2 (ungrasp + static)
            v2 = torch.linalg.norm(self.pick_obj_2.linear_velocity, dim=1)
            av2 = torch.linalg.norm(self.pick_obj_2.angular_velocity, dim=1)
            static_reward_2 = 1.0 - torch.tanh(v2 * 5.0 + av2)
            cand = 21.0 + static_reward_2
            placed_mask_2 = mask_prog1 & info["is_placed_in_basket_obj_2"] & ~info["is_grasped_obj_2"]
            reward = update_max(reward, placed_mask_2, cand)

            # 7. object 2 is placed in basket reward
            cand = 24.0
            reward = update_max(reward, info["success_obj_2"], cand)

        # =========================
        # Final stage: robot goes up and stays static
        # =========================
        robot_qvel = torch.linalg.norm(self.agent.robot.get_qvel(), dim=1)
        robot_static_reward = 1.0 - torch.tanh(5.0 * robot_qvel)

        tcp_to_basket_top_dist = torch.linalg.norm(self.agent.tcp.pose.p - basket_top_target, dim=1)
        reach_basket_top_reward = 1.0 - torch.tanh(3.0 * tcp_to_basket_top_dist)

        final_state = (
            (prev_stage1_done | info["success_obj_1"])
            & info["is_placed_in_basket_obj_2"]
            & info["is_static_obj_2"]
            & (~info["is_grasped_obj_2"])
        )
        final_state_reward = robot_static_reward + reach_basket_top_reward
        cand = 24.0 + final_state_reward
        reward = update_max(reward, final_state, cand)

        # =========================
        # Success bonus
        # =========================
        reward_success = torch.full_like(reward, 28.0)
        reward = update_max(reward, info["success"], reward_success)
        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalize dense reward to a ~[0, 1] range for stability (adjust the divisor after inspecting logs).
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 28.0
