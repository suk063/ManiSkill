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


@register_env("PickYCBSequential-v1", max_episode_steps=200)
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
    
    basket_half_size = 0.04729 # 44.2964 (original_size) * 0.002135 (scale) / 2.0
    basket_pos_offset = torch.tensor([0, 0, 0.08081])

    def __init__(self, *args, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.1, camera_uids: Union[str, List[str]]="hand_camera", **kwargs):
        self.is_eval = kwargs.pop("is_eval", False)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if isinstance(camera_uids, str):
            camera_uids = [camera_uids]
        self.camera_uids = camera_uids
        # self.init_obj_orientations = {}

        self.spawn_z_clearance = 0.001

        self.robot_cumulative_force_limit = 500
        self.robot_force_mult = 0.005
        self.robot_force_penalty_min = 0.1 
        
        # These are for clutter environment where all the objects will appear in all parallel environment
        
        self.object_heights_half = {
            "013_apple": 0.035,
            "014_lemon": 0.025,
            "002_master_chef_can": 0.05,
            "004_sugar_box": 0.0888,
            "006_mustard_bottle": 0.1945 / 2.0,
            "007_tuna_fish_can": 0.013,
            "024_bowl": 0.028,
            "025_mug": 0.041,
            "015_peach": 0.0296,
            "008_pudding_box": 0.015,
            "051_large_clamp": 0.019,
            "011_banana": 0.025,
            "005_tomato_soup_can": 0.05,
            "009_gelatin_box": 0.014,
            "017_orange": 0.036,
            "012_strawberry": 0.021,
            "019_pitcher_base": 0.235 / 2.0,
            "016_pear": 0.033,
            "040_large_marker": 0.009,
            "018_plum": 0.026
        }
        
        # Define target objects
        self.target_model_ids = ["013_apple", "014_lemon", "017_orange"]
        target_model_xy = [[0.03, -0.15], [0.03, 0.15], [0.03, 0.0]]
        self.target_model_poses = [
            sapien.Pose(p=[xy[0], xy[1], self.object_heights_half[model_id]])
            for model_id, xy in zip(self.target_model_ids, target_model_xy)
        ]

        # Define clutter objects
        self.clutter_model_ids = [
            "002_master_chef_can", "004_sugar_box", "006_mustard_bottle", "007_tuna_fish_can", 
            "024_bowl", "025_mug", "015_peach", "008_pudding_box", 
            "011_banana", "005_tomato_soup_can", "009_gelatin_box", "012_strawberry", 
            "019_pitcher_base", "016_pear", "018_plum", "040_large_marker"
        ]
        clutter_model_xy = [
            [-0.096, -0.66], [-0.21, 0.55], [-0.44, 0.34], [-0.39, -0.42],
            [-0.25, -0.35], [-0.3, -0.57], [0.069, -0.45], [-0.29, 0.25],
            [0.0, 0.52], [-0.087, -0.35], [0.03, 0.3], [-0.1, 0.25], 
            [0.12, 0.2], [-0.44, 0.6], [-0.49, -0.64], [-0.44, 0]
        ]
        self.clutter_model_poses = [
            sapien.Pose(p=[xy[0], xy[1], self.object_heights_half[model_id]])
            for model_id, xy in zip(self.clutter_model_ids, clutter_model_xy)
        ]
        
        # Combine them for spawning
        self.model_ids = self.target_model_ids + self.clutter_model_ids
        self.model_poses = self.target_model_poses + self.clutter_model_poses

        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sensor_configs(self):
        base_camera_config = CameraConfig(
            uid="base_camera", 
            pose=sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos), 
            width=224, 
            height=224, 
            fov=np.deg2rad(57), 
            near=0.01, 
            far=100,
        )

        hand_camera_config = CameraConfig(
            uid="hand_camera",
            pose=sapien.Pose(p=[0, 0, 0], q=[0.70710678, 0, 0.70710678, 0]),
            width=224,
            height=224,
            fov=np.deg2rad(57),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["ego_camera_link"],
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
        force_rew_ignore_links = [
            "left_outer_finger",
            "right_outer_finger",
            "left_inner_finger",
            "right_inner_finger",
            "left_inner_finger_pad",
            "right_inner_finger_pad",
            "left_inner_knuckle",
            "right_inner_knuckle",
            "left_outer_knuckle",
            "right_outer_knuckle",
        ]
        self.force_articulation_link_names = [
            link.name
            for link in self.agent.robot.get_links()
            if link.name not in force_rew_ignore_links
        ]

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise, custom_table=True, custom_basket=True, basket_color="orange"
        )
        self.table_scene.build()
        self.basket = self.table_scene.basket

        # Create ALL 5 YCB objects for EACH environment
        # Each environment gets its own set of 5 YCB objects
        self.all_ycb_objects = []
        for env_i in range(self.num_envs):
            env_ycb_objects = []
            for model_i, model_id in enumerate(self.model_ids):
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                # Set scene_idxs to only include this environment
                builder.set_scene_idxs([env_i])
                # Collision-free initial poses while spawning. Objects will be set to the correct positions when episode is initialized.
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1 * (model_i + 1)])

                env_ycb_objects.append(builder.build(name=f"ycb_{model_id}_env_{env_i}"))
            self.all_ycb_objects.append(env_ycb_objects)
        
        self.ycb_objects = []
        for obj_idx in range(len(self.model_ids)):
            obj_type_actors = []
            for env_i in range(self.num_envs):
                obj_type_actors.append(self.all_ycb_objects[env_i][obj_idx])
            merged_actor = Actor.merge(obj_type_actors, name=f"ycb_{self.model_ids[obj_idx]}")
            self.ycb_objects.append(merged_actor)

        # Create stable handles to target actors based on the order in `target_model_ids`
        self.target_obj_indices = [self.model_ids.index(model_id) for model_id in self.target_model_ids]
        self.target_actors = [self.ycb_objects[i] for i in self.target_obj_indices]
        self.target_half_heights = [float(self.model_poses[i].p[2]) for i in self.target_obj_indices]
        
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
        max_offsets[:num_targets] = 0.04
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
            # 1) Final positions/orientations computed for the partial batch (b rows)
            obj_positions_b = final_positions[:, obj_idx, :]               # [b, 3]
            base_orients_b  = torch.tensor([1.0, 0, 0, 0], device=self.device, dtype=torch.float32).expand(len(env_idx), 4)
            rand_rot_b      = z_rot_quats[:, obj_idx, :]                   # [b, 4]
            final_orients_b = quaternion_multiply(rand_rot_b, base_orients_b)  # [b, 4]

            # 2) Get the full pose buffers.
            full_p = obj.pose.p
            full_q = obj.pose.q

            # 3) Replace poses only at env_idx.
            full_p[env_idx] = obj_positions_b
            full_q[env_idx] = final_orients_b

            # 4) Apply poses only for the environments being reset, in ascending env order
            #    to align with the internal boolean reset mask ordering.
            env_idx_sorted, _ = torch.sort(env_idx)
            obj.set_pose(Pose.create_from_pq(full_p[env_idx_sorted], full_q[env_idx_sorted]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Randomly assign the order of target objects to pick for each episode
        if not hasattr(self, "env_pick_order"):
            # It will store permutations of target object indices
            self.env_pick_order = torch.zeros(self.num_envs, len(self.target_model_ids), dtype=torch.long, device=self.device)
            self.env_target_obj_idx_1 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.env_target_obj_idx_2 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.env_target_obj_half_height_1 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.env_target_obj_half_height_2 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # For the environments being reset, generate a new random permutation of target objects.
        # The first two elements of the permutation will be the targets for this episode.
        num_targets = len(self.target_model_ids)

        if self.is_eval:
            # For evaluation, generate deterministic permutations based on global_idx
            if options is not None and "global_idx" in options:
                gidx = options["global_idx"]
                if isinstance(gidx, torch.Tensor):
                    gidx = gidx.to(env_idx.device)
                else:
                    gidx = torch.as_tensor(gidx, device=env_idx.device)
                assert len(gidx) == len(env_idx), "global_idx length mismatch"
                eff_idx = gidx.long()
            else:
                # Fallback for evaluation if global_idx is not provided
                eff_idx = env_idx.long()
            
            # Use a different seed from _initialize_ycb_objects to ensure different permutations
            rngs = [torch.Generator(device=self.device).manual_seed(int(i) + 42) for i in eff_idx]
            perms = torch.stack([torch.randperm(num_targets, generator=g, device=self.device) for g in rngs])
        else:
            # For training, use random permutations
            perms = torch.stack([torch.randperm(num_targets, device=self.device) for _ in range(len(env_idx))])
        
        self.env_pick_order[env_idx] = perms

        # Update all tracking tensors based on the new order FOR ALL ENVS
        target_obj_indices_tensor = torch.tensor(self.target_obj_indices, device=self.device, dtype=torch.long)
        target_half_heights_tensor = torch.tensor(self.target_half_heights, device=self.device, dtype=torch.float32)

        # First object to pick
        first_pick_indices = self.env_pick_order[:, 0]
        self.env_target_obj_idx_1 = target_obj_indices_tensor[first_pick_indices]
        self.env_target_obj_half_height_1 = target_half_heights_tensor[first_pick_indices]

        # Second object to pick
        second_pick_indices = self.env_pick_order[:, 1]
        self.env_target_obj_idx_2 = target_obj_indices_tensor[second_pick_indices]
        self.env_target_obj_half_height_2 = target_half_heights_tensor[second_pick_indices]

        self.table_scene.initialize(env_idx)
        self._initialize_ycb_objects(env_idx, options)
    
        if not hasattr(self, "returned_to_start_flag") or self.returned_to_start_flag.shape[0] != self.num_envs:
            self.returned_to_start_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            self.returned_to_start_flag[env_idx] = False

        if not hasattr(self, "stage1_done") or self.stage1_done.shape[0] != self.num_envs:
            self.stage1_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            self.stage1_done[env_idx] = False
        
        if not hasattr(self, "robot_cumulative_force"):
            self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
        self.robot_cumulative_force[env_idx] = 0.0  # reset for each episode
    
    
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
        
        # if "state" in self.obs_mode:
        #     obs.update(
        #         # obj_pose=self.pick_obj.pose.raw_pose,
        #         tcp_to_obj1_pos=self.pick_obj_1.pose.p - self.agent.tcp.pose.p,
        #         tcp_to_obj2_pos=self.pick_obj_2.pose.p - self.agent.tcp.pose.p,
        #         obj1_to_basket_pos=(self.basket.pose.p + self.basket_pos_offset.to(self.device)) - self.pick_obj_1.pose.p,
        #         obj2_to_basket_pos=(self.basket.pose.p + self.basket_pos_offset.to(self.device)) - self.pick_obj_2.pose.p,
        #     )
        return obs

    def evaluate(self):
        if not hasattr(self, "stage1_done") or self.stage1_done.shape[0] != self.num_envs:
            self.stage1_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        prev_stage1_done = self.stage1_done.clone()

        z_margin = 0.01

        b_idx = torch.arange(self.num_envs, device=self.device)
        first_pick_indices = self.env_pick_order[:, 0]
        second_pick_indices = self.env_pick_order[:, 1]

        # Get poses of all target objects and select the ones for the current episode
        all_target_pos = torch.stack([actor.pose.p for actor in self.target_actors], dim=1)
        pos_obj_1 = all_target_pos[b_idx, first_pick_indices]
        pos_obj_2 = all_target_pos[b_idx, second_pick_indices]
        
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
        entering_basket_z_flag_1 = (obj_bottom_z_1 < (basket_top_z - z_margin)) & (obj_bottom_z_1 > basket_bottom_z)
        entering_basket_z_flag_2 = (obj_bottom_z_2 < (basket_top_z - z_margin)) & (obj_bottom_z_2 > basket_bottom_z)
        
        # placed_in_basket_z_flag: True if the object is entirely inside the basket (vertically).
        placed_in_basket_z_flag_1 = (obj_bottom_z_1 > basket_bottom_z) & (obj_top_z_1 < basket_top_z - z_margin)
        placed_in_basket_z_flag_2 = (obj_bottom_z_2 > basket_bottom_z) & (obj_top_z_2 < basket_top_z - z_margin)
        
        is_entering_basket_obj_1 = torch.logical_and(xy_flag_1, entering_basket_z_flag_1)
        is_entering_basket_obj_2 = torch.logical_and(xy_flag_2, entering_basket_z_flag_2)
        is_placed_in_basket_obj_1 = torch.logical_and(xy_flag_1, placed_in_basket_z_flag_1)
        is_placed_in_basket_obj_2 = torch.logical_and(xy_flag_2, placed_in_basket_z_flag_2)

        all_is_grasped = torch.stack([self.agent.is_grasping(actor) for actor in self.target_actors], dim=1)
        is_grasped_obj_1 = all_is_grasped[b_idx, first_pick_indices]
        is_grasped_obj_2 = all_is_grasped[b_idx, second_pick_indices]

        all_is_static = torch.stack([actor.is_static(lin_thresh=1e-2, ang_thresh=0.5) for actor in self.target_actors], dim=1)
        is_static_obj_1 = all_is_static[b_idx, first_pick_indices]
        is_static_obj_2 = all_is_static[b_idx, second_pick_indices]

        is_robot_static = self.agent.is_static(0.2)

        success_1 = is_placed_in_basket_obj_1 & is_static_obj_1 & (~is_grasped_obj_1)
        self.stage1_done = prev_stage1_done | success_1
        success_2 = self.stage1_done & is_placed_in_basket_obj_2 & is_static_obj_2 & (~is_grasped_obj_2)
        
        success = success_2 & is_robot_static

        # calculate and update robot force
        robot_force_on_links = self.agent.robot.get_net_contact_forces(self.force_articulation_link_names)
        robot_force = torch.sum(torch.norm(robot_force_on_links, dim=-1), dim=-1)
        self.robot_cumulative_force += robot_force

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
            "robot_force": robot_force,
            "robot_cumulative_force": self.robot_cumulative_force,
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
        b_idx = torch.arange(self.num_envs, device=self.device)
        first_pick_indices = self.env_pick_order[:, 0]
        second_pick_indices = self.env_pick_order[:, 1]

        all_target_pos = torch.stack([actor.pose.p for actor in self.target_actors], dim=1)
        obj_pos_1 = all_target_pos[b_idx, first_pick_indices]
        
        obj_to_tcp_dist_1 = torch.linalg.norm(tcp_pose - obj_pos_1, dim=1)

        # 1. Reach reward (dense)
        reward = 3.0 * (1.0 - torch.tanh(5.0 * obj_to_tcp_dist_1))

        # 2. Grasp reward
        is_grasped_1 = info["is_grasped_obj_1"]
        cand = 5.0
        reward = update_max(reward, is_grasped_1, cand)
        
        # 3. Lift reward
        obj_bottom_z_1 = obj_pos_1[..., 2] - self.env_target_obj_half_height_1
        lifted_1 = is_grasped_1 & (obj_bottom_z_1 >= 0.01)
        cand = 6.0
        reward = update_max(reward, lifted_1, cand)

        # 4. Approach basket top (while grasped)
        obj_to_basket_top_dist_1 = torch.linalg.norm(basket_top_target - obj_pos_1, dim=1)
        reach_basket_top_reward_1 = 1.0 - torch.tanh(5.0 * obj_to_basket_top_dist_1)
        cand = 6.0 + 3.0 * reach_basket_top_reward_1
        reward = update_max(reward, lifted_1, cand)

        # 5. Enter basket
        obj_to_basket_inside_dist_1 = torch.linalg.norm(basket_inside_pos_1 - obj_pos_1, dim=1)
        reach_inside_basket_reward_1 = 1.0 - torch.tanh(5.0 * obj_to_basket_inside_dist_1)
        cand = 9.0 + reach_inside_basket_reward_1
        mask_e1 = lifted_1 & info["is_entering_basket_obj_1"]
        reward = update_max(reward, mask_e1, cand)

        # 6. Place inside basket (ungrasp + static)
        all_v = torch.stack([torch.linalg.norm(actor.linear_velocity, dim=1) for actor in self.target_actors], dim=1)
        all_av = torch.stack([torch.linalg.norm(actor.angular_velocity, dim=1) for actor in self.target_actors], dim=1)
        v1 = all_v[b_idx, first_pick_indices]
        av1 = all_av[b_idx, first_pick_indices]
        static_reward_1 = 1.0 - torch.tanh(v1 * 5.0 + av1)
        cand = 10.0 + static_reward_1
        placed_mask_1 = info["is_placed_in_basket_obj_1"] & ~is_grasped_1
        reward = update_max(reward, placed_mask_1, cand)

        # 7. object 1 is placed in basket reward
        cand = 13.0
        reward = update_max(reward, info["success_obj_1"], cand)

        mask_prog1 = prev_stage1_done | info["success_obj_1"]

        # =========================
        # Post-O1: Robot returns to initial pose
        # =========================
        robot_qpos = self.agent.robot.get_qpos()  # [B, DoF]
        target_qpos = torch.tensor(self.agent.keyframes["rest"].qpos, device=robot_qpos.device, dtype=robot_qpos.dtype) # [DoF]

        diff = torch.linalg.norm(robot_qpos - target_qpos, dim=1)
        return_to_start_reward = 0.5 * (1.0 - torch.tanh(diff / 10.0))

        # tol_val = max(float(self.robot_init_qpos_noise), 1e-6)
        # tol = torch.tensor(tol_val, device=diff.device, dtype=diff.dtype)
        # linf_err_soft = tol * torch.logsumexp(diff / tol, dim=1)  # [B]
        # k = torch.atanh(torch.tensor(0.5, device=diff.device, dtype=diff.dtype)) / tol
        # return_to_start_reward = 2.0 * (1.0 - torch.tanh(k * linf_err_soft))

        # --- Gate to start Object 2 task ---
        # linf_err_hard = diff.max(dim=1).values   # [B]
        # near_home_and_slow = (linf_err_hard <= tol) & (v_norm <= 0.1)

        # v_norm = torch.linalg.norm(self.agent.robot.get_qvel(), dim=1)         # [B]
        # robot_static_reward = 1.0 - torch.tanh(5.0 * v_norm)  # [0, 1)
        # cand = 13.0 + robot_static_reward
        # eeward = update_max(reward, mask_prog1, cand)

        # prev_returned_to_start_flag = self.returned_to_start_flag.clone()
        # self.returned_to_start_flag = self.returned_to_start_flag | (mask_prog1 & (v_norm <= 0.02))

        # just_returned_to_start = ~prev_returned_to_start_flag & self.returned_to_start_flag
        # cand = 14.0
        # reward = update_max(reward, just_returned_to_start, cand)

        # =========================
        # Object 2: Reach -> Grasp -> Lift -> Approach -> Enter -> Place (gated by obj1 progress)
        # =========================
        # 1. Reach object 2 (dense) - now gated by return to start
        obj_pos_2 = all_target_pos[b_idx, second_pick_indices]
        obj_to_tcp_dist_2 = torch.linalg.norm(tcp_pose - obj_pos_2, dim=1)
        reach_obj_2_reward = 3.0 * (1.0 - torch.tanh(5.0 * obj_to_tcp_dist_2)) + return_to_start_reward
        cand = 13.0 + reach_obj_2_reward
        reward = update_max(reward, mask_prog1, cand)

        # 2. Grasp reward
        is_grasped_2 = mask_prog1 & info["is_grasped_obj_2"]
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
        basket_inside_pos_2 = self.basket.pose.p.clone() + self.basket_pos_offset.to(self.device)
        basket_inside_pos_2[:, 2] = basket_inside_pos_2[:, 2] + self.env_target_obj_half_height_2 + 0.01
        obj_to_basket_inside_dist_2 = torch.linalg.norm(basket_inside_pos_2 - obj_pos_2, dim=1)
        reach_inside_basket_reward_2 = 1.0 - torch.tanh(5.0 * obj_to_basket_inside_dist_2)
        mask_e2 = lifted_2 & info["is_entering_basket_obj_2"]
        cand = 23.0 + reach_inside_basket_reward_2
        reward = update_max(reward, mask_e2, cand)

        # 6. Place inside basket for O2 (ungrasp + static)
        v2 = all_v[b_idx, second_pick_indices]
        av2 = all_av[b_idx, second_pick_indices]
        static_reward_2 = 1.0 - torch.tanh(v2 * 5.0 + av2)
        cand = 24.0 + static_reward_2
        placed_mask_2 = mask_prog1 & info["is_placed_in_basket_obj_2"] & ~info["is_grasped_obj_2"]
        reward = update_max(reward, placed_mask_2, cand)

        # 7. object 2 is placed in basket reward
        cand = 27.0
        reward = update_max(reward, info["success_obj_2"], cand)

        # =========================
        # Final stage: robot goes up and stays static
        # =========================
        robot_qvel = torch.linalg.norm(self.agent.robot.get_qvel(), dim=1)
        robot_static_reward = 1.0 - torch.tanh(5.0 * robot_qvel)

        # tcp_to_basket_top_dist = torch.linalg.norm(self.agent.tcp.pose.p - basket_top_target, dim=1)
        # reach_basket_top_reward = 1.0 - torch.tanh(5.0 * tcp_to_basket_top_dist)

        final_state = (
            mask_prog1
            & info["is_placed_in_basket_obj_2"]
            & info["is_static_obj_2"]
            & (~info["is_grasped_obj_2"])
        )
        final_state_reward = robot_static_reward
        cand = 27.0 + final_state_reward
        reward = update_max(reward, final_state, cand)

        # =========================
        # Success bonus
        # =========================
        # On success, encourage returning to start pose
        target_qpos = torch.tensor(self.agent.keyframes["rest"].qpos, device=robot_qpos.device, dtype=robot_qpos.dtype) # [DoF]

        diff = torch.linalg.norm(robot_qpos - target_qpos, dim=1)
        return_to_start_reward = (1.0 - torch.tanh(diff / 5.0))
        cand = 30.0 + return_to_start_reward
        reward = update_max(reward, info["success"], cand)
    
        # Add rewards for collision avoidance.
        # 1. Reward for low instantaneous force.
        step_no_col_rew = (1 - torch.tanh(3 * (torch.clamp(self.robot_force_mult * info["robot_force"], 
                                                                min=self.robot_force_penalty_min) - self.robot_force_penalty_min)))
        reward += step_no_col_rew

        # 2. Reward for staying under cumulative force limit.
        # cum_col_under_thresh_rew = (info["robot_cumulative_force"] < self.robot_cumulative_force_limit).float()
        # reward += cum_col_under_thresh_rew
        
        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalize dense reward to a ~[0, 1] range for stability (adjust the divisor after inspecting logs).
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 32.0
