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
    sensor_cam_eye_pos = [0.3, 0, 0.6]
    sensor_cam_target_pos = [-0.1, 0, 0.1]
    human_cam_eye_pos = [0.6, 0.7, 0.6]
    human_cam_target_pos = [0.0, 0.0, 0.35]
    model_ids = ["005_tomato_soup_can", "009_gelatin_box", "024_bowl", "013_apple", "011_banana"]
    obj_half_size = 0.025
    basket_half_size = 0.0807 # 26.9 (original_size) * 0.006 (scale) / 2.0

    def __init__(self, *args, grid_dim: int = 15, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.02, **kwargs):
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

        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
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

    # def evaluate(self):
    #     pos_obj = self.pick_obj.pose.p
    #     pos_basket = self.basket.pose.p
    #     offset = pos_obj - pos_basket
    #     xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
    #     # NOTE: Need to check if these flags are correct
    #     entering_basket_z_flag = (
    #         offset[..., 2] - self.env_target_obj_half_height < self.basket_half_size
    #     )
    #     placed_in_basket_z_flag = (
    #         offset[..., 2] - self.env_target_obj_half_height <= self.basket_half_size * 0.5
    #     )
    #     is_obj_entering_basket = torch.logical_and(xy_flag, entering_basket_z_flag)
    #     is_obj_placed_in_basket = torch.logical_and(xy_flag, placed_in_basket_z_flag)
    #     is_obj_grasped = self.agent.is_grasping(self.pick_obj)
    #     is_obj_static = self.pick_obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    #     is_robot_static = self.agent.is_static(0.2)
    #     success = is_obj_placed_in_basket & is_obj_static & (~is_obj_grasped) & is_robot_static
    #     return {
    #         "env_target_obj_idx": self.env_target_obj_idx,
    #         "is_obj_grasped": is_obj_grasped,
    #         "is_obj_entering_basket": is_obj_entering_basket,
    #         "is_obj_placed_in_basket": is_obj_placed_in_basket,
    #         "is_obj_static": is_obj_static,
    #         "is_robot_static": is_robot_static,
    #         "success": success,
    #     }
    
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
        """
        Stage flags and success computed assuming:
        - self.basket.pose.p is the basket BOTTOM center (x,y at center, z at bottom).
        - self.basket_half_size is the half-height along z.
        """
        # --- Basket geometry from BOTTOM pose ---
        basket_bottom = self.basket.pose.p                          # (N,3), bottom center
        basket_height = self.basket_half_size * 2.0                 # interpret half_size as half-height
        rim_z  = basket_bottom[:, 2] + basket_height                # top rim z = bottom z + height
        base_z = basket_bottom[:, 2]                                # basket bottom z
        center_xy = basket_bottom[:, :2]                            # same XY for bottom and center
        inner_radius = self.basket_half_size * 0.85                 # loose inner radius (tune as needed)

        # --- Object geometry ---
        pos_obj = self.pick_obj.pose.p                              # (N,3) object center
        obj_bottom_z = pos_obj[:, 2] - self.env_target_obj_half_height
        xy_dist = torch.linalg.norm((pos_obj[:, :2] - center_xy), dim=1)
        xy_ok = (xy_dist <= inner_radius)

        # --- Stage flags (z-based, unambiguous) ---
        # entering: object bottom is at/above the rim zone while XY is inside
        is_obj_entering_basket = xy_ok & (obj_bottom_z >= rim_z - 0.01)
        # inside: object bottom passed below the rim (i.e., inside the basket)
        is_obj_placed_in_basket = xy_ok & (obj_bottom_z <= rim_z - 0.005)
        # well_placed: near the basket bottom
        well_placed = is_obj_placed_in_basket & (obj_bottom_z <= base_z + 0.01)

        # --- Dynamics / robot ---
        is_obj_grasped = self.agent.is_grasping(self.pick_obj)
        is_obj_static = self.pick_obj.is_static(lin_thresh=5e-3, ang_thresh=0.3)
        is_robot_static = self.agent.is_static(0.15)

        # --- Success ---
        success = well_placed & (~is_obj_grasped) & is_obj_static & is_robot_static

        return {
            "env_target_obj_idx": self.env_target_obj_idx,
            "is_obj_grasped": is_obj_grasped,
            "is_obj_entering_basket": is_obj_entering_basket,
            "is_obj_placed_in_basket": is_obj_placed_in_basket,
            "is_obj_static": is_obj_static,
            "is_robot_static": is_robot_static,
            "success": success,
            # (Optional) debug values:
            # "obj_bottom_z": obj_bottom_z, "rim_z": rim_z, "base_z": base_z, "xy_dist": xy_dist
        }


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Dense reward shaped for: grasp → LIFT (adaptive clearance) → ABOVE RIM → INSIDE → release & stabilize.

        Geometry assumptions
        --------------------
        - self.basket.pose.p is the basket BOTTOM center (x, y at center; z at bottom).
        - rim_z = base_z + basket_height, where basket_height = 2 * basket_half_size.
        - lift_clearance is ADAPTIVE: 0.5 * basket_height, clamped to [0.04 m, 0.10 m].

        Design notes
        ------------
        - The SAME lift_clearance is used consistently for: lift reward, "go above rim" gate,
        and the dragging penalty gate (to discourage XY motion before lifting enough).
        - Rewards are additive to preserve progress signals.
        """
        device = self.device
        N = self.num_envs

        # --- Basket geometry (from BOTTOM pose) ---
        # NOTE (Sunghwan): need to check why self.basket.pose.p is not the bottom center
        # NOTE (Sunghwan): is basket pose itself is correct?

        basket_bottom = self.basket.pose.p                             # (N, 3)
        base_z = basket_bottom[:, 2]                                   # (N,)
        # Height per env as a vector (even if half_size is a scalar)
        basket_height_scalar = float(self.basket_half_size) * 2.0
        basket_height = torch.full((N,), basket_height_scalar, device=device, dtype=torch.float32)  # (N,)
        rim_z = base_z + basket_height                                 # (N,)
        center_xy = basket_bottom[:, :2]                                # (N, 2)

        # --- Adaptive lift clearance (4 cm ~ 10 cm) ---
        # 0.5 * basket height gives a sensible per-env margin; clamp for stability.
        lift_clearance = torch.clamp(basket_height, min=0.04, max=0.10)  # (N,)

        # --- Object / TCP state ---
        pos_obj = self.pick_obj.pose.p                                  # (N, 3)
        tcp_pos = self.agent.tcp_pose.p                                  # (N, 3)  (use tcp_pose for compatibility)
        obj_bottom_z = pos_obj[:, 2] - self.env_target_obj_half_height   # (N,)
        xy_dist = torch.linalg.norm((pos_obj[:, :2] - center_xy), dim=1) # (N,)

        # --- Stage flags from info ---
        is_grasped = info["is_obj_grasped"]                              # (N,) bool
        inside     = info["is_obj_placed_in_basket"]                     # (N,) bool
        success    = info["success"]                                     # (N,) bool

        # --- Targets (only z is adjusted; XY remains at basket center) ---
        target_above_rim = basket_bottom.clone()
        target_above_rim[:, 2] = rim_z + 0.05                           # 5 cm above rim
        target_inside = basket_bottom.clone()
        target_inside[:, 2] = base_z + 0.02                             # 2 cm above bottom (avoid snagging)

        # --- Reward accumulator ---
        reward = torch.zeros(N, device=device, dtype=torch.float32)

        # (1) Reach the object
        d_tcp_obj = torch.linalg.norm(tcp_pos - pos_obj, dim=1)
        r_reach = 1.0 - torch.tanh(3.0 * d_tcp_obj)
        reward += 2.0 * r_reach

        # (2) Discrete grasp bonus
        reward += 2.0 * is_grasped.float()

        # (3) Lift while grasped (relative to an estimate of the table height)
        # If table top z is known, use that; here we approximate with the basket bottom z.
        table_guess_z = base_z                                          # (N,)
        lift_amount = obj_bottom_z - (table_guess_z + lift_clearance)   # (N,)
        r_lift = torch.clamp(lift_amount * 30.0, min=0.0)               # scaled linear ramp
        reward += 2.0 * r_lift * is_grasped.float()

        # (4) Move ABOVE RIM — gate by lift (encourages lifting before moving toward basket)
        g_lift = torch.sigmoid(20.0 * (obj_bottom_z - (table_guess_z + lift_clearance)))  # (N,)
        d_obj_to_above_rim = torch.linalg.norm(pos_obj - target_above_rim, dim=1)
        r_to_above_rim = 1.0 - torch.tanh(3.0 * d_obj_to_above_rim)
        reward += 2.0 * (g_lift * r_to_above_rim)

        # (5) Go INSIDE (below rim) — gate activates near the rim height
        g_inside = torch.sigmoid(12.0 * (obj_bottom_z - (rim_z - 0.01)))
        d_obj_to_inside = torch.linalg.norm(pos_obj - target_inside, dim=1)
        r_inside = 1.0 - torch.tanh(4.0 * d_obj_to_inside)
        reward += 3.0 * (g_inside * r_inside)

        # (6) Release & stabilize (only after object is inside and released)
        v = torch.linalg.norm(self.pick_obj.linear_velocity, dim=1)
        av = torch.linalg.norm(self.pick_obj.angular_velocity, dim=1)
        r_obj_static = 1.0 - torch.tanh(5.0 * v + 2.0 * av)

        # Robot joint velocity norm; make shape robust if API returns (D,)
        qv = getattr(self.agent.robot, "qvel", None)
        if qv is None:
            qv = self.agent.robot.get_qvel()
        if qv.ndim == 1:
            qv = qv.unsqueeze(0).expand(N, -1)
        r_robot_static = 1.0 - torch.tanh(5.0 * torch.linalg.norm(qv, dim=1))

        post_release_mask = inside & (~is_grasped)
        reward += 3.0 * post_release_mask.float() * (0.5 * r_obj_static + 0.5 * r_robot_static)

        # (7) Dragging penalty: discourage moving toward basket (in XY) BEFORE lifting enough
        g_not_lifted = 1.0 - torch.sigmoid(20.0 * (obj_bottom_z - (table_guess_z + lift_clearance)))
        near_basket_xy = 1.0 - torch.tanh(5.0 * xy_dist)                # larger when closer in XY
        drag_penalty = 0.5 * is_grasped.float() * g_not_lifted * near_basket_xy
        reward -= drag_penalty

        # (8) Final success bonus
        reward += 8.0 * success.float()

        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalize dense reward to a ~[0, 2] range for stability (adjust the divisor after inspecting logs).
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20.0