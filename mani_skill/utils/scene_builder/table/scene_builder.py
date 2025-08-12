import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs import Actor


class TableSceneBuilder(SceneBuilder):
    """A simple scene builder that adds a table to the scene such that the height of the table is at 0, and
    gives reasonable initial poses for robots."""

    def __init__(self, env, robot_init_qpos_noise=0.02, custom_table=False, custom_basket=False, randomize_colors=False, basket_color="orange"):
        super().__init__(env, robot_init_qpos_noise)
        self.custom_table = custom_table
        self.custom_basket = custom_basket
        self.randomize_colors = randomize_colors
        self.basket_color = basket_color

    def _build_custom_table(self, length: float, width: float, height: float, random_i = -1):
        """
        Build a custom table with specified dimensions.
        
        Args:
            length: Length of the table (x-axis)
            width: Width of the table (y-axis) 
            height: Height of the table (z-axis)
            
        Returns:
            The built table actor
        """
        # Create actor builder for collision and visual
        builder = self.scene.create_actor_builder()
        
        # Add box collision for the entire table (tabletop + legs)
        table_half_size = [width/2, length/2, height/2]  # half dimensions
        table_pose = sapien.Pose(p=[0, 0, height/2])  # center of the table
        builder.add_box_collision(table_pose, table_half_size)
        
        # Tabletop
        if random_i >= 0:
            tabletop_material = sapien.render.RenderMaterial(base_color=self.env._batched_episode_rng[random_i].uniform(low=0., high=0.5, size=(3, )).tolist() + [1])
            builder.set_scene_idxs([random_i])
        else:
            # default: black tabletop
            tabletop_material = sapien.render.RenderMaterial(base_color=[0.1, 0.1, 0.1, 1.0])
        tabletop_thickness = 0.05
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, height - tabletop_thickness/2]),
            half_size=[width/2, length/2, tabletop_thickness/2],
            material=tabletop_material
        )
        
        # Table legs
        leg_height = height - tabletop_thickness/2
        leg_margin = 0.03  # margin from corners
        leg_size = 0.05  # square legs
        
        leg_positions = [
            [width/2 - leg_margin, length/2 - leg_margin, leg_height/2],   # front right
            [width/2 - leg_margin, -length/2 + leg_margin, leg_height/2],  # front left
            [-width/2 + leg_margin, length/2 - leg_margin, leg_height/2],  # back right
            [-width/2 + leg_margin, -length/2 + leg_margin, leg_height/2], # back left
        ]
        
        if random_i >= 0:
            leg_material = sapien.render.RenderMaterial(base_color=self.env._batched_episode_rng[random_i].uniform(low=0.5, high=1., size=(3, )).tolist() + [1])
        else:
            leg_material = sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1.0])
        for leg_pos in leg_positions:
            builder.add_box_visual(
                pose=sapien.Pose(p=leg_pos),
                half_size=[leg_size/2, leg_size/2, leg_height/2],
                material=leg_material
            )
        builder.initial_pose = sapien.Pose(p=[0, 0, -height])
        # Build the final table actor
        if random_i >= 0:
            table_actor = builder.build_kinematic(name=f"table-custom-{random_i}")
        else:
            table_actor = builder.build_kinematic(name="table-custom")
        return table_actor
    
    def _build_custom_basket(self, color="orange", table_center_x=0.0, table_center_y=0.0, table_surface_z=0.0, random_i=-1):
        """
        Build a custom basket positioned at the center of the table.
        
        Args:
            color: Color of the basket (used when random_i < 0)
            table_center_x: X coordinate of table center
            table_center_y: Y coordinate of table center  
            table_surface_z: Z coordinate of the table surface
            random_i: Environment index for randomization (-1 for no randomization)
        """
        color_map = {
            "blue":    [0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1],
            "orange":  [1.0, 0.4980392156862745, 0.054901960784313725, 1],
            "green":   [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1],
            "red":     [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1],
            "yellow":  [1.0, 0.8941176470588236, 0.0, 1],
            "purple":  [0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1],
            "brown":   [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1],
            "pink":    [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1],
            "gray":    [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1],
            "olive":   [0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1],
            "cyan":    [0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1],
        }
        self.basket_model_path = str(PACKAGE_ASSET_DIR / "custom/plastic_basket.obj")

        basket_builder = self.scene.create_actor_builder()
        basket_mat = sapien.render.RenderMaterial()
        
        # Set basket color - randomized or fixed
        if random_i >= 0:
            # Randomize color
            random_color = self.env._batched_episode_rng[random_i].uniform(low=0., high=1., size=(3, )).tolist() + [1]
            basket_mat.set_base_color(random_color)
            basket_builder.set_scene_idxs([random_i])
        else:
            # Use fixed color
            basket_mat.set_base_color(color_map[color])
            
        basket_mat.metallic = 0.0
        basket_mat.roughness = 1.0
        basket_mat.specular = 1.0
        
        basket_builder.add_visual_from_file(
            self.basket_model_path,
            scale=[0.003]*3,
            material=basket_mat
        )
        
        basket_builder.add_convex_collision_from_file(
            filename=self.basket_model_path,
            scale=[0.003]*3,
            density=5000
        )
        
        # Position basket at center of table, on top of the table surface
        # The basket's bottom should sit on the table surface
        basket_z = table_surface_z + 0.02  # Small offset above table surface
        basket_pose = sapien.Pose(
            p=[table_center_x, table_center_y, basket_z],
            q=euler2quat(np.pi/2, 0, 0)  # 90 degree rotation around X-axis to lay basket horizontally
        )
        basket_builder.initial_pose = basket_pose
        
        if random_i >= 0:
            basket = basket_builder.build(name=f"basket_{random_i}")
        else:
            basket = basket_builder.build(name=f"basket_{color}")
        return basket
    
    def _build_custom_wall(self, table_pose: sapien.Pose, length: float):
        wall_size = [0.2, 10.0, 4.0]
        wall_half = [s / 2 for s in wall_size]

        wall_offset = 0.25
        table_back_x = table_pose.p[0] - length / 2
        wall_x = table_back_x - wall_half[0] - wall_offset
        wall_pose = sapien.Pose(p=[wall_x, 0, 0])

        white_mat = sapien.render.RenderMaterial(
            base_color=[1, 1, 1, 1], roughness=0.9, metallic=0.0
        )

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(pose=sapien.Pose(), half_size=wall_half)
        builder.add_box_visual   (pose=sapien.Pose(), half_size=wall_half, material=white_mat)

        builder.initial_pose = wall_pose
        wall_actor = builder.build_static(name="white_wall")
        
        return wall_actor

    def build(self):
        if self.custom_table:
            # Use custom table with specified dimensions - height of the glb table, length and width matching real table
            self.table_height = 0.91964292762787
            if self.randomize_colors:
                # Build tables separately for each parallel environment to enable domain randomization        
                self._tables: List[Actor] = []
                for i in range(self.env.num_envs):
                    self._tables.append(self._build_custom_table(length=1.52, width=0.76, height=self.table_height, random_i=i))
                    self.env.remove_from_state_dict_registry(self._tables[-1])  # remove individual table from state dict

                # Merge all tables into a single Actor object
                self.table = Actor.merge(self._tables, name="table")
            else:
                self.table = self._build_custom_table(length=1.52, width=0.76, height=self.table_height)
            table_pose_world = sapien.Pose(p=[0, 0, self.table_height/2])
            self.wall  = self._build_custom_wall(table_pose_world, length=1.52)

        else:
            # Use default GLB table
            builder = self.scene.create_actor_builder()
            model_dir = Path(osp.dirname(__file__)) / "assets"
            table_model_file = str(model_dir / "table.glb")
            scale = 1.75

            table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
            # builder.add_nonconvex_collision_from_file(
            #     filename=table_model_file,
            #     scale=[scale] * 3,
            #     pose=table_pose,
            # )
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
                half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
            )
            builder.add_visual_from_file(
                filename=table_model_file, scale=[scale] * 3, pose=table_pose
            )
            builder.initial_pose = sapien.Pose(
                p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
            )
            self.table = builder.build_kinematic(name="table-workspace")
            # aabb = (
            #     table._objs[0]
            #     .find_component_by_type(sapien.render.RenderBodyComponent)
            #     .compute_global_aabb_tight()
            # )
            # value of the call above is saved below
            aabb = np.array(
                [
                    [-0.7402168, -1.2148621, -0.91964257],
                    [0.4688596, 1.2030163, 3.5762787e-07],
                ]
            )
            self.table_length = aabb[1, 0] - aabb[0, 0]
            self.table_width = aabb[1, 1] - aabb[0, 1]
            self.table_height = aabb[1, 2] - aabb[0, 2]
        
        # Build basket if requested
        if self.custom_basket:
            if self.randomize_colors:
                # Build baskets separately for each parallel environment to enable domain randomization
                self._baskets: List[Actor] = []
                for i in range(self.env.num_envs):
                    if self.custom_table:
                        # For custom table, it will be positioned at [-0.2245382, 0, -table_height] in initialize()
                        basket = self._build_custom_basket(
                            color=self.basket_color, 
                            table_center_x=-0.2245382, 
                            table_center_y=0.0, 
                            table_surface_z=0.0,  # Table surface will be at z=0 after positioning
                            random_i=i
                        )
                    else:
                        # For default GLB table, it's positioned at [-0.12, 0, -0.9196429]
                        basket = self._build_custom_basket(
                            color=self.basket_color,
                            table_center_x=-0.12,
                            table_center_y=0.0,
                            table_surface_z=0.0,  # Table surface will be at z=0 after positioning
                            random_i=i
                        )
                    self._baskets.append(basket)
                    self.env.remove_from_state_dict_registry(self._baskets[-1])  # remove individual basket from state dict

                # Merge all baskets into a single Actor object
                self.basket = Actor.merge(self._baskets, name="basket")
            else:
                # Non-randomized basket
                if self.custom_table:
                    # For custom table, it will be positioned at [-0.2245382, 0, -table_height] in initialize()
                    # So the basket should be positioned relative to that
                    self.basket = self._build_custom_basket(
                        color=self.basket_color, 
                        table_center_x=-0.2245382, 
                        table_center_y=0.0, 
                        table_surface_z=0.0  # Table surface will be at z=0 after positioning
                    )
                else:
                    # For default GLB table, it's positioned at [-0.12, 0, -0.9196429]
                    # So table surface will be at z=0 
                    self.basket = self._build_custom_basket(
                        color=self.basket_color,
                        table_center_x=-0.12,
                        table_center_y=0.0,
                        table_surface_z=0.0  # Table surface will be at z=0 after positioning
                    )
        
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        
        # Build scene objects list
        scene_objects = [self.table, self.ground]
        if self.custom_table:
            if self.custom_basket:
                self.scene_objects: List[sapien.Entity] = [self.table, self.wall, self.basket, self.ground]
            else:
                self.scene_objects: List[sapien.Entity] = [self.table, self.wall, self.ground]
        else:
            if self.custom_basket:
                self.scene_objects: List[sapien.Entity] = [self.table, self.basket, self.ground]
            else:
                self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def cleanup(self):
        """Clean up individual table and basket actors created for domain randomization to prevent memory leaks."""
        if hasattr(self, '_tables'):
            # Remove individual tables from the scene
            for table in self._tables:
                if hasattr(table, 'entity') and table.entity is not None:
                    self.env.scene.remove_actor(table)
            self._tables.clear()
            
        if hasattr(self, '_baskets'):
            # Remove individual baskets from the scene
            for basket in self._baskets:
                if hasattr(basket, 'entity') and basket.entity is not None:
                    self.env.scene.remove_actor(basket)
            self._baskets.clear()

    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        if self.custom_table:
            self.table.set_pose(sapien.Pose(p=[-0.2245382, 0, -self.table_height]))
        else:
            self.table.set_pose(
                sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
            )
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right",
            "xarm6_robotiq",
            "xarm6_nogripper",
            "xarm6_pandagripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "floating_robotiq_2f_85_gripper":
            qpos = self.env.agent.keyframes["open_facing_side"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.5, 0, 0.05]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in ["widowxai", "widowxai_wristcam"]:
            qpos = self.env.agent.keyframes["ready_to_grasp"].qpos
            self.env.agent.reset(qpos)
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, 1.0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
