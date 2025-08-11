from typing import Dict, Any, Union, List
import numpy as np, torch, sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult
from mani_skill.envs.sapien_env import BaseEnv
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.utils.structs import Pose, GPUMemoryConfig, SimConfig
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import XArm6Robotiq
from mani_skill.utils.structs import Actor

def look_at_with_extra_tilt(cam_pos, target, tilt_deg):
    """
    Pose that looks at *target* and then pitches downward (camera-local X)
    by tilt_deg degrees.
    """
    base_pose = sapien_utils.look_at(cam_pos, target)
    base_p = np.asarray(base_pose.p, dtype=np.float32).reshape(3)     # (3,)
    base_q = np.asarray(base_pose.q, dtype=np.float32).reshape(4)     # (4,)

    tilt_q = np.asarray(
        euler2quat(np.deg2rad(tilt_deg), 0, 0, 'sxyz'), dtype=np.float32
    )

    new_q  = qmult(base_q, tilt_q).astype(np.float32)

    return sapien.Pose(base_p, new_q)

@register_env("TableScan-v0", max_episode_steps=1_000)
class TableScanEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["xarm6_robotiq"]
    agent: XArm6Robotiq
    
    # Constants
    cube_half_size = 0.02
    cube_spawn_half_size = 0.1
    cube_spawn_center = (0, 0)
        
    CAM_RADII = [0.2, 0.3, 0.4]
    VIEW_HEIGHTS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    CAM_SPEED  = np.deg2rad(1) # rad / sim-step (≈115 °/s at 60 Hz)
    ANGLE_STEP_DEG = 10
    ANGLE_STEP_RAD = np.deg2rad(ANGLE_STEP_DEG)

    MIN_TILT_DEG = 20          # when cam is at min height
    MAX_TILT_DEG = 40         # when cam is at max height

    THETA_MIN    = np.deg2rad(-150)
    THETA_MAX    = np.deg2rad( 150)

    def __init__(self, *args, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["xarm6_robotiq"]
        self.cube_half_size = cfg["cube_half_size"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        """Hand-mounted camera 25 cm in front of the TCP, looking forward."""
        offset = [0.15, 0.0, -0.2]                     
        down90 = euler2quat(0,  -np.pi/2, 0)        
        hand_cam_pose = sapien.Pose(offset, down90)
        real_pose = sapien_utils.look_at(eye=[0.508, -0.5, 0.42], target=[-0.522, 0.2, 0])
        moving_camera = CameraConfig(
                            "moving_camera", pose=sapien.Pose(), width=224, height=224, 
                            fov=np.pi * 0.4, near=0.01, far=100, 
                            mount=self.cam_mount
                        )

        return [
            # CameraConfig(
            #     "hand_cam",
            #     pose=hand_cam_pose,
            #     width=128,
            #     height=128,
            #     fov=np.pi * 0.4,
            #     near=0.01,
            #     far=100,
            #     mount=self.agent.tcp,
            # )
            # CameraConfig(
            #     "hand_cam",
            #     pose=real_pose,
            #     # width=640,
            #     # height=480,
            #     width=128,
            #     height=128,
            #     fov=np.pi * 0.4,
            #     near=0.01,
            #     far=100,
            # )
            moving_camera
        ]
        
    def _before_simulation_step(self):
        super()._before_simulation_step()

        cam_pos = self.table_center + np.array([
            self._cam_radius * np.cos(self._cam_theta),
            self._cam_radius * np.sin(self._cam_theta),
            self._cam_height,
        ])

        z_min, z_max = self.VIEW_HEIGHTS[0], self.VIEW_HEIGHTS[-1]
        alpha        = (self._cam_height - z_min) / (z_max - z_min)
        tilt_deg     = self.MIN_TILT_DEG + alpha * (self.MAX_TILT_DEG - self.MIN_TILT_DEG)

        self.cam_mount.set_pose(
            look_at_with_extra_tilt(cam_pos, self.table_center, tilt_deg)
        )

        self._angle_idx += 1
        self._cam_theta  = self.THETA_MIN + self._angle_idx * self.ANGLE_STEP_RAD

        if self._cam_theta > self.THETA_MAX:
            self._angle_idx = 0
            self._h_idx = (self._h_idx + 1) % len(self.VIEW_HEIGHTS)
            self._cam_height = self.VIEW_HEIGHTS[self._h_idx]
            if self._h_idx == 0:
                self._r_idx = (self._r_idx + 1) % len(self.CAM_RADII)
                self._cam_radius = self.CAM_RADII[self._r_idx]
            self._cam_theta = self.THETA_MIN

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    @property
    def _supports_sensor_data(self):
        return True

    def _load_agent(self, options: Dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: Dict):
        self.table_scene = TableSceneBuilder(env=self, custom_table=True, randomize_colors=False)
        self.table_scene.build()
        self.table_center = [0, 0, self.table_scene.table_height/2]
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
        
        self._angle_idx  = 0

        self._theta_min  = self.THETA_MIN
        self._theta_max  = self.THETA_MAX
        self._cam_theta  = self._theta_min         
        self._cam_speed  = self.CAM_SPEED

        self._h_idx      = 0
        self._r_idx      = 0
        self._cam_height = self.VIEW_HEIGHTS[self._h_idx]
        self._cam_radius = self.CAM_RADII[self._r_idx]

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

    def _reconfigure(self, options=dict()):
        """Clean up individual actors created for domain randomization to prevent memory leaks during resets."""
        if hasattr(self, '_cubes'):
            # Remove individual cubes from the scene
            for cube in self._cubes:
                if hasattr(cube, 'entity') and cube.entity is not None:
                    self.scene.remove_actor(cube)
            self._cubes.clear()
        
        # Clean up table scene builder if it exists
        if hasattr(self, 'table_scene'):
            self.table_scene.cleanup()

        super()._reconfigure(options)
        
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            xyz = torch.zeros((b, 3))
            
            grid_idx = env_idx % 100
            x_grid = grid_idx // 10
            y_grid = grid_idx % 10
            xyz[:, 0] = torch.linspace(0, 1, 10)[x_grid] * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            xyz[:, 1] = torch.linspace(0, 1, 10)[y_grid] * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
 
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, *_, **__):
        return torch.zeros(self.num_envs, device=self.device)

    def get_table_center(self):
        return self.table_center