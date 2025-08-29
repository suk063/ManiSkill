import argparse
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from transforms3d.quaternions import mat2quat
from typing import List
import sapien
from pathlib import Path
import mani_skill.envs.tasks.tabletop.pick_ycb_custom_no_robot
import gymnasium as gym
import cv2
import glob

# MARK: Parse command line arguments
parser = argparse.ArgumentParser(description="Scan an object and generate a point cloud.")
parser.add_argument("-o", "--obs-mode", type=str, default="rgbd", help="Can be rgb or rgb+depth, rgb+normal, albedo+depth etc. Which ever image-like textures you want to visualize can be tacked on")
parser.add_argument("-n", "--num-envs", type=int, default=120, help="Total number of environments to process")
parser.add_argument("-s","--seed",type=int, default=0, help="Seed the random actions and environment. Default is no seed",)
args = parser.parse_args()

# MARK: Setup directories and paths
SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DIR   = SCRIPT_DIR / "poses"
DATASET_DIR = SCRIPT_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

# MARK: Auxiliary functions
def load_poses_from_npy(path: Path) -> List[np.ndarray]:
    """Return list of (3,4) world->cam matrices from .npy files"""
    pose_files = sorted(glob.glob(str(path / "*.npy")))
    if not pose_files:
        raise FileNotFoundError(f"No .npy pose files found in {path}")
    return [np.load(f) for f in pose_files]


# ------------------------------------------------------------------
if args.seed is not None:
    np.random.seed(args.seed)

# MARK: Capture images and point cloud for multiple episodes
try:
    all_poses_w2c_cv = load_poses_from_npy(POSE_DIR)
    print(f"Loaded {len(all_poses_w2c_cv)} poses from {POSE_DIR}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run the pose generation script first (generate_pose.py).")
    exit()

# Transformation matrix from OpenCV camera space (X-right, Y-down, Z-forward)
# to SAPIEN camera space (X-forward, Y-left, Z-up)
CV_TO_SAPIEN_AXES = np.array([
    [0., -1.,  0.], # SAPIEN Y is OpenCV -X
    [0.,  0., -1.], # SAPIEN Z is OpenCV -Y
    [1.,  0.,  0.]  # SAPIEN X is OpenCV  Z
], dtype=np.float32)

def tile(frames: List[np.ndarray], nrows: int) -> np.ndarray:
    h, w, _ = frames[0].shape
    ncols   = int(np.ceil(len(frames) / nrows))
    grid    = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for idx, f in enumerate(frames):
        r, c = divmod(idx, ncols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = f
    return grid

GRID_ROWS = int(np.ceil(np.sqrt(args.num_envs)))
K_env0 = None

env = gym.make(
    "PickYCBCustomNoRobot-v1",
    obs_mode=args.obs_mode,
    num_envs=args.num_envs,
    control_mode="pd_joint_pos",
    reward_mode="none",
)
# Use a different seed to get different objects
env.reset(seed=args.seed)

# ------------------------------------------------------------------ #
# Prepare output directories for the current batch
# ------------------------------------------------------------------ #
for i in range(args.num_envs):
    env_dir = DATASET_DIR / f"env_{i:03d}"
    (env_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (env_dir / "depth").mkdir(parents=True, exist_ok=True)
    (env_dir / "global_idx.txt").write_text(str(i))

# ------------------------------------------------------------------ #
# Prepare containers for multiview point cloud aggregation
# ------------------------------------------------------------------ #
pts_all: List[List[np.ndarray]]  = [[] for _ in range(args.num_envs)]
rgb_all: List[List[np.ndarray]]  = [[] for _ in range(args.num_envs)]
unwrapped = env.unwrapped
grid_frames: List[np.ndarray] = []

for pose_idx, pose_mat_w2c_cv in enumerate(all_poses_w2c_cv):
    # Convert OpenCV world-to-camera extrinsic to SAPIEN camera-to-world pose
    R_w2c_cv = pose_mat_w2c_cv[:3, :3]
    t_w2c_cv = pose_mat_w2c_cv[:3, 3]

    # Inverse to get camera-to-world in OpenCV convention
    R_c2w_cv = R_w2c_cv.T
    t_c2w_cv = -R_w2c_cv.T @ t_w2c_cv

    # Apply coordinate system transformation for SAPIEN
    R_c2w_sapien = R_c2w_cv @ CV_TO_SAPIEN_AXES
    
    pose = sapien.Pose(p=t_c2w_cv, q=mat2quat(R_c2w_sapien))
    
    unwrapped.cam_mount.set_pose(pose)
    unwrapped.scene.update_render()
    cam = unwrapped.scene.human_render_cameras["moving_camera"]
    
    # Trigger rendering
    cam.camera.take_picture()

    # ------------------------------------------------------------------ #
    # Fetch sensor data & parameters for the entire batch
    # ------------------------------------------------------------------ #
    cam_obs    = cam.get_obs()
    rgb_batch   = cam_obs["rgb"].cpu().numpy()       # (B, H, W, 3)
    depth_batch = cam_obs["depth"].cpu().numpy()      # (B, H, W)
    
    K_batch  = cam.camera.get_intrinsic_matrix()
    if pose_idx == 0 and K_env0 is None:
        K_env0 = K_batch[0].cpu().numpy()
        
    E_batch  = cam.camera.get_extrinsic_matrix()
    
    # ------------------------------------------------------------------ #
    # Accumulate RGB frames for video grid (unchanged behaviour)
    # ------------------------------------------------------------------ #
    frames = list(rgb_batch)
    grid   = tile(frames, nrows=GRID_ROWS)
    grid_frames.append(grid)
    
    # ------------------------------------------------------------------ #
    # Build point clouds per-environment
    # ------------------------------------------------------------------ #
    for env_idx in range(args.num_envs):
        env_dir = DATASET_DIR / f"env_{env_idx:03d}"
        
        rgb_np   = rgb_batch[env_idx].astype(np.uint8)
        depth_np = depth_batch[env_idx]

        # Save RGB image
        rgb_path = env_dir / "rgb" / f"{pose_idx:04d}.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # Save depth image (it is already in mm, save as 16-bit PNG)
        depth_mm_uint16 = depth_np.astype(np.uint16)
        depth_path = env_dir / "depth" / f"{pose_idx:04d}.png"
        cv2.imwrite(str(depth_path), depth_mm_uint16)

        # Skip if depth is invalid (all zeros)
        if np.max(depth_np) == 0:
            continue
            
        K   = K_batch[env_idx].cpu().numpy()
        Ecv = E_batch[env_idx][:3].cpu().numpy()  # (3,4) world→cam (OpenCV)
        H, W, _ = rgb_np.shape
        rgb_o3d   = o3d.geometry.Image(rgb_np)
        depth_o3d = o3d.geometry.Image(depth_np)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0,0], K[1,1], K[0,2], K[1,2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        if not pcd.has_points():
            continue
            
        # Transform from camera to world
        cam_to_world = np.linalg.inv(np.vstack([Ecv, [0,0,0,1]]))
        pcd.transform(cam_to_world)
        pts_all[env_idx].append(np.asarray(pcd.points))
        rgb_all[env_idx].append(np.asarray(pcd.colors))


env.close()

if K_env0 is not None:
    intrinsic_path = DATASET_DIR / "intrinsic.txt"
    np.savetxt(intrinsic_path, K_env0, fmt="%.8f")
    print(f"Saved intrinsics for env_000 to {intrinsic_path}")

# ------------------------------------------------------------------ #
# Reconstruct and visualize point cloud for the first environment
# ------------------------------------------------------------------ #
if pts_all[0]:
    print("\nReconstructing point cloud for env_000 for visualization...")
    pcd_env0 = o3d.geometry.PointCloud()
    pcd_env0.points = o3d.utility.Vector3dVector(np.vstack(pts_all[0]))
    pcd_env0.colors = o3d.utility.Vector3dVector(np.vstack(rgb_all[0]))
    
    pcd_downsampled = pcd_env0.voxel_down_sample(voxel_size=0.005)
    print(f"Original points: {len(pcd_env0.points)}, downsampled to {len(pcd_downsampled.points)}")
    
    points = np.asarray(pcd_downsampled.points)
    colors = np.asarray(pcd_downsampled.colors)
    
    # Filter points to visualize only those within the specified boundary
    scene_min = np.array([-0.8, -1.0, -0.1])
    scene_max = np.array([0.4,  1.0,  0.3])
    mask = np.all((points >= scene_min) & (points <= scene_max), axis=1)
    points = points[mask]
    colors = colors[mask]
    print(f"Filtered to {len(points)} points for visualization within the boundary.")
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.8
        )
    )])

    
    # Set aspect ratio and labels
    x_range = [points[:, 0].min(), points[:, 0].max()]
    y_range = [points[:, 1].min(), points[:, 1].max()]
    z_range = [points[:, 2].min(), points[:, 2].max()]
    max_range = np.array([x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0]]).max() / 2.0
    
    mid_x = (x_range[0] + x_range[1]) / 2.0
    mid_y = (y_range[0] + y_range[1]) / 2.0
    mid_z = (z_range[0] + z_range[1]) / 2.0
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Point Cloud Visualization"
    )
    
    output_path = DATASET_DIR / "point_cloud.html"
    fig.write_html(output_path)
    print(f"Point cloud visualization saved to {output_path}")

else:
    print("\nNo points captured for env_000, skipping visualization.")


print("Done.")

