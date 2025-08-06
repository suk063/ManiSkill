import argparse
import numpy as np
import open3d as o3d
from transforms3d.quaternions import mat2quat
from typing import List
import sapien
from pathlib import Path
import mani_skill.envs.tasks.tabletop.table_scan_discrete_no_robot
import gymnasium as gym
import cv2
from tqdm import tqdm
import glob

# MARK: Parse command line arguments
parser = argparse.ArgumentParser(description="Scan an object and generate a point cloud.")
parser.add_argument("-o", "--obs-mode", type=str, default="rgbd", help="Can be rgb or rgb+depth, rgb+normal, albedo+depth etc. Which ever image-like textures you want to visualize can be tacked on")
parser.add_argument("-n", "--num-envs", type=int, default=100, help="Total number of environments to process (overridden if --grid-dim is given)")
parser.add_argument("--grid-dim", type=int, default=5, help="If provided, total_envs will be grid_dim^2 and passed to the environment.")
parser.add_argument("-b", "--batch-size", type=int, default=25, help="How many envs to load at once.")
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

def visualize_point_cloud(pcd: o3d.geometry.PointCloud):
    """Visualize point cloud with coordinate axes."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    # Set viewing parameters
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0]) # Set Y-down as up vector for better initial view
    ctr.set_front([0, 0, -1])

    print("Visualizing point cloud. Close the window to continue.")
    vis.run()
    vis.destroy_window()


# ------------------------------------------------------------------
# Adjust num_envs if grid_dim provided
# ------------------------------------------------------------------
if args.grid_dim is not None:
    args.num_envs = args.grid_dim ** 2
assert args.batch_size > 0, "batch-size must be positive"

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


# ------------------------------------------------------------------ #
# Prepare output directories for the current batch
# ------------------------------------------------------------------ #
for i in range(args.num_envs):
    env_dir = DATASET_DIR / f"env_{i:03d}"
    (env_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (env_dir / "depth").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Prepare containers for multiview point cloud aggregation
# ------------------------------------------------------------------ #
def process_batch(start_idx: int, n_envs: int):
    env = gym.make(
        "TableScanDiscreteNoRobot-v0",
        robot_uids="none",
        obs_mode=args.obs_mode,
        num_envs=n_envs,
        grid_dim=int(np.ceil(np.sqrt(n_envs))),
        control_mode="pd_joint_pos",
        reward_mode="none",
    )
    env.reset(seed=args.seed)
    unwrapped = env.unwrapped

    pts_batch  = [[] for _ in range(n_envs)]
    rgb_batchc = [[] for _ in range(n_envs)]

    for pose_idx, pose_mat_w2c_cv in enumerate(tqdm(all_poses_w2c_cv, desc=f"Batch {start_idx:03d}-{start_idx+n_envs-1:03d} poses", leave=False)):
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

        cam.camera.take_picture()
        cam_obs = cam.get_obs()
        rgb_b = cam_obs["rgb"].cpu().numpy()       # (B,H,W,3)
        depth_b = cam_obs["depth"].cpu().numpy()
        K_b = cam.camera.get_intrinsic_matrix()
        E_b = cam.camera.get_extrinsic_matrix()

        if start_idx == 0 and pose_idx == 0:
            global K_env0
            K_env0 = K_b[0].cpu().numpy()

        # ---------- save + point-cloud per env in *this batch* ----------
        for local_idx in range(n_envs):
            gi = start_idx + local_idx              # global env index
            env_dir = DATASET_DIR / f"env_{gi:03d}"

            rgb_np   = rgb_b[local_idx].astype(np.uint8)
            depth_np = depth_b[local_idx]
            cv2.imwrite(str(env_dir / "rgb" / f"{pose_idx:04d}.png"),
                        cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(env_dir / "depth" / f"{pose_idx:04d}.png"),
                        depth_np.astype(np.uint16))

            if np.max(depth_np) == 0:
                continue
            K   = K_b[local_idx].cpu().numpy()
            Ecv = E_b[local_idx][:3].cpu().numpy()  # (3,4) world→cam (OpenCV)
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
            pts_batch[local_idx].append(np.asarray(pcd.points))
            rgb_batchc[local_idx].append(np.asarray(pcd.colors))

    env.close()
    return pts_batch, rgb_batchc

for start in tqdm(range(0, args.num_envs, args.batch_size), desc="Batches", unit="batch"):
    size = min(args.batch_size, args.num_envs - start)
    try:
        # Process a batch and get its point cloud data
        pts_b, rgb_b = process_batch(start, size)

        # For each environment in the batch, aggregate and save its data immediately
        for li in range(size):
            gi = start + li
            env_dir = DATASET_DIR / f"env_{gi:03d}"
            
            # Check if any points were captured for this environment
            if not pts_b[li]:
                print(f"No points captured for env_{gi:03d}, skipping save.")
                continue

            # Aggregate all points and colors for this one environment
            points = np.vstack(pts_b[li])
            colors = np.vstack(rgb_b[li])

            # Create an Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Downsample before saving to reduce file size -> for mapping we don't use this anyways
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.005)

            # Save the final point cloud to a file
            output_path = env_dir / "point_cloud.ply"
            o3d.io.write_point_cloud(str(output_path), pcd_downsampled)
            # print(f"Saved aggregated point cloud for env_{gi:03d} to {output_path}")

    finally:
        # guarantee cleanup even if an exception occurs
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

if K_env0 is not None:
    intrinsic_path = DATASET_DIR / "intrinsic.txt"
    np.savetxt(intrinsic_path, K_env0, fmt="%.8f")
    print(f"Saved intrinsics for env_000 to {intrinsic_path}")

# ------------------------------------------------------------------ #
# Reconstruct and visualize point cloud for the first environment
# ------------------------------------------------------------------ #
pcd_path_env0 = DATASET_DIR / "env_000" / "point_cloud.ply"
if pcd_path_env0.exists():
    print(f"\nLoading saved point cloud from {pcd_path_env0} for visualization...")
    pcd_downsampled = o3d.io.read_point_cloud(str(pcd_path_env0))
    visualize_point_cloud(pcd_downsampled)
else:
    print("\nNo point cloud file found for env_000, skipping visualization.")


print("Done.")

