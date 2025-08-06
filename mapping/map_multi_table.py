import os, time, argparse, glob
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open_clip
import open3d as o3d
import plotly.graph_objs as go
import plotly.offline as pyo
import random
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from collections import deque
from statistics import mean
from tqdm import tqdm

# local modules
from mapping_lib.utils import get_visual_features, get_3d_coordinates, transform
from mapping_lib.voxel_hash_table import VoxelHashTable
from mapping_lib.implicit_decoder import ImplicitDecoder

# --------------------------------------------------------------------------- #
#  Dataset Class                                                              #
# --------------------------------------------------------------------------- #
class MultiEnvDataset(Dataset):
    def __init__(self, samples, world_to_cam_poses, transform_func):
        self.samples = samples
        self.world_to_cam = world_to_cam_poses
        self.transform_func = transform_func

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, env_name, pose_idx = self.samples[idx]

        rgb_np = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_np is None:
            raise FileNotFoundError(f"Failed to read {rgb_path}")

        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)

        depth_np = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_np is None or np.max(depth_np) == 0:
            raise FileNotFoundError(f"Failed to read {depth_path} or depth is empty")

        img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        img_tensor = self.transform_func(img_tensor)

        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float() / 1000.0
        depth_t = F.interpolate(depth_t, (16, 16), mode="nearest-exact").squeeze()

        extrinsic_t = torch.from_numpy(self.world_to_cam[pose_idx]).float()

        sample = {
            "img_tensor":  img_tensor,
            "depth_t":     depth_t,
            "extrinsic_t": extrinsic_t,
            "env_name":    env_name,
        }
        return sample


# --------------------------------------------------------------------------- #
#  Arguments                                                                  #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Map multiple environments using pre-generated data.")
parser.add_argument(
    "--dataset-dir",
    type=str,
    default="mapping/dataset",
    help="Path to the dataset directory with env_xxx subfolders."
)
parser.add_argument(
    "--poses-dir",
    type=str,
    default="mapping/poses",
    help="Path to the directory with .npy pose files."
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="mapping/multi_env_maps",
    help="Directory to save the trained grids and decoder."
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Number of training epochs per environment."
)
parser.add_argument(
    "--save",
    action="store_true",
    default=True,
    help="Save the trained voxel grids & shared decoder."
)
parser.add_argument(
    "--pca",
    action="store_true",
    help="Generate PCA visualization of voxel features for the specified environment(s)."
)
parser.add_argument(
    "--vis-fine-grid",
    action="store_true",
    help="Generate visualization of the finest voxel grid vertices for the specified environment(s)."
)
parser.add_argument(
    "--envs",
    type=str,
    nargs='+',
    default=['env_000', 'env_001', 'env_002', 'env_003', 'env_004', 'env_005', 'env_006', 'env_007', 'env_008', 'env_009'],
    help="List of environment directory names to visualize (e.g., env_000 env_001)."
)
parser.add_argument(
    "--decoder-path",
    type=str,
    default="pretrained/implicit_decoder.pt",
    help="Path to pre-trained decoder weights (implicit decoder)."
)
args = parser.parse_args()

# --------------------------------------------------------------------------- #
#  Device / CLIP model                                                        #
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name  = "EVA02-L-14"
clip_weights_id  = "merged2b_s4b_b131k"

clip_model, _, _ = open_clip.create_model_and_transforms(
    clip_model_name, pretrained=clip_weights_id
)
clip_model = clip_model.to(DEVICE).eval()

# --------------------------------------------------------------------------- #
#  Shared Decoder                                                             #
# --------------------------------------------------------------------------- #
GRID_LVLS         = 2
GRID_FEAT_DIM     = 64
decoder = ImplicitDecoder(
    voxel_feature_dim=GRID_FEAT_DIM * GRID_LVLS,
    hidden_dim=240,
    output_dim=768,
).to(DEVICE)

# ----------------------------------------------------------------------- #
#  Load pre-trained decoder weights if available                          #
# ----------------------------------------------------------------------- #
if args.decoder_path and os.path.isfile(args.decoder_path):
    try:
        state_dict = torch.load(args.decoder_path, map_location=DEVICE)['model']
        decoder.load_state_dict(state_dict)
        print(f"[INIT] Loaded pre-trained decoder weights from {args.decoder_path}")
    except Exception as e:
        print(f"[INIT] Failed to load pre-trained decoder weights from {args.decoder_path}: {e}")
else:
    print(f"[INIT] No pre-trained decoder weights found at {args.decoder_path}. Using random initialization.")

OPT_LR = 1e-3

# --------------------------------------------------------------------------- #
#  Scene bounds (should match map_table.py)                                   #
# --------------------------------------------------------------------------- #
SCENE_MIN = (-0.8, -1.0, -0.3)
SCENE_MAX = (0.4,  1.0,  0.3)

# --------------------------------------------------------------------------- #
#  Helper functions                                                           #
# --------------------------------------------------------------------------- #
def load_w2c_poses_from_dir(dir_path: str) -> np.ndarray:
    """Loads world-to-camera poses from .npy files, returns (N, 3, 4) matrices."""
    pose_files = sorted(glob.glob(os.path.join(dir_path, "*.npy")))
    if not pose_files:
        raise FileNotFoundError(f"No .npy pose files found in {dir_path}")
    poses = [np.load(f) for f in pose_files]
    return np.stack(poses, axis=0)

def chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def build_grids(shard_dirs, device):
    grids = {}
    for d in shard_dirs:
        grids[d.name] = VoxelHashTable(
            resolution=0.12,
            num_levels=GRID_LVLS,
            feature_dim=GRID_FEAT_DIM,
            scene_bound_min=SCENE_MIN,
            scene_bound_max=SCENE_MAX,
            device=device,
            mode="train",
        )
    return grids

def collect_samples(shard_dirs):
    samples = []
    for d in shard_dirs:
        env_name = d.name
        rgb_files   = sorted((d / "rgb").glob("*.png"))
        depth_files = sorted((d / "depth").glob("*.png"))
        for i, (r, dep) in enumerate(zip(rgb_files, depth_files)):
            samples.append((r, dep, env_name, i))
    return samples

# --------------------------------------------------------------------------- #
#  Main processing loop                                                       #
# --------------------------------------------------------------------------- #
def main():

    BATCH_SIZE = 128

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    if args.save or args.pca:
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        world_to_cam_poses = load_w2c_poses_from_dir(args.poses_dir)
    except FileNotFoundError:
        print(f"[ERROR] Poses directory '{args.poses_dir}' not found or is empty.")
        return

    # For PCA visualization, we need camera-to-world transformations
    cam_to_world_poses_list = []
    for w2c_3x4 in world_to_cam_poses:
        c2w = np.eye(4)
        R = w2c_3x4[:3, :3]
        t = w2c_3x4[:3, 3]
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        cam_to_world_poses_list.append(c2w)
    cam_to_world_poses = np.array(cam_to_world_poses_list)


    env_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('env_')])

    agg_coords = defaultdict(list)

    env_dirs   = sorted([d for d in dataset_path.iterdir()
                     if d.is_dir() and d.name.startswith("env_")])

    all_grids = {d.name: VoxelHashTable(
                    resolution=0.12,
                    num_levels=GRID_LVLS,
                    feature_dim=GRID_FEAT_DIM,
                    scene_bound_min=SCENE_MIN,
                    scene_bound_max=SCENE_MAX,
                    device=DEVICE,
                    mode="train") for d in env_dirs}

    # MARK: One optimizer for all envs
    optim_params = [p for g in all_grids.values() for p in g.parameters()] \
                + list(decoder.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=OPT_LR)

    for shard_id, shard_dirs in enumerate(chunk(env_dirs, BATCH_SIZE)):
        print(f"\n=== SHARD {shard_id}: {len(shard_dirs)} envs {', '.join(d.name for d in shard_dirs)} ===")

        grids = build_grids(shard_dirs, device=DEVICE)

        shard_samples = collect_samples(shard_dirs)
        if not shard_samples:
            print("  [WARN] no samples, skipping shard")
            continue
        dataset = MultiEnvDataset(shard_samples, world_to_cam_poses, transform)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                              num_workers=4, pin_memory=True, prefetch_factor=2)

        # ---------- Training loop (unchanged except env lookup) ----------
        for epoch in range(args.epochs):
            running = deque(maxlen=50)          # rolling window for smoother tqdm
            epoch_losses = []

            # tqdm over dataloader
            pbar = tqdm(loader, desc=f"[Shard {shard_id} | Epoch {epoch+1}/{args.epochs}]",
                        leave=False, dynamic_ncols=True)

            for data in pbar:
                env_name = data["env_name"][0]
                grid = grids[env_name]

                img_tensor = data["img_tensor"].to(DEVICE, non_blocking=True)
                depth_t = data["depth_t"].to(DEVICE, non_blocking=True)
                depth_t = data["depth_t"].unsqueeze(0).to(DEVICE, non_blocking=True)
                extrinsic_t = data["extrinsic_t"].unsqueeze(0).unsqueeze(0).to(DEVICE, non_blocking=True)

                with torch.no_grad():
                    vis_feat = get_visual_features(clip_model, img_tensor)

                coords_world, _ = get_3d_coordinates(
                    depth_t, extrinsic_t, fx=154.1548, fy=154.1548, cx=112, cy=112,
                )

                feats = vis_feat.permute(0, 2, 3, 1).reshape(-1, vis_feat.size(1))
                pts   = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)

                mask = (
                    (pts[:,0] >= SCENE_MIN[0]) & (pts[:,0] <= SCENE_MAX[0]) &
                    (pts[:,1] >= SCENE_MIN[1]) & (pts[:,1] <= SCENE_MAX[1]) &
                    (pts[:,2] >= SCENE_MIN[2]) & (pts[:,2] <= SCENE_MAX[2])
                )
                if mask.sum() == 0:
                    pbar.set_postfix_str("skip-out-of-bounds")
                    continue

                feats = feats[mask].to(DEVICE, non_blocking=True)
                pts   = pts[mask].to(DEVICE, non_blocking=True)

                voxel_feat = grid.query_voxel_feature(pts)
                pred_feat  = decoder(voxel_feat)
                loss       = 1.0 - F.cosine_similarity(pred_feat, feats, dim=-1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.pca or args.vis_fine_grid:
                    # For now grab 2000 points
                    if pts.size(0) > 0:
                        keep = torch.randperm(pts.size(0))[:2000]
                        agg_coords[env_name].append(pts[keep].cpu())

                # --- tracking ---
                loss_val = loss.item()
                running.append(loss_val)
                epoch_losses.append(loss_val)
                pbar.set_postfix(loss=f"{mean(running):.4f}")

            # end dataloader loop
            avg_epoch_loss = mean(epoch_losses) if epoch_losses else float("nan")
            print(f"Shard {shard_id} | Epoch {epoch+1}/{args.epochs}  "
                f"avg-loss={avg_epoch_loss:.4f}")
            # ----------------------------------------------------------------------- #
        #  Check hash collisions in infer mode                                     #
        # ----------------------------------------------------------------------- #
        print("\n[CHECK] Evaluating hash collisions in infer mode for each environment...")
        for env_name, grid in grids.items():
            sparse_data = grid.export_sparse()
            infer_grid = VoxelHashTable(mode="infer", sparse_data=sparse_data, device=DEVICE)
            stats = infer_grid.collision_stats()
            print(f"  [Infer] {env_name}:")
            for level_name, stat in stats.items():
                total = stat['total']
                collisions = stat['col']
                if total > 0:
                    percentage = (collisions / total) * 100
                    print(f"    {level_name}: {collisions} collisions out of {total} voxels ({percentage:.2f}%)")
                else:
                    print(f"    {level_name}: 0 voxels")

        if args.save:
            for env_name, grid in grids.items():
                dense_grid_path = output_path / f"{env_name}_grid.pt"
                sparse_grid_path = output_path / f"{env_name}_grid.sparse.pt"
                grid.save_dense(dense_grid_path)
                grid.save_sparse(sparse_grid_path)
                print(f"[SAVE] Saved grid for {env_name} to {dense_grid_path} and {sparse_grid_path}")

            decoder_path = output_path / "shared_decoder.pt"
            torch.save(decoder.state_dict(), decoder_path)
            print(f"[SAVE] Saved shared decoder to {decoder_path}")
        
        # --- PCA Visualization ---
        if args.pca:
            intrinsic_path = dataset_path / "intrinsic.txt"
            if not intrinsic_path.exists():
                print(f"[VIS] [ERROR] Intrinsic file not found at {intrinsic_path}. Skipping PCA.")
            else:
                K = np.loadtxt(intrinsic_path)
                for env_dir_name in args.envs:
                    if env_dir_name not in grids:
                        print(f"[VIS] Grid for {env_dir_name} not found, skipping PCA.")
                        continue
                    env_dir = dataset_path / env_dir_name
                    grid = grids[env_dir_name]
                    print(f"\n[VIS] Running PCA on voxel features for {env_dir_name} ...")

                    rgb_files = sorted(list((env_dir / "rgb").glob("*.png")))
                    depth_files = sorted(list((env_dir / "depth").glob("*.png")))

                    if not rgb_files or not depth_files:
                        print(f"[VIS] No image data found in {env_dir_name}, skipping.")
                    else:
                        pcds = []
                        img_for_shape = cv2.imread(str(rgb_files[0]))
                        H, W, _ = img_for_shape.shape
                        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0,0], K[1,1], K[0,2], K[1,2])

                        for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
                            rgb_np = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
                            depth_np = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                            
                            if np.max(depth_np) == 0: continue

                            rgb_o3d = o3d.geometry.Image(rgb_np)
                            depth_o3d = o3d.geometry.Image(depth_np)
                            
                            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                            )
                            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                            if pcd.has_points():
                                pcd.transform(cam_to_world_poses[i])
                                pcds.append(pcd)
                        
                        if not pcds:
                            print("[VIS] No points generated for PCA. Skipping.")
                        else:
                            aggregated_pcd = sum(pcds, o3d.geometry.PointCloud())
                            vertices_np = np.asarray(aggregated_pcd.points).astype(np.float32)

                            ds_size = 0.05
                            voxel_idx = np.floor(vertices_np / ds_size).astype(np.int32)
                            _, uniq = np.unique(voxel_idx, axis=0, return_index=True)
                            vertices_ds = vertices_np[uniq]

                            in_x = (vertices_ds[:,0] >= SCENE_MIN[0]) & (vertices_ds[:,0] <= SCENE_MAX[0])
                            in_y = (vertices_ds[:,1] >= SCENE_MIN[1]) & (vertices_ds[:,1] <= SCENE_MAX[1])
                            in_z = (vertices_ds[:,2] >= SCENE_MIN[2]) & (vertices_ds[:,2] <= SCENE_MAX[2])
                            in_bounds = in_x & in_y & in_z
                            vertices_vis = vertices_ds[in_bounds]
                            
                            if vertices_vis.shape[0] == 0:
                                print("[VIS] No vertices inside grid bounds; skipping PCA visualization.")
                            else:
                                coords_t = torch.from_numpy(vertices_vis).to(DEVICE)
                                with torch.no_grad():
                                    voxel_feat = grid.query_voxel_feature(coords_t)
                                    feats_t    = decoder(voxel_feat)
                                
                                feats_np = feats_t.cpu().numpy()
                                pca = PCA(n_components=3)
                                feats_pca = pca.fit_transform(feats_np)
                                scaler = MinMaxScaler()
                                feats_pca_norm = scaler.fit_transform(feats_pca)
                                
                                p_fig = go.Figure()
                                p_fig.add_trace(go.Scatter3d(
                                    x=vertices_vis[:,0], y=vertices_vis[:,1], z=vertices_vis[:,2],
                                    mode="markers",
                                    marker=dict(
                                        size=10,
                                        color=[f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r,g,b in feats_pca_norm],
                                        opacity=0.8)
                                ))
                                p_fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,b=0,t=0))
                                
                                PCA_HTML = output_path / f"{env_dir_name}_pca.html"
                                pyo.plot(p_fig, filename=str(PCA_HTML), auto_open=False)
                                print(f"[VIS] PCA visualization for {env_dir_name} saved to {PCA_HTML}")


        # --- Fine-level Voxel Grid Visualization ---
        if args.vis_fine_grid:
            for env_dir_name in args.envs:
                if env_dir_name not in grids:
                    print(f"[VIS-GRID] Grid for {env_dir_name} not found, skipping visualization.")
                    continue
                grid = grids[env_dir_name]
                print(f"\n[VIS-GRID] Visualizing finest voxel grid for {env_dir_name} ...")
                
                # The finest level is the last one in the list as they are sorted coarse to fine.
                # However, during training setup, they are created from coarse to fine, 
                # but the resolution is calculated as `resolution * (level_scale ** (num_levels - 1 - lv))`.
                # This means lv=0 is the coarsest. So we want the last level, lv = num_levels - 1.
                finest_level = grid.levels[-1]
                
                # Get only the vertices that were accessed during training.
                accessed_indices = finest_level.get_accessed_indices()
                
                print(f"[VIS-GRID] Found {len(accessed_indices)} accessed vertices in the finest grid level.")

                if len(accessed_indices) == 0:
                    print("[VIS-GRID] No accessed vertices to visualize. Skipping.")
                else:
                    all_coords = finest_level.coords
                    vertices = all_coords[accessed_indices].cpu().numpy()
                

                    fig = go.Figure()

                    # Plot all coords_valid encountered (red)
                    if env_dir_name in agg_coords and len(agg_coords[env_dir_name]) > 0:
                        all_coords_vis = torch.cat(agg_coords[env_dir_name], dim=0).numpy()
                        fig.add_trace(go.Scatter3d(
                            x=all_coords_vis[:,0], y=all_coords_vis[:,1], z=all_coords_vis[:,2],
                            mode="markers",
                            marker=dict(size=2, color='red', opacity=0.6),
                            name="All coords_valid"
                        ))

                    # Plot accessed vertices (green)
                    fig.add_trace(go.Scatter3d(
                        x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                        mode="markers",
                        marker=dict(size=2, color='green', opacity=0.8),
                        name="Accessed vertices"
                    ))
                    fig.update_layout(
                        title=f"Accessed Fine-level Voxel Vertices for {env_dir_name}",
                        scene=dict(aspectmode="data"),
                        margin=dict(l=0,r=0,b=0,t=0)
                    )
                    
                    VIS_HTML_PATH = output_path / f"{env_dir_name}_accessed_fine_grid.html"
                    pyo.plot(fig, filename=str(VIS_HTML_PATH), auto_open=False)
                    print(f"[VIS-GRID] Visualization of accessed vertices saved to {VIS_HTML_PATH}")

        del grids, optimizer, dataset, loader
        torch.cuda.empty_cache()

    print("\nDone.")

if __name__ == "__main__":
    main()
