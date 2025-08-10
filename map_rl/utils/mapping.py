import torch
import torch.nn.functional as F

from mapping.mapping_lib.utils import get_visual_features, get_3d_coordinates
from torchvision import transforms
transform = transforms.Compose(
    [
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
) 

def update_map_online(obs, sensor_param, grids, clip_model, decoder, map_optimizer, args):
    """
    Update voxel grids online using a single optimizer for all maps and fully batched processing.
    """
    if not args.use_online_mapping or not grids:
        return None

    num_envs = len(grids)
    robot_ids_tensor = torch.tensor(args.robot_segmentation_id, device=args.device)

    # --- 1. Data Collection and Filtering ---
    all_coords_valid = []
    all_feats_valid = []
    valid_env_indices = []

    for i in range(num_envs):
        # Prepare inputs for a single environment
        rgb = obs["rgb"][i].permute(2, 0, 1)
        depth = obs["depth"][i]
        segmentation = obs["segmentation"][i]
        
        # Camera intrinsics and extrinsics from sensor_param
        cam_params = sensor_param['hand_camera']
        extrinsic_cv = cam_params['extrinsic_cv'][i]
        intrinsic_cv = cam_params['intrinsic_cv'][i]
        
        fx, fy = intrinsic_cv[0, 0], intrinsic_cv[1, 1]
        cx, cy = intrinsic_cv[0, 2], intrinsic_cv[1, 2]
        extrinsic_t = extrinsic_cv.unsqueeze(0).unsqueeze(0)

        # Preprocess image and extract features
        img_tensor = transform(rgb / 255.0).unsqueeze(0).to(args.device)
        with torch.no_grad():
            vis_feat = get_visual_features(clip_model, img_tensor)
        
        B, C_, Hf, Wf = vis_feat.shape

        # Get 3D coordinates from depth
        depth_t = depth.permute(2, 0, 1).unsqueeze(0)
        depth_t = F.interpolate(depth_t / 1000.0, (Hf, Wf), mode="nearest-exact")
        coords_world, _ = get_3d_coordinates(depth_t, extrinsic_t, fx=fx, fy=fy, cx=cx, cy=cy)

        feats_flat = vis_feat.permute(0, 2, 3, 1).reshape(-1, C_)
        coords_flat = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)

        # Filter out robot pixels
        segmentation_t = segmentation.permute(2, 0, 1).unsqueeze(0).float()
        segmentation_t = F.interpolate(segmentation_t, (Hf, Wf), mode="nearest-exact").long().squeeze()
        is_robot = torch.isin(segmentation_t, robot_ids_tensor)
        non_robot_mask = ~is_robot.reshape(-1)
        
        if non_robot_mask.sum() == 0:
            continue

        coords_non_robot = coords_flat[non_robot_mask]
        feats_non_robot = feats_flat[non_robot_mask]

        # Filter out-of-bounds coordinates

        scene_min, scene_max = grids[i].get_scene_bounds()
        in_x = (coords_non_robot[:, 0] >= scene_min[0]) & (coords_non_robot[:, 0] <= scene_max[0])
        in_y = (coords_non_robot[:, 1] >= scene_min[1]) & (coords_non_robot[:, 1] <= scene_max[1])
        in_z = (coords_non_robot[:, 2] >= scene_min[2]) & (coords_non_robot[:, 2] <= scene_max[2])
        in_bounds = in_x & in_y & in_z

        if in_bounds.sum() > 0:
            all_coords_valid.append(coords_non_robot[in_bounds])
            all_feats_valid.append(feats_non_robot[in_bounds])
            valid_env_indices.append(i)

    if not valid_env_indices:
        return

    # --- 2. Batched Optimization Loop ---
    for _ in range(args.online_map_update_steps):
        
        # Collect voxel features from all valid environments into a single batch
        voxel_features_list = [grids[i].query_voxel_feature(all_coords_valid[idx]) for idx, i in enumerate(valid_env_indices)]
        target_features_list = [all_feats_valid[idx] for idx in range(len(valid_env_indices))]

        if not voxel_features_list:
            continue

        voxel_features_batch = torch.cat(voxel_features_list, dim=0)
        target_features_batch = torch.cat(target_features_list, dim=0)

        # Single forward pass through the decoder
        pred_features_batch = decoder(voxel_features_batch)

        # Compute a single loss for the entire batch
        cos_sim = F.cosine_similarity(pred_features_batch, target_features_batch, dim=-1)
        loss = 1.0 - cos_sim.mean()
        
        # Single backward pass and optimizer step
        map_optimizer.zero_grad()
        loss.backward()
        map_optimizer.step()