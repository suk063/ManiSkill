import torch

def get_3d_coordinates(
    depth: torch.Tensor,
    camera_extrinsic: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    original_size,
):
    """
    Computes 3D coordinates from 2D feature maps and depth.
    If camera_extrinsic is provided, returns (coords_world, coords_cam).
    Otherwise, returns coords_cam.
    Args:
        feature_maps: [B, C, H_feat, W_feat].
        depth: [B, H_feat, W_feat] or [B, 1, H_feat, W_feat].
        camera_extrinsic: [B, 1, 3, 4], world->cam transform (None if absent).
        fx, fy, cx, cy: Camera intrinsics for original_size x original_size.
        original_size: Original image size (default=224).
    Returns:
        coords_world or coords_cam: [B, 3, H_feat, W_feat].
    """
    device = depth.device

    # Adjust depth shape if needed
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    
    B, H_feat, W_feat = depth.shape  

    if isinstance(original_size, int):
        original_h, original_w = original_size, original_size
    else:
        original_h, original_w = original_size

    # Scale intrinsics
    scale_x = W_feat / float(original_w)
    scale_y = H_feat / float(original_h)
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    # Create pixel coordinate grid
    u = torch.arange(W_feat, device=device).view(1, -1).expand(H_feat, W_feat) + 0.5
    v = torch.arange(H_feat, device=device).view(-1, 1).expand(H_feat, W_feat) + 0.5
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    # Camera coordinates
    x_cam = (u - cx_new) * depth / fx_new
    y_cam = (v - cy_new) * depth / fy_new
    z_cam = depth
    coords_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)

    # If extrinsic is given, convert cam->world
    if camera_extrinsic is not None:
        camera_extrinsic = camera_extrinsic.squeeze(1)  # [B, 3, 4]
        ones_row = torch.tensor(
            [0, 0, 0, 1], device=device, dtype=camera_extrinsic.dtype
        ).view(1, 1, 4)
        ones_row = ones_row.expand(B, 1, 4)  # [B, 1, 4]
        extrinsic_4x4 = torch.cat([camera_extrinsic, ones_row], dim=1)  # [B, 4, 4]
        extrinsic_inv = torch.inverse(extrinsic_4x4)  # [B, 4, 4]

        _, _, Hf, Wf = coords_cam.shape
        ones_map = torch.ones(B, 1, Hf, Wf, device=device)
        coords_hom = torch.cat([coords_cam, ones_map], dim=1)  # [B, 4, Hf, Wf]
        coords_hom_flat = coords_hom.view(B, 4, -1)  # [B, 4, Hf*Wf]

        world_coords_hom = torch.bmm(extrinsic_inv, coords_hom_flat)  # [B, 4, Hf*Wf]
        world_coords_hom = world_coords_hom.view(B, 4, Hf, Wf)
        coords_world = world_coords_hom[:, :3, :, :]
        return coords_world, coords_cam
    else:
        return coords_cam
